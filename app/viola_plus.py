"""Network architectures for the ISLES'26 submission.

Two model families are defined here:
  - ResidualViolaPlusUNet: ResNet-D residual encoder + attention-gated decoder
    (tri-axial global attention in deep stages, boost-only local spatial gate
    in shallow stages).
  - ResidualEncoderUNet: stock nnU-Net ResEnc-L architecture, reassembled from
    local blocks (ResidualEncoder + model_arch.UNetDecoder); loads checkpoints
    trained with dynamic_network_architectures (0 missing / 0 unexpected keys,
    forward parity verified).

Both attention modules implement reset_parameters() so that generic weight
re-initialization (He/Kaiming sweeps) cannot wipe their identity-initialized
gating parameters.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_arch import (
    InitWeights_He,
    StackedConvBlocks,
    UNetDecoder,
    get_matching_convtransp,
    get_matching_pool_op,
    maybe_convert_scalar_to_list,
    ConvDropoutNormReLU,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0. ResNet-D Helpers & Residual Encoder
# ─────────────────────────────────────────────────────────────────────────────

def make_divisible(v: int, divisor: int = 8, min_value: Optional[int] = None, round_limit: float = 0.9) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, conv_op, rd_ratio: float = 1. / 16, rd_channels: Optional[int] = None,
                 rd_divisor: int = 8, add_maxpool: bool = False, act_layer=nn.ReLU,
                 norm_layer=None, gate_layer=nn.Sigmoid):
        super().__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = conv_op(channels, rd_channels, kernel_size=1, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)
        self.fc2 = conv_op(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = gate_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean(tuple(range(2, x.ndim)), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax(tuple(range(2, x.ndim)), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class BasicBlockD(nn.Module):
    def __init__(self, conv_op, input_channels: int, output_channels: int, kernel_size, stride,
                 conv_bias: bool = False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 stochastic_depth_p: float = 0.0, squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size,
                                         stride, conv_bias, norm_op, norm_op_kwargs,
                                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size,
                                         1, conv_bias, norm_op, norm_op_kwargs, None, None, None, None)
        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else (lambda x: x)

        self.apply_stochastic_depth = stochastic_depth_p != 0.0
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio)

        has_stride = any(i != 1 for i in stride)
        requires_projection = input_channels != output_channels
        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1,
                                               False, norm_op, norm_op_kwargs, None, None, None, None))
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)


class BottleneckD(nn.Module):
    def __init__(self, conv_op, input_channels: int, bottleneck_channels: int, output_channels: int,
                 kernel_size, stride, conv_bias: bool = False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 stochastic_depth_p: float = 0.0, squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bottleneck_channels = bottleneck_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, bottleneck_channels, 1, 1,
                                         conv_bias, norm_op, norm_op_kwargs, None, None,
                                         nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, bottleneck_channels, bottleneck_channels,
                                         kernel_size, stride, conv_bias, norm_op, norm_op_kwargs,
                                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv3 = ConvDropoutNormReLU(conv_op, bottleneck_channels, output_channels, 1, 1,
                                         conv_bias, norm_op, norm_op_kwargs, None, None, None, None)
        self.nonlin3 = nonlin(**nonlin_kwargs) if nonlin is not None else (lambda x: x)

        self.apply_stochastic_depth = stochastic_depth_p != 0.0
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio)

        has_stride = any(i != 1 for i in stride)
        requires_projection = input_channels != output_channels
        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1,
                                               False, norm_op, norm_op_kwargs, None, None, None, None))
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin3(out)


def init_last_bn_before_add_to_0(module: nn.Module):
    if isinstance(module, BasicBlockD):
        module.conv2.norm.weight = nn.init.constant_(module.conv2.norm.weight, 0)
        module.conv2.norm.bias = nn.init.constant_(module.conv2.norm.bias, 0)
    elif isinstance(module, BottleneckD):
        module.conv3.norm.weight = nn.init.constant_(module.conv3.norm.weight, 0)
        module.conv3.norm.bias = nn.init.constant_(module.conv3.norm.bias, 0)


class StackedResidualBlocks(nn.Module):
    def __init__(self, n_blocks: int, conv_op, input_channels: int, output_channels: Union[int, List[int]],
                 kernel_size, initial_stride, conv_bias: bool = False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 block=BasicBlockD, bottleneck_channels=None, stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False, squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        assert n_blocks > 0
        assert block in [BasicBlockD, BottleneckD]
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks

        if block == BasicBlockD:
            blocks = nn.Sequential(
                block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride,
                      conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                      squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1,
                        conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        else:
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0],
                      kernel_size, initial_stride, conv_bias, norm_op, norm_op_kwargs,
                      dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                      stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n],
                        kernel_size, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                        dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        self.blocks = blocks
        self.output_channels = output_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResidualEncoder(nn.Module):
    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Sequence[int],
                 conv_op, kernel_sizes, strides, n_blocks_per_stage: Sequence[int],
                 conv_bias: bool = False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 block=BasicBlockD, bottleneck_channels=None, return_skips: bool = False,
                 disable_default_stem: bool = False, stem_channels: Optional[int] = None, pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0, squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels,
                                          kernel_sizes[0], 1, conv_bias, norm_op, norm_op_kwargs,
                                          dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1
            stage = StackedResidualBlocks(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s],
                kernel_sizes[s], stride_for_conv, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block=block,
                bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation, squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Learnable Hybrid Axis Pooling
# ─────────────────────────────────────────────────────────────────────────────

class ViolaAxisPool(nn.Module):
    """Hybrid Avg + Max Pooling with Learnable Softmax Weighting.

    Pure AvgPool dilutes tiny lacunar targets over large spatial planes. MaxPool retains peaks.
    `alpha` is parameterized via Softmax and initialized to zeros, yielding exact 0.5/0.5
    at startup while allowing the network to adaptively learn peak vs statistic ratios.
    """
    def __init__(self, output_size: Tuple[Optional[int], Optional[int], Optional[int]]):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(output_size)
        self.max = nn.AdaptiveMaxPool3d(output_size)
        # Learnable Softmax parameter: initialized to [0.0, 0.0] -> Softmax yields [0.5, 0.5]
        self.alpha = nn.Parameter(torch.zeros(2))
        self.reset_parameters()

    def reset_parameters(self):
        """Protected reset for ViolaAxisPool learnable alpha weights."""
        nn.init.constant_(self.alpha, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.alpha, dim=0)
        return w[0] * self.avg(x) + w[1] * self.max(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Global Hybrid Viola Attention with Adaptive Fusion
# ─────────────────────────────────────────────────────────────────────────────

class ViolaGlobalAttention(nn.Module):
    """Global Tri-Axial Attention with Learnable Axis Softmax Fusion."""

    def __init__(self, channel: int, reduction: int = 32, min_dim: int = 4):
        super().__init__()
        self.x_pool = ViolaAxisPool((None, 1, 1))
        self.y_pool = ViolaAxisPool((1, None, 1))
        self.z_pool = ViolaAxisPool((1, 1, None))
        self.c_pool = ViolaAxisPool((1, 1, 1))

        reduced_ch = max(channel // reduction, min_dim)
        self.kan1 = self._create_sequential_block(channel, reduced_ch)
        self.kan2 = self._create_sequential_block(channel, reduced_ch)
        self.kan3 = self._create_sequential_block(channel, reduced_ch)
        self.kan4 = self._create_sequential_block(channel, reduced_ch)

        # Learnable fusion parameters across X, Y, Z, C axes.
        self.axis_weights = nn.Parameter(torch.zeros(4))
        self.reset_parameters()

    def _create_sequential_block(self, channel: int, d: int) -> nn.Sequential:
        return nn.Sequential(
            nn.GroupNorm(1, channel),
            nn.Conv3d(channel, d, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=1e-2, inplace=False),
            nn.Conv3d(d, channel, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def reset_parameters(self):
        """Reset KAN weights, axis_weights, and child pool alpha parameters.

        kan[3] is zero-mean initialized so each Tanh branch starts at ~0 and the
        fused multiplier `1.0 + softmax(w) . branches` starts as an exact identity
        map (1.0 +/- ~0.03). std=0.01 (instead of exact zeros) keeps a non-zero
        gradient path into kan[1] from the very first step.
        """
        for kan in (self.kan1, self.kan2, self.kan3, self.kan4):
            nn.init.kaiming_normal_(kan[1].weight, a=1e-2)
            nn.init.normal_(kan[3].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.axis_weights, 0.0)
        for pool in (self.x_pool, self.y_pool, self.z_pool, self.c_pool):
            pool.reset_parameters()

    def forward(self, x: torch.Tensor, gate_feat: Optional[torch.Tensor] = None, return_attn: bool = False):
        # gate_feat accepted for signature compatibility with LocalSpatialGate
        xt = self.kan1(self.x_pool(x))  # (b, c, h, 1, 1)
        yt = self.kan2(self.y_pool(x))  # (b, c, 1, w, 1)
        zt = self.kan3(self.z_pool(x))  # (b, c, 1, 1, d)
        zc = self.kan4(self.c_pool(x))  # (b, c, 1, 1, 1)

        # Adaptive Softmax Fusion across axes (starts at exactly 0.25 each)
        w = F.softmax(self.axis_weights, dim=0)
        viola_map = 1.0 + (w[0] * xt + w[1] * yt + w[2] * zt + w[3] * zc)
        out = x * viola_map

        return (out, viola_map) if return_attn else out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Non-Linear Channel-Mixed Guided Local Spatial Gate
# ─────────────────────────────────────────────────────────────────────────────

class LocalSpatialGate(nn.Module):
    """Local 3D Spatial Gate with MobileNet-style DW-PW Conv & GroupNorm Alignment.

    Combines 5x5x5 effective receptive field depthwise conv with non-linear pointwise
    channel mixing, plus GroupNorm-aligned deep semantic guidance (`gate_feat`).
    """

    def __init__(self, channels: int, gate_channels: Optional[int] = None,
                 gate_mode: str = "boost", suppress_scale: float = 1.0):
        super().__init__()
        gate_channels = gate_channels or channels
        assert gate_channels > 0, "gate_channels must be positive"
        assert gate_mode in ("boost", "bidirectional"), f"unknown gate_mode: {gate_mode}"
        # Plain python attributes — NOT in state_dict, so checkpoints trained with
        # the default ("boost") load unchanged with strict=True.
        self.gate_mode = gate_mode
        self.suppress_scale = suppress_scale
        inter_ch = max(channels // 4, 4)

        # MobileNet-style Depthwise + Non-linear Pointwise Channel Mixing
        self.local_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),  # PW Conv for Inter-Channel Communication
            nn.GroupNorm(1, channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=False),
            nn.Conv3d(channels, inter_ch, kernel_size=1, bias=False),
        )

        # GroupNorm-aligned semantic projection
        self.gate_proj = nn.Sequential(
            nn.Conv3d(gate_channels, inter_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, inter_ch),
        )
        self.psi = nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.local_conv[0].weight, a=1e-2)
        nn.init.kaiming_normal_(self.local_conv[3].weight, a=1e-2)
        nn.init.kaiming_normal_(self.local_conv[6].weight, a=1e-2)
        nn.init.kaiming_normal_(self.gate_proj[0].weight, a=1e-2)
        if self.gate_mode == "boost":
            nn.init.kaiming_normal_(self.psi.weight, a=1e-2)
            # Sigmoid(-4.0) ≈ 0.018 -> Multiplier starts near 1.018 (Identity start)
            nn.init.constant_(self.psi.bias, -4.0)
        else:
            # bidirectional: psi=0 -> raw=0 -> sigmoid=0.5 -> signed=0 -> multiplier
            # exactly 1.0 (EXACT identity start, no approximation). Gradient to
            # psi.weight is nonzero (sigmoid'(0)=0.25), same zero-init pattern as
            # init_last_bn_before_add_to_0 elsewhere in this file.
            nn.init.zeros_(self.psi.weight)
            nn.init.zeros_(self.psi.bias)

    def forward(self, x: torch.Tensor, gate_feat: Optional[torch.Tensor] = None, return_attn: bool = False):
        """If return_attn=True, returns (out, spatial_map) with spatial_map in [0,1].

        ⚠️ spatial_map semantics depend on gate_mode: in "boost" mode 0 = neutral
        (identity), 1 = max boost; in "bidirectional" mode 0.5 = neutral,
        0 = MAX SUPPRESSION, 1 = max boost. For visualization/analysis of
        bidirectional models, convert first: signed = 2*spatial_map - 1
        (negative = suppression, positive = boost). Do NOT read 0 as
        "no attention" in bidirectional mode — it means the opposite.
        """
        local = self.local_conv(x)
        if gate_feat is not None:
            gate_up = F.interpolate(gate_feat, size=x.shape[2:], mode='trilinear', align_corners=False)
            local = local + self.gate_proj(gate_up)

        spatial_map = torch.sigmoid(self.psi(F.leaky_relu(local, negative_slope=1e-2)))

        if self.gate_mode == "boost":
            out = x * (1.0 + spatial_map)  # Multiplicative boost [1.0, ~2.0]
        else:
            signed = 2.0 * spatial_map - 1.0  # [-1, 1]
            out = x * (1.0 + self.suppress_scale * signed)  # [1-s, 1+s]: can suppress

        return (out, spatial_map) if return_attn else out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dual-Stage Viola-Plus UNet Decoder
# ─────────────────────────────────────────────────────────────────────────────

class UNetViolaPlusDecoder(nn.Module):
    """Dual-Stage Decoder with Resolution-Aware Attention & Deep Supervision."""

    def __init__(self, encoder, num_classes: int, n_conv_per_stage, deep_supervision: bool,
                 nonlin_first: bool = False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 conv_bias: Optional[bool] = None,
                 gate_mode: str = "boost", suppress_scale: float = 1.0):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        # The s > 3 shallow/deep attention split below is calibrated for the 6-stage
        # ISLES plans geometry (deep Global on s=1..3, shallow Local on s=4..5).
        assert n_stages_encoder == 6, \
            f"UNetViolaPlusDecoder attention split assumes 6 encoder stages, got {n_stages_encoder}"
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        conv_blocks = []
        att_modules = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv,
                stride_for_transpconv, bias=conv_bias))

            conv_blocks.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip,
                input_features_skip, encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first))

            # Dual-stage resolution-aware attention scheduling
            if s > 3:  # Shallow, High-Res stages (64 & 32 ch) -> Local Spatial Gate
                att_modules.append(LocalSpatialGate(input_features_skip, gate_channels=input_features_below,
                                                    gate_mode=gate_mode, suppress_scale=suppress_scale))
            else:      # Deep, Low-Res stages (320, 256, 128 ch) -> Global Tri-Axial Attention
                att_modules.append(ViolaGlobalAttention(input_features_skip))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.transpconvs = nn.ModuleList(transpconvs)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.att_modules = nn.ModuleList(att_modules)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: List[torch.Tensor]):
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.conv_blocks)):
            x_transp = self.transpconvs[s](lres_input)
            x_cat = torch.cat((skips[-(s + 2)], x_transp), 1)
            x_conv = self.conv_blocks[s](x_cat)

            # Pass upsampled deep features (lres_input) as semantic gate feature
            x = self.att_modules[s](x_conv, gate_feat=lres_input)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.conv_blocks) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Full models (final submission architectures)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualViolaPlusUNet(nn.Module):
    """ResidualEncoder + UNetViolaPlusDecoder."""

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Sequence[int],
                 conv_op, kernel_sizes, strides, n_blocks_per_stage: Sequence[int], num_classes: int,
                 n_conv_per_stage_decoder, conv_bias: bool = False, norm_op=None,
                 norm_op_kwargs=None, dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, deep_supervision: bool = False,
                 nonlin_first: bool = False, pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0, squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 gate_mode: str = "boost", suppress_scale: float = 1.0):
        super().__init__()
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op,
                                       kernel_sizes, strides, n_blocks_per_stage, conv_bias,
                                       norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                       nonlin, nonlin_kwargs, return_skips=True, pool_type=pool_type,
                                       stochastic_depth_p=stochastic_depth_p,
                                       squeeze_excitation=squeeze_excitation,
                                       squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        self.decoder = UNetViolaPlusDecoder(self.encoder, num_classes, n_conv_per_stage_decoder,
                                            deep_supervision, nonlin_first=nonlin_first,
                                            gate_mode=gate_mode, suppress_scale=suppress_scale)
        # apply() recurses bottom-up, matching nnUNet's `network.apply(network.initialize)`
        # exactly -> standalone (e.g. MONAI) construction gets identical initialization.
        self.apply(self.initialize)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def initialize(module: nn.Module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
        for m in module.modules():
            if isinstance(m, (ViolaGlobalAttention, LocalSpatialGate)):
                m.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Stock ResidualEncoderUNet (nnU-Net ResEnc-L plans compatible)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualEncoderUNet(nn.Module):
    """Stock nnU-Net ResidualEncoderUNet, assembled from local blocks only:
    viola_plus.ResidualEncoder + model_arch.UNetDecoder. Loads checkpoints
    trained with dynamic_network_architectures' ResidualEncoderUNet
    (verified: 0 missing / 0 unexpected keys + forward parity).
    """

    def __init__(self, input_channels: int, n_stages: int, features_per_stage,
                 conv_op, kernel_sizes, strides, n_blocks_per_stage, num_classes: int,
                 n_conv_per_stage_decoder, conv_bias: bool = False, norm_op=None,
                 norm_op_kwargs=None, dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, deep_supervision: bool = False,
                 block=BasicBlockD, bottleneck_channels=None, stem_channels=None):
        super().__init__()
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0")
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op,
                                       kernel_sizes, strides, n_blocks_per_stage, conv_bias,
                                       norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                       nonlin, nonlin_kwargs, block=block,
                                       bottleneck_channels=bottleneck_channels, return_skips=True,
                                       stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder,
                                   deep_supervision)

    def forward(self, x):
        return self.decoder(self.encoder(x))

