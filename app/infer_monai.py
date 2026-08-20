#!/usr/bin/env python3
"""ISLES'26 MONAI inference — exact replication of nnU-Net v2 3d_fullres pipeline.

Replicates the complete nnU-Net v2 inference pipeline step by step:
  1. Preprocess: load → reorient RAS → (z,y,x) → crop to nonzero (fill holes)
     → per-case ZScore within nonzero mask → resample to 1mm iso
     (skimage resize order=3, mode='edge', separate-z for anisotropic data)
  2. Infer:      sliding-window 128³ + Gaussian + 8-fold TTA (mirror axes 0,1,2)
  3. Postprocess: resample LOGITS to cropped native shape (order=1) → softmax
     → revert cropping (bg class=1 outside bbox) → transpose → revert orientation

Key facts verified against the installed nnunetv2 2.8.1 source:
  - ZScoreNormalization is PER-CASE: mean/std computed over the current case's
    nonzero voxels (default_normalization_schemes.py). The dataset-level
    foreground_intensity_properties in plans.json are only used by CTNormalization.
  - Normalization happens BEFORE resampling (default_preprocessor.py:
    "normalization MUST happen before resampling").
  - Resampling uses skimage.transform.resize(mode='edge', anti_aliasing=False),
    with a separate-z path (in-plane resize + nearest along z, order_z=0) for
    anisotropic spacings (default_resampling.py).
  - Export resamples LOGITS to the cropped native shape first, THEN applies
    softmax, THEN reverts cropping (export_prediction.py).

Architecture: read from plans.json (`architecture.network_class_name`); stock
nnU-Net plans use the self-contained PlainConvUNet in model_arch.py (identical to
dynamic_network_architectures), custom classes (e.g. model_viola3.PlainViolaConvUNetV2)
are imported by dotted path. Checkpoint: nnU-Net v2 format, key='network_weights'.

Usage:
  # single image
  python infer_monai.py --input /path/to/t1.nii.gz --out_root preds_monai --fold 0 --gpu 1

  # batch (directory of .nii.gz or nnU-Net imagesTr layout)
  python infer_monai.py --input /path/to/imagesTr --out_root preds_monai --fold 0 --gpu 1

  # verify against nnU-Net official prediction (Dice-based, NOT whole-volume agreement)
  python infer_monai.py --input t1.nii.gz --fold 0 --gpu 1 \
      --compare /path/to/nnunet_pred/sub-xxx/soft.nii.gz
"""

import argparse
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import map_coordinates, binary_fill_holes

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

# nnU-Net PlainConvUNet architecture (self-contained, identical to dynamic_network_architectures)
from model_arch import PlainConvUNet

# Network class used to build the model; apply_plans() overrides this from
# plans.json's architecture.network_class_name (supports custom architectures
# such as model_viola3.PlainViolaConvUNetV2 / violaplus_arch.PlainViolaPlusUNet).
NET_CLASS = PlainConvUNet

# ─────────────────────────────────────────────────────────────────────────────
# Constants — from nnUNetPlans.json (3d_fullres) / nnunetv2 source
# ─────────────────────────────────────────────────────────────────────────────

# Project root (this file's directory)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Network architecture (3d_fullres, from nnUNetPlans.json)
NET_PARAMS = dict(
    input_channels        = 1,
    n_stages              = 6,
    features_per_stage    = (32, 64, 128, 256, 320, 320),
    conv_op               = torch.nn.Conv3d,
    kernel_sizes          = [(3, 3, 3)] * 6,
    strides               = [(1, 1, 1)] + [(2, 2, 2)] * 5,
    n_conv_per_stage      = (2, 2, 2, 2, 2, 2),
    num_classes           = 2,
    n_conv_per_stage_decoder = (2, 2, 2, 2, 2),
    conv_bias             = True,
    norm_op               = torch.nn.InstanceNorm3d,
    norm_op_kwargs        = {"eps": 1e-5, "affine": True},
    dropout_op            = None,
    dropout_op_kwargs     = None,
    nonlin                = torch.nn.LeakyReLU,
    nonlin_kwargs         = {"inplace": True},
    deep_supervision      = False,   # inference mode; seg_layers still built so weights load
)

TARGET_SPACING = (1.0, 1.0, 1.0)   # 3d_fullres target spacing (mm), (z, y, x) order
PATCH_SIZE     = (128, 128, 128)   # (z, y, x) order
NUM_CLASSES    = 2
ANISO_THRESHOLD = 3.0              # nnunetv2.configuration.ANISO_THRESHOLD
USE_MASK_FOR_NORM = True           # Dataset001 plans; apply_plans() overrides
CKPT_ROOT      = os.path.join(PROJECT_ROOT, "checkpoints", "Dataset001_ISLES26")


def apply_plans(plans_path: str, dataset_json_path: str = None):
    """Override NET_PARAMS / PATCH_SIZE / TARGET_SPACING / NUM_CLASSES from a
    plans.json (as nnUNetPredictor does). Makes this script work for ANY
    nnU-Net v2 plans — e.g. Dataset002_ISLES26_RAS1x1x3 with anisotropic
    patch/strides and target spacing (3,1,1) — without editing code.

    dataset_json defaults to dataset.json next to plans.json (nnU-Net results
    folder layout).
    """
    global NET_PARAMS, NET_CLASS, PATCH_SIZE, TARGET_SPACING, NUM_CLASSES, USE_MASK_FOR_NORM
    import importlib
    import json as _json

    plans = _json.load(open(plans_path))
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]["arch_kwargs"]
    class_name = cfg["architecture"].get("network_class_name", "")

    def imp(dotted):
        mod, name = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(mod), name)

    if dataset_json_path is None:
        dataset_json_path = os.path.join(os.path.dirname(plans_path), "dataset.json")
    ds = _json.load(open(dataset_json_path))

    # Generic arch_kwargs passthrough (works for Plain/Residual/custom classes):
    # copy plans arch_kwargs verbatim, resolve the *_requires_import entries,
    # then add input/num_classes + inference-mode deep_supervision.
    NET_PARAMS = dict(arch)
    for k in cfg["architecture"].get("_kw_requires_import",
                                     ["conv_op", "norm_op", "dropout_op", "nonlin"]):
        if NET_PARAMS.get(k) is not None:
            NET_PARAMS[k] = imp(NET_PARAMS[k])
    NET_PARAMS["input_channels"] = len(ds["channel_names"])
    NET_PARAMS["num_classes"] = len(ds["labels"])
    NET_PARAMS["deep_supervision"] = False  # inference; seg_layers still built so weights load

    # Custom network class (e.g. viola variants living in this repo); fall back
    # to the built-in PlainConvUNet for the stock nnU-Net class path.
    # ResidualEncoderUNet routes to the local viola_plus implementation
    # (state-dict + forward-parity verified against dynamic_network_architectures).
    if class_name.endswith("ResidualEncoderUNet"):
        from viola_plus import ResidualEncoderUNet
        NET_CLASS = ResidualEncoderUNet
    elif class_name and not class_name.startswith("dynamic_network_architectures"):
        NET_CLASS = imp(class_name)
    else:
        NET_CLASS = PlainConvUNet
    # This inference core assumes plans stored in (z,y,x) transposed space
    # (identity transpose). All 15 ISLES'26 plans satisfy this, but it is luck,
    # not a design guarantee — fail loudly instead of silently mis-axis-ing.
    tf = plans.get("transpose_forward", [0, 1, 2])
    if list(tf) != [0, 1, 2]:
        raise NotImplementedError(
            f"plans transpose_forward={tf} not supported by this inference core")
    PATCH_SIZE = tuple(cfg["patch_size"])          # plans are in (z, y, x) transposed space
    TARGET_SPACING = tuple(cfg["spacing"])         # same
    NUM_CLASSES = len(ds["labels"])
    USE_MASK_FOR_NORM = bool(cfg["use_mask_for_norm"][0])
    print(f"Applied plans: {plans_path}")
    print(f"  network={NET_CLASS.__module__}.{NET_CLASS.__name__}  "
          f"spacing(zyx)={TARGET_SPACING}  patch(zyx)={PATCH_SIZE}  "
          f"stages={arch['n_stages']}  features={tuple(arch['features_per_stage'])}  "
          f"use_mask_for_norm={USE_MASK_FOR_NORM}")


# ─────────────────────────────────────────────────────────────────────────────
# nnU-Net resampling — exact replication of default_resampling.py (is_seg=False)
#
# nnU-Net uses skimage.transform.resize(order, mode='edge', anti_aliasing=False)
# for images AND probabilities. For anisotropic spacings (ratio > 3) it resizes
# in-plane per slice (order) and then along the low-res axis with map_coordinates
# (order_z=0 → nearest, align_corners=False grid).
# ─────────────────────────────────────────────────────────────────────────────

def _determine_sep_z(current_spacing, new_spacing):
    """Replicates determine_do_sep_z_and_axis with force_separate_z=None."""
    def aniso(sp):
        return np.max(sp) / np.min(sp) > ANISO_THRESHOLD

    def lowres_axis(sp):
        return np.where(max(sp) / np.array(sp) == 1)[0]

    if aniso(current_spacing):
        axis = lowres_axis(current_spacing)
    elif aniso(new_spacing):
        axis = lowres_axis(new_spacing)
    else:
        return False, None
    if len(axis) >= 2:  # 2 or 3 low-res axes → no separate z (nnU-Net behavior)
        return False, None
    return True, int(axis[0])


def compute_new_shape(old_shape, old_spacing, new_spacing):
    """Replicates nnunetv2 compute_new_shape."""
    out = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    # sanity bounds: a misread spacing (e.g. meters vs mm) would otherwise
    # explode memory (blow-up) or collapse an axis to zero (shrink-to-zero).
    # 20x per axis is far beyond any legitimate medical scan.
    if (out > 20 * np.asarray(old_shape)).any() or (out < 1).any():
        raise ValueError(f"resample target {tuple(out)} implausible for input {tuple(old_shape)} "
                         f"— spacing likely misread ({old_spacing} -> {new_spacing})")
    return out


def nnunet_resample_nonseg(data: np.ndarray, new_shape, order: int,
                           current_spacing, new_spacing, order_z: int = 0) -> np.ndarray:
    """Replicates resample_data_or_seg(..., is_seg=False) for (C, z, y, x) arrays."""
    from skimage.transform import resize

    shape = np.array(data.shape[1:])
    new_shape = np.array(new_shape)
    if np.all(shape == new_shape):
        return data.astype(np.float32, copy=False)

    do_sep, axis = _determine_sep_z(current_spacing, new_spacing)
    out = np.zeros((data.shape[0], *new_shape), dtype=np.float32)
    data = data.astype(float, copy=False)  # nnU-Net casts to float before resizing

    if do_sep:
        axes_2d = [i for i in range(3) if i != axis]
        new_shape_2d = tuple(new_shape[axes_2d])
        tmp_shape = tuple(new_shape[:axis]) + (shape[axis],) + tuple(new_shape[axis + 1:])
        for c in range(data.shape[0]):
            reshaped_here = np.zeros(tmp_shape, dtype=float)
            for slice_id in range(shape[axis]):
                idx = [slice(None)] * 3
                idx[axis] = slice_id
                reshaped_here[tuple(idx)] = resize(
                    data[(c, *idx)], new_shape_2d, order, mode='edge', anti_aliasing=False)
            if shape[axis] != new_shape[axis]:
                # align_corners=False grid along all axes (copied from nnU-Net / skimage)
                rows, cols, dim = new_shape
                orig_rows, orig_cols, orig_dim = reshaped_here.shape
                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim
                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                coord_map = np.array([
                    row_scale * (map_rows + 0.5) - 0.5,
                    col_scale * (map_cols + 0.5) - 0.5,
                    dim_scale * (map_dims + 0.5) - 0.5,
                ])
                out[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')
            else:
                out[c] = reshaped_here
    else:
        for c in range(data.shape[0]):
            out[c] = resize(data[c], tuple(new_shape), order, mode='edge', anti_aliasing=False)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — exact nnU-Net DefaultPreprocessor replication (inference path)
#
# nnU-Net order (default_preprocessor.run_case_npy):
#   1. transpose (z,y,x)
#   2. crop_to_nonzero: mask = binary_fill_holes(any channel != 0), crop to bbox
#   3. normalize BEFORE resampling: per-case ZScore within the nonzero mask,
#      voxels outside the mask stay exactly 0
#   4. resample data (order=3, mode='edge', separate-z if anisotropic)
#
# We additionally reorient to RAS first (orientation safeguard for non-RAS
# test data); orientation is reverted in postprocessing.
# ─────────────────────────────────────────────────────────────────────────────

_load_orient = Compose([
    LoadImaged(keys="image", reader="NibabelReader"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
])


def preprocess(image_path: str):
    """Returns (data_1mm_zyx, info_dict) — data is (C, z, y, x) float32 at 1mm."""
    # Input hardening beyond nnU-Net (which does almost none). Detection +
    # safe fallback only — behavior on clean inputs is unchanged.
    raw = nib.load(image_path)
    if raw.ndim != 3:
        raise ValueError(f"input ndim={raw.ndim} (shape {raw.shape}) not supported "
                         f"— nnU-Net asserts ndim==3; a hard fail is a visible, "
                         f"resubmittable error, a silent channel-misread is not")
    d = _load_orient({"image": image_path})
    img = d["image"]  # MetaTensor (C, x, y, z) in RAS
    affine = np.asarray(img.affine)
    spacing_xyz = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # pixdim vs affine consistency (nnU-Net reads pixdim, we read affine; a
    # mismatch silently resamples to a wrong geometry — detect, keep affine)
    zooms = np.asarray(raw.header.get_zooms()[:3], dtype=float)
    if not np.allclose(np.sort(zooms), np.sort(spacing_xyz), rtol=0.01):
        print(f"[preproc] WARNING: header pixdim {tuple(zooms)} != affine spacing "
              f"{tuple(np.round(spacing_xyz, 4))}; using affine")

    # (C, x, y, z) → (C, z, y, x), float32
    data = np.asarray(img, dtype=np.float32).transpose(0, 3, 2, 1)
    spacing_zyx = spacing_xyz[::-1].copy()
    if not (spacing_zyx > 0).all():
        raise ValueError(f"non-positive spacing {spacing_zyx}")

    # non-finite voxels would poison the masked ZScore (nan mean/std -> whole
    # volume NaN -> silently garbage output on a one-shot final). Zeroing puts
    # them outside the nonzero mask, consistent with crop_to_nonzero semantics.
    n_bad = int((~np.isfinite(data)).sum())
    if n_bad:
        print(f"[preproc] WARNING: {n_bad} non-finite voxels -> zeroed")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    shape_before_cropping = data.shape[1:]

    # crop_to_nonzero: mask = binary_fill_holes(any channel != 0)
    mask = (data != 0).any(axis=0)
    mask = binary_fill_holes(mask)
    if mask.any():
        coords = np.argwhere(mask)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
    else:  # all-zero image: keep full
        lo = np.zeros(3, dtype=int)
        hi = np.array(mask.shape)
    bbox = [[int(l), int(h)] for l, h in zip(lo, hi)]
    slicer = tuple(slice(l, h) for l, h in bbox)
    data = data[(slice(None), *slicer)]
    mask = mask[slicer]

    shape_after_cropping = data.shape[1:]

    # per-case ZScore (BEFORE resampling). Two modes, from plans' use_mask_for_norm:
    #  True  (Dataset001): mean/std over nonzero-brain voxels only; outside stays 0
    #  False (Dataset002): mean/std over the whole cropped volume (nnU-Net default
    #                      when cropping does not shrink the FOV much)
    if USE_MASK_FOR_NORM:
        if mask.any():
            mean = data[:, mask].mean()
            std = data[:, mask].std()
            data[:, mask] = (data[:, mask] - mean) / max(std, 1e-8)
    else:
        mean = data.mean()
        std = data.std()
        data -= mean
        data /= max(std, 1e-8)

    # resample to target spacing (order=3 for image)
    new_shape = compute_new_shape(shape_after_cropping, spacing_zyx, TARGET_SPACING)
    data = nnunet_resample_nonseg(data, new_shape, order=3,
                                  current_spacing=spacing_zyx, new_spacing=TARGET_SPACING)

    info = dict(
        bbox=bbox,
        shape_before_cropping=tuple(shape_before_cropping),   # native grid (z,y,x), RAS
        shape_after_cropping=tuple(shape_after_cropping),     # cropped native grid (z,y,x)
        spacing_zyx=tuple(spacing_zyx),                       # native spacing (z,y,x)
    )
    return data, info


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing — exact replication of
# convert_predicted_logits_to_segmentation_with_correct_shape (+ probabilities)
#
#   1. resample LOGITS to cropped native shape (order=1)
#   2. softmax → probabilities
#   3. revert cropping (background class = 1 outside bbox, lesion = 0)
#   4. transpose (z,y,x) → (x,y,z)  [+ orientation revert done by caller]
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_logits(logits: torch.Tensor, info: dict) -> np.ndarray:
    """logits: (2, z, y, x) at 1mm, cropped. Returns soft lesion probability
    map in RAS space, shape = shape_before_cropping reversed to (x,y,z)."""
    logits_np = logits.float().cpu().numpy()

    # 1. resample logits to cropped native shape (order=1, separate-z if needed)
    logits_native = nnunet_resample_nonseg(
        logits_np, info["shape_after_cropping"], order=1,
        current_spacing=TARGET_SPACING, new_spacing=info["spacing_zyx"])

    # 2. softmax (numerically stable)
    logits_native -= logits_native.max(axis=0, keepdims=True)
    exp = np.exp(logits_native)
    probs = exp / exp.sum(axis=0, keepdims=True)  # (2, z, y, x) cropped native

    # 3. revert cropping: zeros full volume, background class = 1 outside bbox
    full = np.zeros((NUM_CLASSES, *info["shape_before_cropping"]), dtype=np.float32)
    full[0] = 1.0
    bbox = info["bbox"]
    full[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = probs

    # 4. transpose (z,y,x) → (x,y,z)
    soft_xyz = full[1].transpose(2, 1, 0)
    return np.clip(soft_xyz, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(fold: int, chk: str) -> str:
    return os.path.join(CKPT_ROOT, f"fold_{fold}", f"{chk}.pth")


def load_model(fold: int, chk: str, device: torch.device) -> torch.nn.Module:
    path = ckpt_path(fold, chk)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NET_CLASS(**NET_PARAMS)
    model.load_state_dict(ckpt["network_weights"], strict=True)
    print(
        f"Loaded fold {fold} | {chk} | epoch {ckpt.get('current_epoch','?')} "
        f"| best_ema {ckpt.get('_best_ema', 0):.4f}"
    )
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window — exact nnU-Net replication
#
# nnU-Net step computation (compute_steps_for_sliding_window):
#   1. target_step = patch_size * tile_step_size
#   2. num_steps = ceil((image_size - patch_size) / target_step) + 1
#   3. actual_step = (image_size - patch_size) / (num_steps - 1)
#   4. positions = [round(actual_step * i) for i in range(num_steps)]
#
# This distributes patches EVENLY across the image, unlike MONAI's fixed-step
# approach which clamps the last position to the boundary.
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache


@lru_cache(maxsize=4)
def compute_gaussian(patch_size: tuple, sigma_scale: float = 1.0 / 8) -> torch.Tensor:
    """nnU-Net style Gaussian importance map (scipy gaussian_filter on impulse).

    Cached per patch_size — the map only depends on the patch size."""
    from scipy.ndimage import gaussian_filter
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = tuple(i // 2 for i in patch_size)
    sigmas = tuple(i * sigma_scale for i in patch_size)
    tmp[center] = 1.0
    gaussian = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    gaussian /= gaussian.max()
    gaussian[gaussian == 0] = gaussian[gaussian > 0].min()
    return torch.from_numpy(gaussian).float()


def compute_steps(image_size: tuple, patch_size: tuple, tile_step_size: float = 0.5) -> list:
    """Exact nnU-Net compute_steps_for_sliding_window."""
    target_steps = [p * tile_step_size for p in patch_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_steps, patch_size)]
    steps = []
    for dim in range(len(patch_size)):
        max_step = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step = max_step / (num_steps[dim] - 1)
        else:
            actual_step = 99999999999
        steps.append([int(np.round(actual_step * i)) for i in range(num_steps[dim])])
    return steps


def sliding_window_predict(
    model: torch.nn.Module,
    x: torch.Tensor,          # (1, C, D, H, W)
    patch_size: tuple,
    predictor_fn,             # function that takes (G,C,D,H,W) and returns (G,cls,D,H,W)
    device: torch.device,
    patch_batch: int = 1,     # number of patch positions per forward pass
    state: dict = None,       # optional shared dict — OOM-adapted patch_batch persists across cases
) -> torch.Tensor:
    """nnU-Net style sliding window with Gaussian weighting.

    Pads to at least patch_size with zeros first (as nnUNetPredictor does).
    Patch positions are processed patch_batch at a time (batched forward
    passes are mathematically identical — InstanceNorm is per-instance).
    Returns aggregated logits (1, num_classes, D, H, W), cropped back to the
    input size.
    """
    # pad to at least patch_size, CENTERED (nnU-Net pad_nd_image: content centered,
    # pad_below = diff//2, pad_above = diff//2 + diff%2)
    image_size_in = x.shape[2:]
    pad = [max(0, p - i) for p, i in zip(patch_size, image_size_in)]
    pad_before = [d // 2 for d in pad]
    pad_after = [d - b for d, b in zip(pad, pad_before)]
    if any(pad):
        # F.pad takes (last_dim_before, last_dim_after, ...)
        x = F.pad(x, (pad_before[2], pad_after[2],
                      pad_before[1], pad_after[1],
                      pad_before[0], pad_after[0]))

    image_size = x.shape[2:]
    steps = compute_steps(image_size, patch_size, tile_step_size=0.5)
    gaussian = compute_gaussian(patch_size).to(device)

    aggregated = torch.zeros((1, NUM_CLASSES, *image_size), dtype=torch.float32, device=device)
    n_predictions = torch.zeros(image_size, dtype=torch.float32, device=device)

    positions = [(z0, y0, x0) for z0 in steps[0] for y0 in steps[1] for x0 in steps[2]]

    # adaptive patch_batch: halve on CUDA OOM (safe on shared GPUs)
    pb = patch_batch if state is None else state.get("patch_batch", patch_batch)
    while True:
        try:
            with torch.no_grad():
                for i in range(0, len(positions), pb):
                    chunk = positions[i:i + pb]
                    xb = torch.cat([
                        x[:, :, z0:z0+patch_size[0], y0:y0+patch_size[1], x0:x0+patch_size[2]]
                        for z0, y0, x0 in chunk
                    ], dim=0)  # (G, C, D, H, W)
                    pbatch = predictor_fn(xb)  # (G, cls, D, H, W)
                    for j, (z0, y0, x0) in enumerate(chunk):
                        pred = pbatch[j] * gaussian
                        aggregated[0, :, z0:z0+patch_size[0], y0:y0+patch_size[1], x0:x0+patch_size[2]] += pred
                        n_predictions[z0:z0+patch_size[0], y0:y0+patch_size[1], x0:x0+patch_size[2]] += gaussian
                    del xb, pbatch
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if pb <= 1:
                raise
            pb //= 2
            print(f"[WARN] CUDA OOM in sliding window — reducing patch_batch to {pb}")
            if state is not None:
                state["patch_batch"] = pb
            aggregated.zero_()
            n_predictions.zero_()

    aggregated /= n_predictions
    if any(pad):
        aggregated = aggregated[:, :,
                                pad_before[0]:pad_before[0] + image_size_in[0],
                                pad_before[1]:pad_before[1] + image_size_in[1],
                                pad_before[2]:pad_before[2] + image_size_in[2]]
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# TTA predictor — 8-fold mirroring (matches nnU-Net inference_allowed_mirroring_axes)
# ─────────────────────────────────────────────────────────────────────────────

def make_tta_predictor(model: torch.nn.Module, use_tta: bool = True, tta_batch: int = 8,
                       tta_combos: list = None):
    """Returns a predictor that averages mirror augmentations (logit space).

    tta_combos: optional explicit list of flip combos (axes in zyx array order,
    e.g. [(), (0,), (2,)] = identity + flip-z + flip-x). None = all 8 (nnU-Net
    default). Must include () if provided. Flip combinations are evaluated in
    batched forward passes (up to tta_batch at a time) — same math, much better
    GPU utilization. On CUDA OOM the chunk is halved automatically (down to 1 =
    fully sequential), which makes this safe on shared GPUs.
    """
    flip_combos = tta_combos if tta_combos is not None else [
        (), (0,), (1,), (2,),
        (0, 1), (0, 2), (1, 2), (0, 1, 2),
    ]
    flip_combos = [tuple(c) for c in flip_combos]
    state = {"chunk": min(tta_batch, len(flip_combos))}

    def predictor(xb: torch.Tensor) -> torch.Tensor:
        # xb: (G, C, D, H, W) — G patch positions
        if not use_tta:
            return model(xb)
        g = xb.shape[0]
        while True:
            try:
                out = None
                for i in range(0, len(flip_combos), state["chunk"]):
                    sub = flip_combos[i:i + state["chunk"]]
                    flipped = torch.cat([
                        torch.flip(xb, dims=[a + 2 for a in axes]) if axes else xb
                        for axes in sub
                    ], dim=0)
                    pb = model(flipped)  # (len(sub)*G, cls, D, H, W)
                    del flipped
                    for k, axes in enumerate(sub):
                        p = pb[k * g:(k + 1) * g]
                        if axes:
                            p = torch.flip(p, dims=[a + 2 for a in axes])
                        out = p if out is None else out + p
                    del pb
                return out / len(flip_combos)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if state["chunk"] <= 1:
                    raise
                state["chunk"] //= 2
                print(f"[WARN] CUDA OOM in TTA — reducing TTA chunk to {state['chunk']}")

    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# Single-image inference
# ─────────────────────────────────────────────────────────────────────────────

def _revert_orientation(arr_ras: np.ndarray, orig_ornt: np.ndarray) -> np.ndarray:
    """Revert orientation from RAS back to original (flips/permutes)."""
    ras_ornt = nib.orientations.axcodes2ornt("RAS")
    transform = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
    return nib.orientations.apply_orientation(arr_ras, transform)


def infer_one(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    binary_threshold: float = 0.5,
    use_tta: bool = True,
    patch_batch: int = 1,
    precomputed: tuple = None,   # optional (data_1mm, info) from preprocess()
    predictor=None,              # optional shared predictor (keeps OOM-adapted TTA chunk across cases)
    sw_state: dict = None,       # optional shared dict (keeps OOM-adapted patch_batch across cases)
    patch_size: tuple = None,    # inference-time override; must be divisible by 32.
                                 # Larger than training patch is valid (fully-conv net +
                                 # size-agnostic attention) but changes effective context —
                                 # only use values validated on OOF.
) -> tuple:
    """Returns (soft_np, binary_np, native_affine, native_header).

    soft_np has EXACTLY the original image shape and native geometry.
    """
    ref = nib.load(image_path)
    orig_ornt = nib.orientations.io_orientation(ref.affine)
    orig_axcodes = nib.aff2axcodes(ref.affine)

    # ── Preprocess (exact nnU-Net pipeline; RAS-standardized) ───────────────
    data_1mm, info = precomputed if precomputed is not None else preprocess(image_path)
    img_tensor = torch.from_numpy(data_1mm).unsqueeze(0).to(device)  # (1, C, z, y, x)

    # ── Sliding-window + TTA (nnU-Net style) ────────────────────────────────
    if predictor is None:
        predictor = make_tta_predictor(model, use_tta=use_tta)
    ps = tuple(patch_size) if patch_size is not None else PATCH_SIZE
    assert all(p % 32 == 0 for p in ps), f"patch_size {ps} must be divisible by 32 (2**(n_stages-1))"
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = sliding_window_predict(
            model, img_tensor, ps, predictor, device,
            patch_batch=patch_batch, state=sw_state,
        )  # (1, 2, z, y, x) at 1mm, cropped

    # ── Postprocess: resample logits → softmax → revert crop → (x,y,z) ──────
    soft_ras = postprocess_logits(logits.squeeze(0), info)

    # ── Revert orientation: RAS → original ──────────────────────────────────
    if orig_axcodes != ("R", "A", "S"):
        soft_native = _revert_orientation(soft_ras, orig_ornt)
    else:
        soft_native = soft_ras

    assert soft_native.shape == ref.shape, \
        f"internal error: output shape {soft_native.shape} != native shape {ref.shape}"
    binary_native = (soft_native > binary_threshold).astype(np.uint8)
    return soft_native, binary_native, ref.affine, ref.header


def save_outputs(soft, binary, affine, hdr, out_dir, sid):
    dst = os.path.join(out_dir, sid)
    os.makedirs(dst, exist_ok=True)

    h_soft = hdr.copy(); h_soft.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(soft, affine, h_soft), os.path.join(dst, "soft.nii.gz"))

    h_bin = hdr.copy(); h_bin.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(binary, affine, h_bin), os.path.join(dst, "binary.nii.gz"))


# ─────────────────────────────────────────────────────────────────────────────
# Verification: compare MONAI inference vs nnU-Net official prediction
#
# NOTE: whole-volume binary agreement is dominated by background (>99.9%) and
# hides large disagreements ON the lesions. Use Dice + lesion-region soft diff.
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_nnunet(soft_monai: np.ndarray, nnunet_soft_path: str):
    """Print Dice-based agreement statistics between MONAI and nnU-Net soft maps."""
    ref = np.asanyarray(nib.load(nnunet_soft_path).dataobj).astype(np.float32)
    if ref.shape != soft_monai.shape:
        print(f"[WARN] shape mismatch: monai{soft_monai.shape} vs nnunet{ref.shape}")
        # try transpose (z,y,x) ↔ (x,y,z)
        for perm in [(2, 1, 0), (0, 1, 2), (1, 0, 2)]:
            cand = ref.transpose(perm)
            if cand.shape == soft_monai.shape:
                ref = cand
                print(f"  → transposed nnunet with perm={perm}")
                break

    b_m = soft_monai > 0.5
    b_n = ref > 0.5
    inter = (b_m & b_n).sum()
    dice = 2 * inter / max(b_m.sum() + b_n.sum(), 1e-8)
    union = b_m | b_n
    diff = np.abs(soft_monai - ref)

    print("\n===== MONAI vs nnU-Net comparison =====")
    print(f"  shape                 : {soft_monai.shape}")
    print(f"  lesion voxels         : monai={int(b_m.sum())} nnunet={int(b_n.sum())}")
    print(f"  binary Dice           : {dice:.6f}")
    print(f"  mean |soft diff|      : {diff.mean():.6f}")
    print(f"  mean |diff| on union  : {diff[union].mean():.6f}" if union.any() else "  (both empty)")
    print(f"  max  |diff|           : {diff.max():.6f}")
    print("  PASS" if dice > 0.99 else "  CHECK: Dice < 0.99")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_nifti_files(input_path: str) -> list:
    """Find .nii.gz files: flat dir or nnU-Net imagesTr layout (sub-XXXX_0000.nii.gz)."""
    if os.path.isfile(input_path):
        return [input_path]
    # nnU-Net imagesTr: {sid}_0000.nii.gz
    files = sorted(
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith("_0000.nii.gz")
    )
    if not files:
        # flat dir: {sid}.nii.gz
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".nii.gz")
        )
    return files


def sid_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ("_0000.nii.gz", ".nii.gz"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def main():
    ap = argparse.ArgumentParser(description="ISLES26 MONAI inference (nnU-Net compatible)")
    ap.add_argument("--input", required=True,
                    help="Single .nii.gz file, or directory of images")
    ap.add_argument("--out_root", default=None,
                    help="Output dir (default: skip saving, just run)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--chk", default="checkpoint_best",
                    help="checkpoint_best | checkpoint_final")
    ap.add_argument("--ckpt_root", default=None,
                    help="Checkpoint root containing fold_N/ (default: {project}/checkpoints)")
    ap.add_argument("--plans", default=None,
                    help="Path to plans.json — overrides built-in config (spacing/patch/arch). "
                         "Use this for Dataset002_ISLES26_RAS1x1x3 or any other nnU-Net model.")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binary threshold for soft map")
    ap.add_argument("--no_tta", action="store_true",
                    help="Disable 8-fold mirror TTA (8x faster, slightly lower quality — for debugging)")
    ap.add_argument("--patch_batch", type=int, default=1,
                    help="Patch positions per forward pass (effective batch = patch_batch x 8 with TTA; "
                         "increase only if the GPU has plenty of free memory)")
    ap.add_argument("--patch_size", default=None,
                    help="Inference-time patch override as 'z,y,x' (each divisible by 32). "
                         "Larger than the training patch is valid (fully-conv net) — "
                         "use only OOF-validated values.")
    ap.add_argument("--compare", default=None,
                    help="Path to nnU-Net soft.nii.gz for verification")
    args = ap.parse_args()

    global CKPT_ROOT
    if args.ckpt_root:
        CKPT_ROOT = args.ckpt_root
    # plans priority: explicit --plans > plans.json inside ckpt_root (self-describing
    # model dir) > built-in Dataset001 defaults
    plans_path = args.plans
    if plans_path is None and os.path.exists(os.path.join(CKPT_ROOT, "plans.json")):
        plans_path = os.path.join(CKPT_ROOT, "plans.json")
    if plans_path:
        apply_plans(plans_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.fold, args.chk, device)

    files = find_nifti_files(args.input)
    if not files:
        print(f"No .nii.gz files found in {args.input}")
        sys.exit(1)

    print(f"Processing {len(files)} image(s) | fold {args.fold} | {args.chk} | device {device}"
          f" | tta={not args.no_tta} | patch_batch={args.patch_batch}")
    t0 = time.time()

    # shared predictor: OOM-adapted TTA chunk size persists across cases
    predictor = make_tta_predictor(model, use_tta=not args.no_tta)
    sw_state = {}  # OOM-adapted patch_batch persists across cases

    # Overlap CPU preprocessing of the next case with GPU inference of the current one
    from concurrent.futures import ThreadPoolExecutor
    pre_pool = ThreadPoolExecutor(1) if len(files) > 1 else None
    pre_future = pre_pool.submit(preprocess, files[0]) if pre_pool else None

    for i, path in enumerate(files):
        sid = sid_from_path(path)
        t1 = time.time()
        try:
            precomputed = None
            if pre_pool:
                precomputed = pre_future.result()
                if i + 1 < len(files):
                    pre_future = pre_pool.submit(preprocess, files[i + 1])
            soft, binary, affine, hdr = infer_one(
                model, path, device, binary_threshold=args.threshold,
                use_tta=not args.no_tta, patch_batch=args.patch_batch,
                precomputed=precomputed, predictor=predictor, sw_state=sw_state,
                patch_size=tuple(int(v) for v in args.patch_size.split(",")) if args.patch_size else None,
            )
            dt = time.time() - t1
            print(f"  [{dt:6.1f}s] {sid}  shape={soft.shape}  "
                  f"lesion_vox={int(binary.sum())}")

            if args.out_root:
                save_outputs(soft, binary, affine, hdr, args.out_root, sid)

            if args.compare:
                compare_with_nnunet(soft, args.compare)

        except Exception as e:
            import traceback
            print(f"  [ERROR] {sid}: {e}")
            traceback.print_exc()

    if pre_pool:
        pre_pool.shutdown(wait=False)

    print(f"\nDone in {time.time()-t0:.1f}s"
          + (f" → {args.out_root}" if args.out_root else ""))


if __name__ == "__main__":
    main()
