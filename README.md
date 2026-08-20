# ISLES'26 — Stroke Lesion Segmentation (Inference Container)

Grand Challenge inference container for our ISLES'26 submission: an ensemble of
10 residual-encoder 3D U-Net models with [volume-conditioned post-processing](https://arxiv.org/abs/2608.16377).

## Method summary

- **Preprocessing**: reorientation to RAS, resampling to 1×1×1 mm isotropic,
  z-score normalization within the brain mask.
- **Architectures** (two nnU-Net residual-encoder families, ~102M params each):
  - `Dataset001ResEncL_ISLES26` — nnU-Net ResEnc-L preset (patch 160×192×160), 5-fold CV
  - `Dataset004ResEncMViola2Plus_ISLES26_s7` — ResEnc-M encoder + attention-gated
    decoder (patch 128×128×128), trained with lesion copy-paste augmentation;
    folds {0, 2, 4, 5, 6} of a 7-fold CV
- **Ensemble**: equal-weight averaging of the soft probability maps of all
  10 models. Test-time augmentation is disabled (internal ablation showed no
  statistically significant difference vs 8-fold mirroring).
- **Post-processing**: probability threshold 0.35, then volume-conditioned
  connected-component filtering (26-connectivity) — components ≤ 0.02 / 0.02 /
  0.1 ml are removed when the total predicted volume is < 3.5 ml / 3.5–35 ml /
  > 35 ml; cases with a remaining volume < 0.02 ml are reported as empty.

## Repository layout

```
app.py / inference.py     Grand Challenge invoke API (HTTP server on :4743)
app/ensemble_predict.py   model loading, routing, ensemble averaging
app/infer_monai.py        preprocessing + sliding-window inference core
app/postproc.py           threshold + VCAP component filter + empty rule
app/model_arch.py         PlainConvUNet building blocks
app/viola_plus.py         residual encoder + attention-gated decoder variants
config/ensemble_config.json   the exact submission configuration
run_local.py              run one case without Docker (for debugging)
do_build.sh / do_test_run.sh / do_save.sh   build → test → export
```

## Model weights

Weights are **not** part of this repository. The container expects them mounted
at `/opt/ml/model` with the following layout (paths are relative to
`model_root`, set in `config/ensemble_config.json`):

```
model_root/
├── Dataset001ResEncL_ISLES26/
│   ├── plans.json
│   ├── dataset.json
│   └── fold_{0..4}/checkpoint_final.pth
└── Dataset004ResEncMViola2Plus_ISLES26_s7/
    ├── plans.json
    ├── dataset.json
    └── fold_{0,2,4,5,6}/checkpoint_final.pth
```

Place them under `./model/` (the scripts mount that directory by default).

## Build and test

```bash
./do_build.sh        # builds the image "isles26-ensemble"
./do_test_run.sh     # boots the container, runs the case in test/input, writes test/output
./do_save.sh         # exports isles26-ensemble_<timestamp>.tar.gz + model.tar.gz for upload
```

`do_test_run.sh` expects a test image at
`test/input/interf0/images/t1-brain-mri/` (`.mha`, `.nii.gz` or `.nii`).

To run a single case without Docker:

```bash
ISLES_MODEL_ROOT=$PWD/model python run_local.py \
    --image /path/to/t1.nii.gz --config config/ensemble_config.json --out out/
```

## Runtime

Approximately 60 s per typical case on a single 16 GB GPU (NVIDIA T4); the
largest observed cases complete in under 6 minutes.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
