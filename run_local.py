#!/usr/bin/env python3
"""Local (non-docker) test of the ensemble pipeline.

Usage:
  ISLES_MODEL_ROOT=/path/to/model_root \
  python run_local.py \
      --image /path/to/t1.nii.gz \
      --config config/ensemble_config.json \
      --out /path/to/output_dir
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib

import ensemble_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="config/ensemble_config.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ensemble_predict.load_config(args.config)
    soft, binary, ref, info = ensemble_predict.predict_case(args.image, cfg)

    os.makedirs(args.out, exist_ok=True)
    h = ref.header.copy()
    h.set_data_dtype("float32")
    nib.save(nib.Nifti1Image(soft, ref.affine, h), os.path.join(args.out, "soft.nii.gz"))
    h2 = ref.header.copy()
    h2.set_data_dtype("uint8")
    nib.save(nib.Nifti1Image(binary, ref.affine, h2), os.path.join(args.out, "binary.nii.gz"))
    import json
    with open(os.path.join(args.out, "info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
