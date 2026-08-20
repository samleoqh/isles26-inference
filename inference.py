"""
ISLES'26 submission — config-driven ensemble inference.

Grand Challenge invoke API implementation. The heavy lifting lives in
ensemble_predict.py (model loading / ensemble averaging) and postproc.py;
the inference core is infer_monai.py (preprocessing, sliding-window
prediction, TTA).
"""
import glob
import json
import os
from pathlib import Path

import numpy
import SimpleITK

import ensemble_predict

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
CONFIG_PATH = Path(os.environ.get("ISLES_CONFIG", "/opt/app/config/ensemble_config.json"))

RESOURCE_PATH = Path("resources")


def init_model():
    """Load the ensemble configuration at startup (weights load lazily per case —
    the full checkpoint set does not need to sit in GPU memory, and per-case
    loading is ~1s)."""
    cfg = ensemble_predict.load_config(CONFIG_PATH)
    print(f"Loaded ensemble config from {CONFIG_PATH}")
    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2))
    return cfg


def run(model):
    """Called on each /invoke: read /input, run inference, write /output."""
    interface_key = get_interface_key()
    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]
    return handler(model)


def interf0_handler(cfg):
    # Locate the input image (any of the accepted formats)
    input_files = (
        glob.glob(str(INPUT_PATH / "images/t1-brain-mri" / "*.mha"))
        + glob.glob(str(INPUT_PATH / "images/t1-brain-mri" / "*.nii.gz"))
        + glob.glob(str(INPUT_PATH / "images/t1-brain-mri" / "*.nii"))
    )
    if not input_files:
        raise FileNotFoundError("No t1-brain-mri image found in /input")
    src = input_files[0]

    # Keep a SimpleITK reference for output geometry
    ref_sitk = SimpleITK.ReadImage(src)

    # infer pipeline reads via nibabel (.nii/.nii.gz); convert .mha if needed
    if src.endswith(".mha"):
        tmp = "/tmp/_input_t1.nii.gz"
        SimpleITK.WriteImage(ref_sitk, tmp, useCompression=True)
        image_path = tmp
    else:
        image_path = src

    # Metadata is read (and logged) but intentionally NOT used by the algorithm
    try:
        metadata = load_json_file(location=INPUT_PATH / "stroke-metadata.json")
        print("Stroke metadata (not used by algorithm):", json.dumps(metadata))
    except FileNotFoundError:
        pass

    soft, binary, ref_nii, info = ensemble_predict.predict_case(image_path, cfg)
    print("ensemble info:", json.dumps(info, indent=2))

    # nibabel arrays are (x, y, z); SimpleITK expects (z, y, x)
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=binary.transpose(2, 1, 0).astype(numpy.uint8),
        reference_image=ref_sitk,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/lesion-probability-map",
        array=soft.transpose(2, 1, 0).astype(numpy.float32),
        reference_image=ref_sitk,
    )
    return 0


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    # GC payload uses "interface"; the template's local test file uses "socket"
    socket_slugs = sorted(
        (s.get("interface") or s.get("socket"))["slug"] for s in inputs
    )
    return tuple(socket_slugs)


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)
    image = SimpleITK.GetImageFromArray(array)
    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
    SimpleITK.WriteImage(image, location / "output.mha", useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
