"""Post-processing for the ISLES'26 submission.

Pipeline: threshold -> VCAP volume-conditioned min-component filter -> empty rule.

Final configuration (validated on out-of-fold predictions): thr=0.35,
VCAP[V1=3.5, V2=35, tau=(0.02,0.02,0.1)ml], empty rule tau_vol=0.02ml
(soft map zeroed as well, for empty-GT PR-AUC).
"""
import numpy as np
from scipy import ndimage

DEFAULTS = {
    "threshold": 0.35,
    "vcap": {"v1_ml": 3.5, "v2_ml": 35.0, "tau_ml": [0.02, 0.02, 0.1]},
    "empty": {"tau_vol_ml": 0.02, "zero_soft": True},
}


def apply_postproc(soft: np.ndarray, voxel_ml: float, cfg: dict = None,
                   structure: np.ndarray = None):
    """soft: float32 prob map (native geometry). Returns (binary uint8, soft_out).

    cfg schema (all optional, defaults = final submission values):
      threshold: float
      vcap: {v1_ml, v2_ml, tau_ml: [small, mid, large]}  (null -> no CC filter)
      empty: {tau_vol_ml, zero_soft}

    structure: connectivity for ndimage.label. Default = 26-connectivity
    (np.ones((3,3,3))), matching the calibration code path (cc3d
    connectivity=26). Pass scipy's default (None handled here as 6-conn)
    ONLY for parity testing — never in production.
    """
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=np.int8)
    cfg = cfg or {}
    thr = cfg.get("threshold", DEFAULTS["threshold"])
    binary = (soft > thr).astype(np.uint8)

    vcap = cfg.get("vcap", DEFAULTS["vcap"])
    if vcap:
        v1, v2 = vcap["v1_ml"], vcap["v2_ml"]
        ts, tm, tl = vcap["tau_ml"]
        total_ml = binary.sum() * voxel_ml
        tau = ts if total_ml < v1 else (tm if total_ml <= v2 else tl)
        if tau > 0:
            lab, n = ndimage.label(binary, structure=structure)
            if n:
                sizes = ndimage.sum(binary, lab, range(1, n + 1)) * voxel_ml
                keep = np.zeros(n + 1, dtype=bool)
                keep[1:] = sizes > tau  # strict '>' (boundary is measure-zero)
                binary = keep[lab].astype(np.uint8)

    empty = cfg.get("empty", DEFAULTS["empty"])
    if empty and binary.sum() * voxel_ml <= empty.get("tau_vol_ml", 0.0):
        binary = np.zeros_like(binary)
        if empty.get("zero_soft", True):
            soft = np.zeros_like(soft)

    return binary, soft.astype(np.float32)
