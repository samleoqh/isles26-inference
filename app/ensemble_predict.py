"""Config-driven ensemble inference for ISLES'26 (see config/ensemble_config.json).

Per case:
  1. route thin/thick by max anatomical spacing (if the routed arm has no
     models, fall back to thin with a warning)
  2. preprocess ONCE per plans (models sharing a plans reuse the result;
     a different plans gets its own preprocessing)
  3. run each model in the selected arm(s) via the infer_monai core,
     average soft maps (weighted)
  4. post-process (threshold + VCAP + empty rule)

TTA is configurable globally, per arm, and per model:
  {"mode": "full"}              all 8 mirror combos (nnU-Net default)
  {"mode": "none"}              identity only
  {"mode": "single"}            identity + 3 single-axis flips (4 passes)
  {"mode": "subset", "flips": [["z"], ["y", "x"]]}
                                identity + the listed flip combos (axes z/y/x)

No TTA degradation and no hard cutoff exists anywhere in this code — every
configured model runs the same TTA setting on every case. The platform's own
per-case time limit is the only external bound. The time_guard config key is
accepted but ignored.
"""
import json
import os
import time

import nibabel as nib
import numpy as np

import infer_monai as im
import postproc

_AXIS = {"z": 0, "y": 1, "x": 2}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # model_root: config value, overridable by env (for local non-docker runs)
    cfg["_model_root"] = os.environ.get("ISLES_MODEL_ROOT",
                                        cfg.get("model_root", "/opt/ml/model"))
    return cfg


def tta_combos_from_spec(spec):
    """Resolve a TTA spec to an explicit combo list (None = infer_monai default 8)."""
    if spec is None:
        return None
    mode = spec.get("mode", "full")
    if mode == "full":
        return None
    if mode == "none":
        return [()]
    if mode == "single":
        return [(), (0,), (1,), (2,)]
    if mode == "subset":
        combos = []
        for c in spec.get("flips", []):
            t = tuple(sorted(_AXIS[a] for a in c))
            if t not in combos:
                combos.append(t)
        if () in combos:
            combos.remove(())
        return [()] + combos
    raise ValueError(f"unknown TTA mode: {mode}")


def anatomical_spacings(img) -> list:
    """Voxel sizes (mm) along anatomical R/A/S axes, from the affine."""
    ornt = nib.orientations.io_orientation(img.affine)
    zooms = img.header.get_zooms()[:3]
    sp = [None, None, None]
    for storage_ax in range(3):
        sp[int(ornt[storage_ax, 0])] = float(zooms[storage_ax])
    return sp


def _expand_models(cfg, arm_name):
    """Flatten an arm's model list to per-fold entries, inheriting arm settings.
    Returns [] when the arm is absent from the config (caller may fall back)."""
    out = []
    arm = cfg["arms"].get(arm_name)
    if arm is None:
        return out
    if isinstance(arm, dict):  # allow {"models": [...], "tta": {...}} arm-level
        arm_tta = arm.get("tta")
        entries = arm["models"]
    else:
        arm_tta = None
        entries = arm
    for e in entries:
        for fold in e["folds"]:
            out.append({
                "name": f"{e['name']}_f{fold}",
                "ckpt_root": os.path.join(cfg["_model_root"], e["ckpt_root"]),
                "fold": fold,
                "chk": e.get("chk", "checkpoint_final"),
                "weight": float(e.get("weight", 1.0)),
                "tta": e.get("tta", arm_tta),  # None -> global
            })
    return out


def _run_one_model(image_path, m, tta_spec, device, patch_batch, pre_cache, patch_size=None):
    """Predict one case with one fold checkpoint; returns native-geometry soft map.

    pre_cache: {(TARGET_SPACING, PATCH_SIZE, USE_MASK_FOR_NORM): (data, info)} —
    preprocessing (load → RAS → ZScore → resample) depends only on the case and
    these plans-geometry fields, so it is computed once per distinct geometry
    and shared by all models using it. Models with DIFFERENT geometry (e.g. the
    1x1x3 thick-slice arm) transparently get their own cache entry.
    """
    plans = os.path.join(m["ckpt_root"], "plans.json")
    im.apply_plans(plans)
    im.CKPT_ROOT = m["ckpt_root"]
    model = im.load_model(m["fold"], m["chk"], device)
    combos = tta_combos_from_spec(tta_spec)
    predictor = im.make_tta_predictor(model, use_tta=True, tta_combos=combos)
    # cache key = preprocessing-relevant plans geometry, NOT the plans path:
    # Dataset001 vs Dataset001Viola2Plus (and the two D4 roots) use different
    # plans files with identical geometry — path-keying ran the ~40s
    # preprocess once per plans file (2x for a 10-model mix). Geometry-keying
    # is a pure optimization: preprocess output depends only on these fields.
    key = (im.TARGET_SPACING, im.PATCH_SIZE, im.USE_MASK_FOR_NORM)
    if key not in pre_cache:
        t_pre = time.time()
        pre_cache[key] = im.preprocess(image_path)  # under THIS plans' globals
        print(f"[ensemble] preprocessed once for geometry {key}: "
              f"{time.time() - t_pre:.1f}s (shared by all models with same plans geometry)")
    soft, _, _, _ = im.infer_one(
        model, image_path, device,
        predictor=predictor, sw_state={}, patch_batch=patch_batch,
        precomputed=pre_cache[key], patch_size=patch_size,
    )
    del model, predictor
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return soft


def predict_case(image_path: str, cfg: dict):
    """Full pipeline for one case. Returns (soft, binary, ref_nifti_image, info)."""
    import torch
    t0 = time.time()
    info = {"models_used": [], "warnings": []}

    ref = nib.load(image_path)
    sp = anatomical_spacings(ref)
    thr_mm = cfg.get("routing", {}).get("thick_max_spacing_mm", 2.0)
    arm = "thick" if max(sp) >= thr_mm else "thin"
    info["spacings_ras_mm"] = sp
    info["arm"] = arm
    print(f"[ensemble] spacings(R,A,S)={['%.2f' % s for s in sp]} -> arm={arm}")

    models = _expand_models(cfg, arm)
    if not models and arm != "thin":
        # no thick-arm models configured: fall back to thin
        info["warnings"].append(f"no models configured for arm '{arm}', falling back to thin")
        print(f"[ensemble] WARN {info['warnings'][-1]}")
        arm = "thin"
        info["arm"] = arm
        models = _expand_models(cfg, arm)
    if not models:
        raise RuntimeError(f"no models configured for arm '{arm}'")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patch_batch = cfg.get("patch_batch", 1)
    patch_size = cfg.get("patch_size")  # optional inference-time override (validated values only)
    if patch_size is not None:
        patch_size = tuple(patch_size)
        print(f"[ensemble] inference patch_size override: {patch_size}")
    acc, w_sum = None, 0.0
    pre_cache = {}  # plans -> preprocessed (data, info), shared by models on same plans
    for i, m in enumerate(models):
        tta_spec = m["tta"] if m["tta"] is not None else cfg.get("tta", {"mode": "full"})
        # No TTA-degrade or hard-cutoff branches exist — no code path may drop
        # or degrade a model. Every model runs the same configured TTA; only
        # the platform's own per-case limit applies externally.

        ckpt = os.path.join(m["ckpt_root"], f"fold_{m['fold']}", f"{m['chk']}.pth")
        if not os.path.exists(ckpt):
            info["warnings"].append(f"missing checkpoint, skipped: {m['name']}")
            print(f"[ensemble] WARN missing checkpoint, skipped: {ckpt}")
            continue

        t1 = time.time()
        soft = _run_one_model(image_path, m, tta_spec, device, patch_batch, pre_cache,
                              patch_size=patch_size)
        dt = time.time() - t1
        n_tta = len(tta_combos_from_spec(tta_spec) or [()] * 8)
        print(f"[ensemble] {m['name']}: {dt:.1f}s (tta x{n_tta})")
        info["models_used"].append({"name": m["name"], "sec": round(dt, 1), "tta": n_tta})

        w = m["weight"]
        acc = soft * w if acc is None else acc + soft * w
        w_sum += w
        del soft

    if acc is None:
        raise RuntimeError(f"no usable checkpoints for arm '{arm}' under {cfg['_model_root']}")
    soft = (acc / w_sum).astype(np.float32)
    voxel_ml = float(np.prod(ref.header.get_zooms()[:3])) / 1000.0
    binary, soft = postproc.apply_postproc(soft, voxel_ml, cfg.get("postproc"))
    info["total_sec"] = round(time.time() - t0, 1)
    print(f"[ensemble] done in {info['total_sec']}s, lesion {binary.sum() * voxel_ml:.3f} ml")
    return soft, binary, ref, info
