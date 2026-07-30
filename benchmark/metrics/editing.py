from __future__ import annotations

import numpy as np

from benchmark.metrics.image_metrics import mae


def edit_locality_metrics(before, after, target_mask, change_threshold: float = 1 / 255) -> dict:
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    if before.shape != after.shape:
        raise ValueError("Before and after images must have the same shape")
    if before.max(initial=0) > 1 or after.max(initial=0) > 1:
        before, after = before / 255.0, after / 255.0
    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != before.shape[:2]:
        raise ValueError("Target mask must match image dimensions")
    pixel_delta = np.abs(before - after).mean(axis=-1)
    changed = pixel_delta > float(change_threshold)
    inside_changed = float(changed[mask].mean()) if mask.any() else 0.0
    outside = ~mask
    outside_changed = float(changed[outside].mean()) if outside.any() else 0.0
    outside_mae = float(pixel_delta[outside].mean()) if outside.any() else 0.0
    inside_mae = float(pixel_delta[mask].mean()) if mask.any() else 0.0
    leakage_ratio = outside_mae / max(inside_mae, 1e-12)
    locality = inside_mae / max(inside_mae + outside_mae, 1e-12)
    return {
        "inside_changed_percent": inside_changed * 100,
        "outside_changed_percent": outside_changed * 100,
        "inside_mae": inside_mae, "outside_mae": outside_mae,
        "edit_leakage_ratio": float(leakage_ratio),
        "edit_locality_score": float(locality),
    }
