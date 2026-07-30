from __future__ import annotations

import numpy as np


def _bool_pair(prediction, truth):
    pred, gt = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} != {gt.shape}")
    return pred, gt


def mask_metrics(prediction, truth, boundary_tolerance: int = 2) -> dict:
    pred, gt = _bool_pair(prediction, truth)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gt.any() else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    result = {
        "iou": tp / union if union else 1.0,
        "dice": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": precision, "recall": recall,
        "false_positive_pixels": fp, "false_negative_pixels": fn,
    }
    result["boundary_fscore"] = boundary_fscore(pred, gt, boundary_tolerance)
    return result


def boundary_fscore(prediction, truth, tolerance: int = 2) -> float | None:
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
    except ImportError:
        return None
    pred, gt = _bool_pair(prediction, truth)
    pred_boundary = pred ^ binary_erosion(pred)
    gt_boundary = gt ^ binary_erosion(gt)
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    pred_match = pred_boundary & binary_dilation(gt_boundary, structure=structure)
    gt_match = gt_boundary & binary_dilation(pred_boundary, structure=structure)
    precision = pred_match.sum() / max(1, pred_boundary.sum())
    recall = gt_match.sum() / max(1, gt_boundary.sum())
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def seam_crossing(mask) -> bool:
    value = np.asarray(mask, dtype=bool)
    return bool(value[:, 0].any() and value[:, -1].any())
