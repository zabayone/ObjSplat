''' MPS Splat Backend - Core logic for adaptive topology updates and training loop.'''

import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from utils.labelgs_mps import infer_point_labels, write_layerpano_compatible_ply


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _estimate_log_scales(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.full((points.shape[0], 3), -4.0, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nn = np.maximum(distances[:, 1] * 0.5, 1e-3)
        log_scale = np.log(nn).astype(np.float32)
        return np.repeat(log_scale[:, None], 3, axis=1)
    except Exception:
        return np.full((points.shape[0], 3), -4.0, dtype=np.float32)

def _apply_training_profile(
    profile: Optional[str],
    densify_interval: int,
    prune_threshold: float,
    clone_fraction: float,
    opacity_init: float,
    opacity_reg_weight: float,
    scale_log_min: float,
    scale_log_max: float,
) -> Tuple[int, float, float, float, float, float, float]:
    if profile == "per_layer":
        densify_interval = max(densify_interval, 250)
        prune_threshold = max(prune_threshold, 0.06)
        clone_fraction = min(max(clone_fraction, 0.02), 0.03)
        opacity_init = min(opacity_init, 0.08)
        opacity_reg_weight = max(opacity_reg_weight, 0.008)
        scale_log_min = min(scale_log_min, -7.5)
        scale_log_max = max(scale_log_max, 0.12)
        
    elif profile == "deva_instances":
        densify_interval = max(120, min(densify_interval, 180))
        prune_threshold = min(prune_threshold, 0.02)
        clone_fraction = min(max(clone_fraction, 0.08), 0.15)
        opacity_init = min(opacity_init, 0.08)
        opacity_reg_weight = max(opacity_reg_weight, 0.0075)
        scale_log_min = min(scale_log_min, -7.5)
        scale_log_max = max(scale_log_max, 0.28)

    elif profile == "final_fill":
        densify_interval = max(densify_interval, 120)
        prune_threshold = max(prune_threshold, 0.04)
        clone_fraction = max(clone_fraction, 0.04)
        opacity_init = min(opacity_init, 0.06)
        opacity_reg_weight = min(opacity_reg_weight, 0.002)
        scale_log_min = min(scale_log_min, -5.0)
        scale_log_max = min(scale_log_max, -2.0)

    return (
        densify_interval,
        prune_threshold,
        clone_fraction,
        opacity_init,
        opacity_reg_weight,
        scale_log_min,
        scale_log_max,
    )



def _voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    labels: Optional[np.ndarray],
    voxel_size: float = 0.01,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if len(points) == 0:
        return points, colors, labels
    min_pt = points.min(axis=0)
    quantized = ((points - min_pt) / voxel_size).astype(np.int32)
    max_coord = quantized.max(axis=0) + 1
    hash_vals = (
        quantized[:, 0] * (max_coord[1] * max_coord[2])
        + quantized[:, 1] * max_coord[2]
        + quantized[:, 2]
    )
    _, unique_idx = np.unique(hash_vals, return_index=True)
    unique_idx = np.sort(unique_idx)
    ds_points = points[unique_idx]
    ds_colors = colors[unique_idx]
    ds_labels = labels[unique_idx] if labels is not None else None
    if return_indices:
        return ds_points, ds_colors, ds_labels, unique_idx
    return ds_points, ds_colors, ds_labels


def _voxel_downsample_to_fraction(
    points: np.ndarray,
    colors: np.ndarray,
    labels: Optional[np.ndarray],
    target_fraction: float,
    max_iters: int = 20,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not (0.0 < target_fraction < 1.0) or len(points) == 0:
        return points, colors, labels

    target_count = max(1, int(round(len(points) * target_fraction)))
    bbox = points.max(axis=0) - points.min(axis=0)
    hi = float(max(np.max(bbox), 1e-6))
    lo = 1e-6

    best = (points, colors, labels)
    best_idx = np.arange(len(points), dtype=np.int64)
    best_diff = abs(len(points) - target_count)

    for _ in range(max_iters):
        voxel_size = (lo + hi) / 2.0
        ds_points, ds_colors, ds_labels, ds_idx = _voxel_downsample(
            points, colors, labels, voxel_size=voxel_size, return_indices=True
        )
        diff = abs(len(ds_points) - target_count)
        if diff < best_diff:
            best = (ds_points, ds_colors, ds_labels)
            best_idx = ds_idx
            best_diff = diff
        if len(ds_points) > target_count:
            lo = voxel_size
        else:
            hi = voxel_size

    if return_indices:
        return best[0], best[1], best[2], best_idx
    return best


def extract_gaussian_params_from_ply(ply_path: str) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Extract gaussian parameters and labels from a trained PLY file for re-activation
    in subsequent layers.
    
    Returns:
        (params_dict, labels) where params_dict has keys: means, scales, quaternions, opacities, sh_coeffs
    """
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    
    # Reconstruct scales from scale_0, scale_1, scale_2
    scales = np.stack([
        vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]
    ], axis=1).astype(np.float32)
    
    # Reconstruct quaternions from rot_0, rot_1, rot_2, rot_3
    quaternions = np.stack([
        vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]
    ], axis=1).astype(np.float32)
    
    opacities = vertex["opacity"][:, None].astype(np.float32)
    
    # SH coefficients: f_dc_0, f_dc_1, f_dc_2, then f_rest_*
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
    
    # Collect all f_rest_* fields
    vertex_names = getattr(getattr(vertex, "data", vertex), "dtype", None)
    vertex_names = getattr(vertex_names, "names", ()) or ()
    f_rest_fields = [k for k in vertex_names if k.startswith("f_rest_")]
    f_rest = np.stack([vertex[k] for k in f_rest_fields], axis=1).astype(np.float32) if f_rest_fields else np.zeros((len(means), 0), dtype=np.float32)
    
    # Reshape to (N, (sh_degree+1)^2 - 1, 3)
    n_coeffs = 3 * len(f_rest_fields) // 3  # Should be 3 * (sh_degree^2)
    sh_coeffs = np.zeros((len(means), 16, 3), dtype=np.float32)  # Assuming sh_degree=3
    sh_coeffs[:, 0, :] = f_dc
    sh_coeffs[:, 1:, :] = f_rest.reshape(len(means), -1, 3) if f_rest.size > 0 else np.zeros((len(means), 15, 3), dtype=np.float32)
    
    labels = None
    if "label" in vertex_names:
        labels = vertex["label"].astype(np.int32)
    
    params = {
        "means": means,
        "scales": scales,
        "quaternions": quaternions,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }
    
    return params, labels


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _build_optimizer(gaussians, mean_lr_scale: float = 1.0) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [gaussians.means],      "lr": 0.00016 * float(mean_lr_scale)},
            {"params": [gaussians.scales],      "lr": 0.005},
            {"params": [gaussians.quaternions], "lr": 0.005},
            {"params": [gaussians.opacities],   "lr": 0.05},
            {"params": [gaussians.sh_coeffs],   "lr": 0.0025},
        ],
        lr=0.001,
        eps=1e-15,
    )


# ---------------------------------------------------------------------------
# Adaptive topology - torch_gs
# ---------------------------------------------------------------------------

def _adaptive_topology_update(
    gaussians,
    prune_threshold: float = 0.05,
    clone_fraction: float = 0.02,
    min_points: int = 128,
    max_points: Optional[int] = None,
    grad_scores: Optional[np.ndarray] = None,
):
    means_all     = gaussians.means.detach().cpu().numpy().astype(np.float32)
    scales_all    = gaussians.scales.detach().cpu().numpy().astype(np.float32)
    quats_all     = gaussians.quaternions.detach().cpu().numpy().astype(np.float32)
    opacities_all = gaussians.opacities.detach().cpu().numpy().astype(np.float32)
    sh_coeffs_all = gaussians.sh_coeffs.detach().cpu().numpy().astype(np.float32)
    labels_all    = getattr(gaussians, "_labels_np", None)
    frozen_mask_all = getattr(gaussians, "_frozen_mask_np", None)
    if frozen_mask_all is None or len(frozen_mask_all) != len(means_all):
        frozen_mask_all = np.zeros(len(means_all), dtype=bool)
    else:
        frozen_mask_all = np.asarray(frozen_mask_all, dtype=bool).reshape(-1)

    active_idx = np.where(~frozen_mask_all)[0]
    frozen_idx = np.where(frozen_mask_all)[0]

    frozen_means = means_all[frozen_idx]
    frozen_scales = scales_all[frozen_idx]
    frozen_quats = quats_all[frozen_idx]
    frozen_opacities = opacities_all[frozen_idx]
    frozen_sh = sh_coeffs_all[frozen_idx]
    frozen_labels = labels_all[frozen_idx] if labels_all is not None else None

    means = means_all[active_idx]
    scales = scales_all[active_idx]
    quats = quats_all[active_idx]
    opacities = opacities_all[active_idx]
    sh_coeffs = sh_coeffs_all[active_idx]
    labels = labels_all[active_idx] if labels_all is not None else None

    if len(means) == 0:
        return gaussians

    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    score = opacity_sigmoid * np.exp(scales).mean(axis=1)

    if grad_scores is not None:
        grad_scores = np.asarray(grad_scores, dtype=np.float32)
        if grad_scores.ndim > 1:
            grad_scores = grad_scores.mean(axis=1)
        grad_scores = grad_scores.reshape(-1)
        if grad_scores.shape[0] == len(means_all):
            grad_scores = grad_scores[active_idx]
        if grad_scores.shape[0] == score.shape[0]:
            score = score * 0.5 + grad_scores * 0.5

    keep_mask = opacity_sigmoid >= prune_threshold
    if keep_mask.sum() < min_points and len(score) > 0:
        keep_mask = np.zeros(len(score), dtype=bool)
        keep_mask[np.argsort(score)[-min(min_points, len(score)):]] = True

    means     = means[keep_mask]
    scales    = scales[keep_mask]
    quats     = quats[keep_mask]
    opacities = opacities[keep_mask]
    sh_coeffs = sh_coeffs[keep_mask]
    score     = score[keep_mask]
    if labels is not None:
        labels = labels[keep_mask]

    if max_points is not None and len(score) > int(max_points):
        keep_top = int(max(max_points, min_points))
        top_idx  = np.argsort(score)[-keep_top:]
        means     = means[top_idx];    scales    = scales[top_idx]
        quats     = quats[top_idx];    opacities = opacities[top_idx]
        sh_coeffs = sh_coeffs[top_idx]; score    = score[top_idx]
        if labels is not None:
            labels = labels[top_idx]

    # ---- SPLIT large gaussians, CLONE small ones ----
    opacity_sigmoid_local = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    exp_scales = np.exp(scales)
    max_scale = exp_scales.max(axis=1)

    split_mask = (max_scale > 0.05) & (opacity_sigmoid_local > prune_threshold)
    clone_mask = (max_scale <= 0.05) & (opacity_sigmoid_local > prune_threshold)

    donor_means = means
    donor_scales = scales
    donor_quats = quats
    donor_opacities = opacities
    donor_sh_coeffs = sh_coeffs
    donor_labels = labels
    donor_score = score

    split_idx = np.where(split_mask)[0]
    clone_candidates = np.where(clone_mask)[0]

    if split_idx.size > 0:
        split_means = means[split_idx]
        split_scales = np.clip(scales[split_idx] - np.float32(np.log(1.6)), -7.0, 0.1)
        split_quats = quats[split_idx]
        split_opac = opacities[split_idx]
        split_sh = sh_coeffs[split_idx]

        local_extent = np.exp(scales[split_idx]).max(axis=1, keepdims=True)
        direction = np.random.randn(*split_means.shape).astype(np.float32)
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(direction_norm, 1e-6)
        split_means2 = split_means + direction * (local_extent * 0.75)

        keep_after_split = ~split_mask
        means = np.concatenate([means[keep_after_split], split_means, split_means2], axis=0)
        scales = np.concatenate([scales[keep_after_split], split_scales, split_scales], axis=0)
        quats = np.concatenate([quats[keep_after_split], split_quats, split_quats], axis=0)
        opacities = np.concatenate([opacities[keep_after_split], split_opac, split_opac], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs[keep_after_split], split_sh, split_sh], axis=0)
        score = np.concatenate([score[keep_after_split], score[split_idx], score[split_idx]], axis=0)
        if labels is not None:
            labels = np.concatenate([labels[keep_after_split], labels[split_idx], labels[split_idx]], axis=0)

    clone_count = 0
    if clone_candidates.size > 0 and clone_fraction > 0:
        clone_count = int(max(1, round(clone_candidates.size * clone_fraction)))
    if max_points is not None and clone_count > 0:
        clone_count = max(0, min(clone_count, int(max_points) - len(score)))

    if clone_count > 0:
        donor_scores = donor_score[clone_candidates]
        clone_indices = clone_candidates[np.argsort(donor_scores)[-clone_count:]]
        noise_scale = np.maximum(np.exp(donor_scales[clone_indices]) * 0.18, 1e-3)
        clone_means = donor_means[clone_indices] + np.random.normal(0.0, noise_scale).astype(np.float32)
        clone_scales = np.clip(donor_scales[clone_indices] - np.float32(np.log(2.0)), -7.0, 0.1)
        clone_quats = donor_quats[clone_indices]
        clone_opacities = np.full_like(donor_opacities[clone_indices], _logit(0.02), dtype=np.float32)
        clone_sh = donor_sh_coeffs[clone_indices]

        means = np.concatenate([means, clone_means], axis=0)
        scales = np.concatenate([scales, clone_scales], axis=0)
        quats = np.concatenate([quats, clone_quats], axis=0)
        opacities = np.concatenate([opacities, clone_opacities], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs, clone_sh], axis=0)
        if labels is not None and donor_labels is not None:
            labels = np.concatenate([labels, donor_labels[clone_indices]], axis=0)

    if max_points is not None and len(means) > int(max_points):
        keep_top = int(max(max_points, min_points))
        local_score = (1.0 / (1.0 + np.exp(-opacities[:, 0]))) * np.exp(scales).mean(axis=1)
        top_idx = np.argsort(local_score)[-keep_top:]
        means = means[top_idx]
        scales = scales[top_idx]
        quats = quats[top_idx]
        opacities = opacities[top_idx]
        sh_coeffs = sh_coeffs[top_idx]
        if labels is not None:
            labels = labels[top_idx]

    merged_means = np.concatenate([frozen_means, means], axis=0)
    merged_scales = np.concatenate([frozen_scales, scales], axis=0)
    merged_quats = np.concatenate([frozen_quats, quats], axis=0)
    merged_opacities = np.concatenate([frozen_opacities, opacities], axis=0)
    merged_sh = np.concatenate([frozen_sh, sh_coeffs], axis=0)
    merged_frozen_mask = np.concatenate([
        np.ones(len(frozen_means), dtype=bool),
        np.zeros(len(means), dtype=bool),
    ], axis=0)

    merged_labels = None
    if labels is not None or frozen_labels is not None:
        if frozen_labels is None:
            frozen_labels = np.zeros((len(frozen_means),), dtype=np.int32)
        if labels is None:
            labels = np.zeros((len(means),), dtype=np.int32)
        merged_labels = np.concatenate([frozen_labels, labels], axis=0)

    dev = gaussians.means.device
    gaussians.means       = torch.nn.Parameter(torch.tensor(merged_means,     dtype=torch.float32, device=dev))
    gaussians.scales      = torch.nn.Parameter(torch.tensor(merged_scales,    dtype=torch.float32, device=dev))
    gaussians.quaternions = torch.nn.Parameter(torch.tensor(merged_quats,     dtype=torch.float32, device=dev))
    gaussians.opacities   = torch.nn.Parameter(torch.tensor(merged_opacities, dtype=torch.float32, device=dev))
    gaussians.sh_coeffs   = torch.nn.Parameter(torch.tensor(merged_sh,        dtype=torch.float32, device=dev))
    gaussians._frozen_mask_np = merged_frozen_mask
    if merged_labels is not None:
        gaussians._labels_np = merged_labels
    return gaussians


# ---------------------------------------------------------------------------
# Adaptive topology - MLX
# ---------------------------------------------------------------------------

def _adaptive_topology_update_mlx(
    params: Dict[str, np.ndarray],
    labels: Optional[np.ndarray],
    frozen_mask: Optional[np.ndarray],
    prune_threshold: float = 0.05,
    clone_fraction: float = 0.02,
    min_points: int = 128,
    max_points: Optional[int] = None,
    scale_log_max: float = -2.8,
    split_scale_factor: float = 1.9,
):
    means_all     = np.asarray(params["means"],       dtype=np.float32)
    scales_all    = np.asarray(params["scales"],      dtype=np.float32)
    quats_all     = np.asarray(params["quaternions"], dtype=np.float32)
    opacities_all = np.asarray(params["opacities"],   dtype=np.float32)
    sh_coeffs_all = np.asarray(params["sh_coeffs"],   dtype=np.float32)

    if frozen_mask is None or len(frozen_mask) != len(means_all):
        frozen_mask_all = np.zeros(len(means_all), dtype=bool)
    else:
        frozen_mask_all = np.asarray(frozen_mask, dtype=bool).reshape(-1)

    frozen_idx = np.where(frozen_mask_all)[0]
    active_idx = np.where(~frozen_mask_all)[0]

    frozen_means = means_all[frozen_idx]
    frozen_scales = scales_all[frozen_idx]
    frozen_quats = quats_all[frozen_idx]
    frozen_opacities = opacities_all[frozen_idx]
    frozen_sh = sh_coeffs_all[frozen_idx]
    frozen_labels = np.asarray(labels, dtype=np.int32).reshape(-1)[frozen_idx] if labels is not None else None

    means = means_all[active_idx]
    scales = scales_all[active_idx]
    quats = quats_all[active_idx]
    opacities = opacities_all[active_idx]
    sh_coeffs = sh_coeffs_all[active_idx]
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)[active_idx] if labels is not None else None

    if len(means) == 0:
        merged_frozen_mask = np.ones(len(frozen_means), dtype=bool)
        return {
            "means": frozen_means,
            "scales": frozen_scales,
            "quaternions": frozen_quats,
            "opacities": frozen_opacities,
            "sh_coeffs": frozen_sh,
        }, frozen_labels, merged_frozen_mask

    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    score = opacity_sigmoid * np.exp(scales).mean(axis=1)

    keep_mask = opacity_sigmoid >= prune_threshold
    if keep_mask.sum() < min_points and len(score) > 0:
        keep_mask = np.zeros(len(score), dtype=bool)
        keep_mask[np.argsort(score)[-min(min_points, len(score)):]] = True

    means     = means[keep_mask];    scales    = scales[keep_mask]
    quats     = quats[keep_mask];    opacities = opacities[keep_mask]
    sh_coeffs = sh_coeffs[keep_mask]; score    = score[keep_mask]
    if labels is not None:
        labels = labels[keep_mask]

    if max_points is not None and len(score) > int(max_points):
        keep_top = int(max(max_points, min_points))
        top_idx  = np.argsort(score)[-keep_top:]
        means     = means[top_idx];    scales    = scales[top_idx]
        quats     = quats[top_idx];    opacities = opacities[top_idx]
        sh_coeffs = sh_coeffs[top_idx]; score    = score[top_idx]
        if labels is not None:
            labels = labels[top_idx]

    # ---- SPLIT large gaussians, CLONE small ones ----
    # Recompute local stats after keep/top-k so masks always match current arrays.
    opacity_sigmoid_local = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    exp_scales = np.exp(scales)
    max_scale = exp_scales.max(axis=1)

    split_threshold = float(np.exp(float(scale_log_max) * 0.9))
    split_mask = (max_scale > split_threshold) & (opacity_sigmoid_local > prune_threshold)
    clone_mask = (max_scale <= split_threshold) & (opacity_sigmoid_local > prune_threshold)

    # Snapshot donors before any modification so clone indexing stays valid.
    donor_means = means
    donor_scales = scales
    donor_quats = quats
    donor_opacities = opacities
    donor_sh_coeffs = sh_coeffs
    donor_labels = labels
    donor_score = score

    # Use donor indices from the same pre-update tensor so split/clone stay consistent.
    split_idx = np.where(split_mask)[0]
    clone_candidates = np.where(clone_mask)[0]

    # SPLIT: subdivide large gaussians into 2 smaller ones.
    if split_idx.size > 0:
        split_means = means[split_idx]
        split_scales = np.clip(
            scales[split_idx] - np.float32(np.log(float(split_scale_factor))),
            -7.0,
            scale_log_max,
        )
        split_quats = quats[split_idx]
        split_opac = opacities[split_idx]
        split_sh = sh_coeffs[split_idx]

        # Second copy offset along a random local direction so the split is not a near-duplicate.
        local_extent = np.exp(scales[split_idx]).max(axis=1, keepdims=True)
        direction = np.random.randn(*split_means.shape).astype(np.float32)
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(direction_norm, 1e-6)
        split_means2 = split_means + direction * (local_extent * 0.75)

        keep_after_split = ~split_mask
        means = np.concatenate([means[keep_after_split], split_means, split_means2], axis=0)
        scales = np.concatenate([scales[keep_after_split], split_scales, split_scales], axis=0)
        quats = np.concatenate([quats[keep_after_split], split_quats, split_quats], axis=0)
        opacities = np.concatenate([opacities[keep_after_split], split_opac, split_opac], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs[keep_after_split], split_sh, split_sh], axis=0)
        score = np.concatenate([score[keep_after_split], score[split_idx], score[split_idx]], axis=0)
        if labels is not None:
            labels = np.concatenate([labels[keep_after_split], labels[split_idx], labels[split_idx]], axis=0)

    # CLONE: only for small gaussians (using pre-update donors).
    clone_count = 0
    if clone_candidates.size > 0 and clone_fraction > 0:
        clone_count = int(max(1, round(clone_candidates.size * clone_fraction)))
    if max_points is not None and clone_count > 0:
        clone_count = max(0, min(clone_count, int(max_points) - len(score)))

    if clone_count > 0:
        donor_scores = donor_score[clone_candidates]
        donor_idx = clone_candidates[np.argsort(donor_scores)[-clone_count:]]

        noise_scale = np.maximum(np.exp(donor_scales[donor_idx]) * 0.18, 1e-3)
        clone_means = donor_means[donor_idx] + np.random.normal(0.0, noise_scale).astype(np.float32)
        clone_scales = np.clip(donor_scales[donor_idx] - np.float32(np.log(2.0)), -7.0, 0.1)
        clone_quats = donor_quats[donor_idx]
        clone_opacities = np.full_like(donor_opacities[donor_idx], _logit(0.02), dtype=np.float32)
        clone_sh = donor_sh_coeffs[donor_idx]

        means = np.concatenate([means, clone_means], axis=0)
        scales = np.concatenate([scales, clone_scales], axis=0)
        quats = np.concatenate([quats, clone_quats], axis=0)
        opacities = np.concatenate([opacities, clone_opacities], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs, clone_sh], axis=0)
        if labels is not None and donor_labels is not None:
            labels = np.concatenate([labels, donor_labels[donor_idx]], axis=0)

    # Hard cap in case split/clone exceeded max_points.
    if max_points is not None and len(means) > int(max_points):
        keep_top = int(max(max_points, min_points))
        local_score = (1.0 / (1.0 + np.exp(-opacities[:, 0]))) * np.exp(scales).mean(axis=1)
        top_idx = np.argsort(local_score)[-keep_top:]
        means = means[top_idx]
        scales = scales[top_idx]
        quats = quats[top_idx]
        opacities = opacities[top_idx]
        sh_coeffs = sh_coeffs[top_idx]
        if labels is not None:
            labels = labels[top_idx]

    merged_means = np.concatenate([frozen_means, means], axis=0)
    merged_scales = np.concatenate([frozen_scales, scales], axis=0)
    merged_quats = np.concatenate([frozen_quats, quats], axis=0)
    merged_opacities = np.concatenate([frozen_opacities, opacities], axis=0)
    merged_sh = np.concatenate([frozen_sh, sh_coeffs], axis=0)
    merged_frozen_mask = np.concatenate([
        np.ones(len(frozen_means), dtype=bool),
        np.zeros(len(means), dtype=bool),
    ], axis=0)

    merged_labels = None
    if labels is not None or frozen_labels is not None:
        if frozen_labels is None:
            frozen_labels = np.zeros((len(frozen_means),), dtype=np.int32)
        if labels is None:
            labels = np.zeros((len(means),), dtype=np.int32)
        merged_labels = np.concatenate([frozen_labels, labels], axis=0)

    return {
        "means": merged_means, "scales": merged_scales, "quaternions": merged_quats,
        "opacities": merged_opacities, "sh_coeffs": merged_sh,
    }, merged_labels, merged_frozen_mask


def _gaussian_selector(frozen_means: np.ndarray, frozen_scales: np.ndarray, 
                        new_points: np.ndarray, beta3: float = 10.0) -> np.ndarray:
    """
    Finds frozen gaussians that lie in front of new_points along the same ray
    from origin (0,0,0) - implements paper eq. 7-8.
    Returns a boolean mask of gaussians to re-activate.
    """
    d_new = np.linalg.norm(new_points, axis=1, keepdims=True)          # (M,1)
    d_g   = np.linalg.norm(frozen_means, axis=1)                        # (N,)
    max_s = np.exp(frozen_scales).max(axis=1)                           # (N,)
    d_g_adj = d_g - max_s                                               # eq. 8
    
    ray_new = new_points / np.maximum(d_new, 1e-6)                      # (M,3)
    ray_g   = frozen_means / np.maximum(d_g[:, None], 1e-6)             # (N,3)

    # Fast 3D hashing: logarithmic grid as in paper
    def ray_to_grid(r, beta3):
        return np.ceil(beta3 * np.log(np.abs(r) + 1) * np.sign(r)).astype(np.int32)

    # Build grid -> minimum distance among new_points falling in that grid cell
    grid_new = {}
    grid_cells_new = ray_to_grid(ray_new, beta3)
    for idx, cell in enumerate(grid_cells_new):
        key = tuple(cell)
        d = float(d_new[idx, 0])
        if key in grid_new:
            if d < grid_new[key]:
                grid_new[key] = d
        else:
            grid_new[key] = d

    # For each frozen gaussian, check if its quantized ray exists and is further than
    # the nearest new_point along that ray (using per-cell min distance).
    active_mask = np.array([
        (tuple(ray_to_grid(ray_g[i:i+1], beta3)[0]) in grid_new)
        and (d_g_adj[i] < grid_new.get(tuple(ray_to_grid(ray_g[i:i+1], beta3)[0]), np.inf))
        for i in range(len(frozen_means))
    ])
    return active_mask


def _transfer_frozen_gaussians(
    frozen_params: Optional[Dict[str, np.ndarray]],
    frozen_labels: Optional[np.ndarray],
    new_xyz: np.ndarray,
    new_rgb: np.ndarray,
    new_labels: Optional[np.ndarray],
    beta3: float = 8.0,
    min_distance: float = 0.03,
    max_reactivated: int = 250000,
    min_transfer_opacity: float = 0.05,
    max_transfer_scale: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Transfer frozen gaussians from previous layer that match ray-geometry with
    new points in current layer. Uses _gaussian_selector to identify candidates.
    
    Args:
        frozen_params: Dict with 'means', 'scales', 'quaternions', 'opacities', 'sh_coeffs'
        frozen_labels: int32 array of frozen gaussian labels (layer_idx, instance_id)
        new_xyz: (N, 3) point cloud for current layer
        new_rgb: (N, 3) colors for new points
        new_labels: (N,) labels for new points
        beta3: Ray-matching grid parameter (paper eq. 7)
        min_distance: Min distance threshold for ray-matching
    
    Returns:
        (merged_xyz, merged_rgb, merged_labels) - new point cloud with transferred gaussians
    """
    if frozen_params is None or "means" not in frozen_params:
        return new_xyz, new_rgb, new_labels, np.zeros((len(new_xyz),), dtype=bool)
    
    frozen_means = np.asarray(frozen_params["means"], dtype=np.float32)
    frozen_scales = np.asarray(frozen_params["scales"], dtype=np.float32)
    
    if len(frozen_means) == 0 or len(new_xyz) == 0:
        return new_xyz, new_rgb, new_labels, np.zeros((len(new_xyz),), dtype=bool)

    frozen_opacities = np.asarray(frozen_params.get("opacities", np.zeros((len(frozen_means), 1), dtype=np.float32)), dtype=np.float32).reshape(-1)
    frozen_opacity_sig = 1.0 / (1.0 + np.exp(-frozen_opacities))
    frozen_scale_max = np.exp(frozen_scales).max(axis=1)
    eligible = (frozen_opacity_sig >= float(min_transfer_opacity)) & (frozen_scale_max <= float(max_transfer_scale))
    if not np.any(eligible):
        return new_xyz, new_rgb, new_labels, np.zeros((len(new_xyz),), dtype=bool)

    frozen_means = frozen_means[eligible]
    frozen_scales = frozen_scales[eligible]
    if frozen_labels is not None:
        frozen_labels = np.asarray(frozen_labels, dtype=np.int32).reshape(-1)[eligible]
    
    # Select frozen gaussians that lie on same rays as new points
    active_mask = _gaussian_selector(frozen_means, frozen_scales, new_xyz, beta3=beta3)
    
    if not active_mask.any():
        # No frozen gaussians to transfer
        return new_xyz, new_rgb, new_labels, np.zeros((len(new_xyz),), dtype=bool)
    
    # Extract active frozen gaussians
    active_idx = np.where(active_mask)[0]
    if len(active_idx) > int(max_reactivated):
        keep_idx = np.random.choice(active_idx, size=int(max_reactivated), replace=False)
        active_idx = np.sort(keep_idx)
    transfer_xyz = frozen_means[active_idx]
    
    # For transferred gaussians, inherit color from frozen state
    # (assume frozen_opacities were extracted elsewhere if needed)
    transfer_rgb = np.ones((len(transfer_xyz), 3), dtype=np.float32) * 0.5  # mid-gray fallback
    
    # Merge point clouds
    merged_xyz = np.concatenate([new_xyz, transfer_xyz], axis=0)
    merged_rgb = np.concatenate([new_rgb, transfer_rgb], axis=0)
    
    # Merge labels if provided
    merged_labels = new_labels
    if new_labels is not None and frozen_labels is not None:
        frozen_labels_active = frozen_labels[active_idx]
        merged_labels = np.concatenate([new_labels, frozen_labels_active], axis=0).astype(np.int32)
    elif frozen_labels is not None:
        frozen_labels_active = frozen_labels[active_idx]
        merged_labels = np.concatenate([np.zeros_like(new_labels), frozen_labels_active], axis=0).astype(np.int32)
    
    frozen_mask = np.concatenate([
        np.zeros((len(new_xyz),), dtype=bool),
        np.ones((len(transfer_xyz),), dtype=bool),
    ], axis=0)

    print(f"[Layer Transfer] Re-activated {len(transfer_xyz)} frozen gaussians from previous layer")
    return merged_xyz, merged_rgb, merged_labels, frozen_mask


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _ensure_torch_gs_importable() -> None:
    try:
        import torch_gs  # noqa: F401
        return
    except Exception:
        pass
    candidates = []
    env_path = os.environ.get("SPLAT_APPLE_PATH")
    if env_path:
        candidates.append(env_path)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(repo_root, "submodules", "splat-apple"))
    candidates.append(os.path.join(repo_root, "external",   "splat-apple"))
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
        try:
            import torch_gs  # noqa: F401
            return
        except Exception:
            continue
    raise ModuleNotFoundError(
        "Modulo torch_gs non trovato. Installa splat-apple e compila estensioni: "
        "`python setup.py build_ext --inplace` nel repo splat-apple; "
        "poi esporta SPLAT_APPLE_PATH o clona in submodules/splat-apple."
    )


def _ensure_mlx_gs_importable() -> None:
    try:
        import mlx_gs  # noqa: F401
        import mlx.core  # noqa: F401
        return
    except Exception:
        pass
    candidates = []
    env_path = os.environ.get("SPLAT_APPLE_PATH")
    if env_path:
        candidates.append(env_path)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(repo_root, "submodules", "splat-apple"))
    candidates.append(os.path.join(repo_root, "external",   "splat-apple"))
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
        try:
            import mlx_gs  # noqa: F401
            import mlx.core  # noqa: F401
            return
        except Exception:
            continue
    raise ModuleNotFoundError(
        "Modulo mlx_gs non trovato. Installa splat-apple + MLX e compila il rasterizer Metal: "
        "`python setup_mlx.py build_ext --inplace` nel repo splat-apple."
    )


def _resolve_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _to_target_tensor(image, device: str) -> torch.Tensor:
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).to(device)


def _build_training_batch(traindata: Dict, device: str, downsample_ratio: float = 1.0):
    from torch_gs.training.trainer import Camera
    from mlx_gs.io.colmap import pose_to_w2c

    fov_deg = float(traindata["fov"])
    width   = int(traindata["W"])
    height  = int(traindata["H"])
    fovx = math.radians(fov_deg)
    fovy = height * fovx / width
    fx   = width  / (2.0 * math.tan(fovx / 2.0))
    fy   = height / (2.0 * math.tan(fovy / 2.0))
    cx   = width  / 2.0
    cy   = height / 2.0

    cameras: List[Camera] = []
    targets: List[torch.Tensor] = []
    for frame in traindata["frames"]:
        w2c = pose_to_w2c(np.array(frame["transform_matrix"], dtype=np.float32))
        cameras.append(Camera(
            W=width, H=height, fx=fx, fy=fy, cx=cx, cy=cy,
            W2C=torch.tensor(w2c, dtype=torch.float32, device=device),
        ))
        targets.append(_to_target_tensor(frame["image"], device))

    xyz    = np.asarray(traindata["pcd_points"], dtype=np.float32)
    rgb    = np.asarray(traindata["pcd_colors"], dtype=np.float32)
    labels = traindata.get("pcd_labels", None)
    frozen_mask = traindata.get("frozen_mask", None)
    if labels is None:
        labels = infer_point_labels(traindata)

    if 0.0 < downsample_ratio < 1.0 and len(xyz) > 0:
        xyz, rgb, labels, keep_idx = _voxel_downsample_to_fraction(
            xyz, rgb, labels, downsample_ratio, return_indices=True
        )
        if frozen_mask is not None:
            frozen_mask = np.asarray(frozen_mask, dtype=bool).reshape(-1)[keep_idx]

    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz, rgb, labels, frozen_mask, cameras, targets


def _build_training_batch_mlx(traindata: Dict, downsample_ratio: float = 1.0):
    import mlx.core as mx
    from mlx_gs.io.colmap import pose_to_w2c
    from mlx_gs.training.trainer import Camera

    fov_deg = float(traindata["fov"])
    width   = int(traindata["W"])
    height  = int(traindata["H"])
    fovx = math.radians(fov_deg)
    fovy = height * fovx / width
    fx   = width  / (2.0 * math.tan(fovx / 2.0))
    fy   = height / (2.0 * math.tan(fovy / 2.0))
    cx   = width  / 2.0
    cy   = height / 2.0

    cameras, targets = [], []
    for frame in traindata["frames"]:
        w2c = pose_to_w2c(np.array(frame["transform_matrix"], dtype=np.float32))
        cameras.append(Camera(W=width, H=height, fx=fx, fy=fy, cx=cx, cy=cy,
                               W2C=mx.array(w2c, dtype=mx.float32)))
        rgb = np.array(frame["image"].convert("RGB"), dtype=np.float32) / 255.0
        targets.append(mx.array(rgb, dtype=mx.float32))

    xyz    = np.asarray(traindata["pcd_points"], dtype=np.float32)
    rgb    = np.asarray(traindata["pcd_colors"], dtype=np.float32)
    labels = traindata.get("pcd_labels", None)
    frozen_mask = traindata.get("frozen_mask", None)
    if labels is None:
        labels = infer_point_labels(traindata)

    if 0.0 < downsample_ratio < 1.0 and len(xyz) > 0:
        xyz, rgb, labels, keep_idx = _voxel_downsample_to_fraction(
            xyz, rgb, labels, downsample_ratio, return_indices=True
        )
        if frozen_mask is not None:
            frozen_mask = np.asarray(frozen_mask, dtype=bool).reshape(-1)[keep_idx]

    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz, rgb, labels, frozen_mask, cameras, targets

def _fill_labels_by_nearest_neighbor(points: np.ndarray, labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if labels is None:
        return None

    pts = np.asarray(points, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)

    if pts.shape[0] == 0 or lab.shape[0] != pts.shape[0]:
        return lab

    known = lab > 0
    if known.all() or not known.any():
        return lab

    known_pts = pts[known]
    known_lab = lab[known]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(known_pts)
        _, nn_idx = tree.query(pts[~known], k=1)
        lab[~known] = known_lab[np.asarray(nn_idx, dtype=np.int64)]
    except Exception:
        diff = pts[~known][:, None, :] - known_pts[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nn_idx = np.argmin(dist2, axis=1)
        lab[~known] = known_lab[np.asarray(nn_idx, dtype=np.int64)]

    return lab


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

def _save_layerpano_compatible_ply(
    path: str,
    gaussians,
    labels_np: Optional[np.ndarray],
    sh_degree: int = 3,
) -> None:
    xyz       = gaussians.means.detach().cpu().numpy().astype(np.float32)
    normals   = np.zeros_like(xyz, dtype=np.float32)
    f_dc      = gaussians.sh_coeffs[:, 0, :].detach().cpu().numpy().astype(np.float32)
    expected_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    f_rest    = np.zeros((xyz.shape[0], expected_rest), dtype=np.float32)
    opacities = gaussians.opacities.detach().cpu().numpy().astype(np.float32)
    scales    = gaussians.scales.detach().cpu().numpy().astype(np.float32)
    rots      = gaussians.quaternions.detach().cpu().numpy().astype(np.float32)

    finite_mask = (
        np.isfinite(xyz).all(axis=1) & np.isfinite(f_dc).all(axis=1)
        & np.isfinite(opacities).all(axis=1) & np.isfinite(scales).all(axis=1)
        & np.isfinite(rots).all(axis=1)
    )
    if not np.all(finite_mask):
        print(f"Filtering out {int((~finite_mask).sum())} non-finite gaussians before export.")
        xyz = xyz[finite_mask]; normals = normals[finite_mask]; f_dc = f_dc[finite_mask]
        f_rest = f_rest[finite_mask]; opacities = opacities[finite_mask]
        scales = scales[finite_mask]; rots = rots[finite_mask]
        if labels_np is not None:
            labels_np = labels_np[finite_mask]

    labels_np = _fill_labels_by_nearest_neighbor(xyz, labels_np)

    write_layerpano_compatible_ply(
        path=path, xyz=xyz, normals=normals, f_dc=f_dc, f_rest=f_rest,
        opacities=opacities, scales=scales, rots=rots, labels=labels_np,
    )


def _save_layerpano_compatible_ply_mlx(
    path: str,
    params: Dict[str, np.ndarray],
    labels: Optional[np.ndarray],
    scale_log_min: float = -7.0,
    scale_log_max: float = 0.28,
    sh_degree: int = 3,
) -> None:
    params = dict(params)

    xyz = np.asarray(params["means"], dtype=np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)
    sh_coeffs = np.asarray(params["sh_coeffs"], dtype=np.float32)

    if sh_coeffs.ndim != 3 or sh_coeffs.shape[2] != 3:
        raise ValueError(f"Unexpected SH shape {sh_coeffs.shape}; expected (N, C, 3)")

    f_dc = sh_coeffs[:, 0, :].astype(np.float32)
    expected_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    rest_coeffs = sh_coeffs[:, 1:, :].transpose(0, 2, 1).reshape(xyz.shape[0], -1).astype(np.float32)
    f_rest = (
        np.pad(rest_coeffs, ((0, 0), (0, max(0, expected_rest - rest_coeffs.shape[1]))), mode="constant")
        if rest_coeffs.shape[1] < expected_rest
        else rest_coeffs[:, :expected_rest]
    )

    opacities = np.asarray(params["opacities"], dtype=np.float32)
    opacities = np.clip(opacities, -4.0, 6.0)

    scales = np.asarray(params["scales"], dtype=np.float32)
    scales = np.clip(scales, scale_log_min, scale_log_max)
    params["scales"] = scales

    if scales.ndim == 1:
        scales = scales[:, None]

    if not np.isfinite(scales).all():
        raise ValueError("Non-finite scales detected before export")

    rots = np.asarray(params["quaternions"], dtype=np.float32)

    finite_mask = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(f_dc).all(axis=1)
        & np.isfinite(f_rest).all(axis=1)
        & np.isfinite(opacities).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(rots).all(axis=1)
    )

    if not np.all(finite_mask):
        print(f"Filtering out {int((~finite_mask).sum())} non-finite gaussians before export.")
        xyz = xyz[finite_mask]
        normals = normals[finite_mask]
        f_dc = f_dc[finite_mask]
        f_rest = f_rest[finite_mask]
        opacities = opacities[finite_mask]
        scales = scales[finite_mask]
        rots = rots[finite_mask]

    labels_np = None
    if labels is not None:
        labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels_np.shape[0] != finite_mask.shape[0]:
            raise ValueError(f"labels length mismatch: {labels_np.shape[0]} != {finite_mask.shape[0]}")
        labels_np = labels_np[finite_mask]

    labels_np = _fill_labels_by_nearest_neighbor(xyz, labels_np)

    write_layerpano_compatible_ply(
        path=path,
        xyz=xyz,
        normals=normals,
        f_dc=f_dc,
        f_rest=f_rest,
        opacities=opacities,
        scales=scales,
        rots=rots,
        labels=labels_np,
    )



# ---------------------------------------------------------------------------
# torch_gs backend
# ---------------------------------------------------------------------------

def _train_torch_gs(
    traindata: Dict,
    out_ply_path: str,
    num_iterations: int,
    rasterizer: str,
    device: str,
    adaptive: bool,
    densify_interval: int,
    prune_threshold: float,
    clone_fraction: float,
    max_points: Optional[int],
    downsample_ratio: float = 1.0,
    mean_lr_scale: float = 1.0,
    opacity_init: float = 0.1,
    opacity_reg_weight: float = 0.005,
    scale_log_min: float = -7.0,
    scale_log_max: float = 1.0,
    initial_gaussian_params: Optional[Dict[str, np.ndarray]] = None,
    initial_gaussian_labels: Optional[np.ndarray] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    lr_plateau_patience: Optional[int] = None,
    lr_plateau_factor: float = 0.5,
    lr_plateau_min_lr: float = 1e-6,
) -> str:
    _ensure_torch_gs_importable()
    from torch_gs.core.gaussians import init_gaussians_from_pcd
    from torch_gs.training.trainer import train_step

    if device == "auto":
        device = _resolve_device()
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this host.")

    xyz, rgb, labels, frozen_mask, cameras, targets = _build_training_batch(traindata, device, downsample_ratio)
    # record initial scene scale so we can preserve it after training
    init_scene_scale = max(float(np.linalg.norm(np.std(xyz, axis=0))), 1e-3)

    if initial_gaussian_params is not None:
        init_xyz = np.asarray(initial_gaussian_params.get("means"), dtype=np.float32)
        init_sh = np.asarray(initial_gaussian_params.get("sh_coeffs"), dtype=np.float32)
        if init_xyz.size == 0:
            raise ValueError("initial_gaussian_params has no points")
        if init_sh.ndim == 3 and init_sh.shape[-1] == 3:
            init_rgb = np.clip(init_sh[:, 0, :], 0.0, 1.0).astype(np.float32)
        else:
            init_rgb = np.zeros((init_xyz.shape[0], 3), dtype=np.float32)
        xyz = init_xyz
        rgb = init_rgb
        labels = initial_gaussian_labels

    scale_init = _estimate_log_scales(xyz)
    scale_init = np.clip(scale_init, scale_log_min, scale_log_max)
    gaussians  = init_gaussians_from_pcd(
        torch.tensor(xyz, dtype=torch.float32),
        torch.tensor(rgb, dtype=torch.float32),
        device=device,
    )
    gaussians.scales    = torch.nn.Parameter(torch.tensor(scale_init, dtype=torch.float32, device=device))
    gaussians.opacities = torch.nn.Parameter(torch.full(
        (gaussians.opacities.shape[0], 1), _logit(opacity_init), dtype=torch.float32, device=device
    ))
    gaussians.means.requires_grad = gaussians.scales.requires_grad = True
    gaussians.quaternions.requires_grad = gaussians.opacities.requires_grad = True
    gaussians.sh_coeffs.requires_grad = True
    gaussians._labels_np = labels if labels is not None else None
    gaussians._frozen_mask_np = (
        np.asarray(frozen_mask, dtype=bool).reshape(-1)
        if frozen_mask is not None else np.zeros((len(xyz),), dtype=bool)
    )

    if initial_gaussian_params is not None:
        gaussians.means.data = torch.tensor(initial_gaussian_params["means"], dtype=torch.float32, device=device)
        gaussians.scales.data = torch.tensor(initial_gaussian_params["scales"], dtype=torch.float32, device=device)
        gaussians.quaternions.data = torch.tensor(initial_gaussian_params["quaternions"], dtype=torch.float32, device=device)
        gaussians.opacities.data = torch.tensor(initial_gaussian_params["opacities"], dtype=torch.float32, device=device)
        gaussians.sh_coeffs.data = torch.tensor(initial_gaussian_params["sh_coeffs"], dtype=torch.float32, device=device)

    optimizer = _build_optimizer(gaussians, mean_lr_scale=mean_lr_scale)

    best_loss = float("inf")
    no_improve = 0
    no_improve_lr = 0

    grad_accum = torch.zeros(gaussians.means.shape, device=device)
    grad_count = 0

    # Densify only during the first 60% of iterations, following standard 3DGS practice.
    densify_stop_iter = int(num_iterations * 0.6)

    progress = tqdm(range(num_iterations), desc="Splat-Apple training")
    t0 = time.time()

    for i in progress:
        idx = random.randint(0, len(cameras) - 1)
        try:
            loss, psnr, _ = train_step(
                gaussians, optimizer,
                targets[idx], cameras[idx],
                lambda_ssim=0.25,
                device=device,
                rasterizer_type=rasterizer,
                scale_reg_weight=opacity_reg_weight,
                opacity_reg_weight=opacity_reg_weight,
            )
        except RuntimeError as e:
            if device == "mps" and "MPS backend out of memory" in str(e):
                print("[splat-apple] MPS OOM, pruning and continuing.", flush=True)
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                current_n = int(gaussians.means.shape[0])
                prune_to  = int(max(4096, current_n * 0.8))
                if max_points is not None:
                    prune_to = min(prune_to, int(max_points))
                gaussians = _adaptive_topology_update(
                    gaussians, prune_threshold=max(prune_threshold, 0.05),
                    clone_fraction=0.0, min_points=512, max_points=prune_to,
                )
                gaussians.means.requires_grad = gaussians.scales.requires_grad = True
                gaussians.quaternions.requires_grad = gaussians.opacities.requires_grad = True
                gaussians.sh_coeffs.requires_grad = True
                optimizer = _build_optimizer(gaussians, mean_lr_scale=mean_lr_scale)
                continue
            raise

        if gaussians.means.grad is not None:
            if grad_accum.shape != gaussians.means.shape:
                grad_accum = torch.zeros(gaussians.means.shape, device=device)
                grad_count = 0
            grad_accum += gaussians.means.grad.norm(dim=-1).detach().unsqueeze(-1).expand_as(grad_accum)
            grad_count += 1

        loss_value = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
        if loss_value < best_loss - float(early_stop_min_delta or 0.0):
            best_loss = loss_value
            no_improve = 0
            no_improve_lr = 0
        else:
            no_improve += 1
            no_improve_lr += 1

        if lr_plateau_patience is not None and lr_plateau_patience > 0:
            if no_improve_lr >= lr_plateau_patience:
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * float(lr_plateau_factor), float(lr_plateau_min_lr))
                no_improve_lr = 0
                lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
                print(f"[splat-apple] lr reduced on plateau -> {lrs}", flush=True)

        if early_stop_patience is not None and early_stop_patience > 0:
            if no_improve >= early_stop_patience:
                print(f"[splat-apple] early stopping at iter={i} (best_loss={best_loss:.6f})", flush=True)
                break

        if i % 20 == 0:
            progress.set_postfix({"loss": f"{loss:.4f}", "psnr": f"{psnr:.2f}"})
        if i % 100 == 0:
            print(f"[splat-apple] iter={i}/{num_iterations} loss={loss:.5f} psnr={psnr:.2f} elapsed={time.time()-t0:.1f}s", flush=True)

        if i > 0 and i % 3000 == 0:
            gaussians.opacities.data[:] = torch.full(
                (gaussians.opacities.shape[0], 1), _logit(opacity_init), dtype=torch.float32, device=device
            )
            print(f"[splat-apple] iter={i} opacity reset", flush=True)

        if scale_log_min is not None or scale_log_max is not None:
            with torch.no_grad():
                gaussians.scales.data = torch.clamp(
                    gaussians.scales.data,
                    min=scale_log_min,
                    max=scale_log_max,
                )

        if adaptive and i > 0 and i % densify_interval == 0 and i <= densify_stop_iter:
            grad_scores_np = grad_accum.cpu().numpy() / max(grad_count, 1)
            gaussians = _adaptive_topology_update(
                gaussians,
                prune_threshold=prune_threshold,
                clone_fraction=clone_fraction,
                max_points=max_points,
                grad_scores=grad_scores_np,
            )
            gaussians.means.requires_grad = gaussians.scales.requires_grad = True
            gaussians.quaternions.requires_grad = gaussians.opacities.requires_grad = True
            gaussians.sh_coeffs.requires_grad = True
            optimizer = _build_optimizer(gaussians, mean_lr_scale=mean_lr_scale)
            grad_accum = torch.zeros(gaussians.means.shape, device=device)
            grad_count = 0

    labels_out = getattr(gaussians, "_labels_np", None)
    # Preserve original scene scale (torch branch) to avoid training-driven
    # shrinking/expansion that produces tiny gaussians. Adjust means and
    # log-scales consistently if necessary.
    try:
        final_means = gaussians.means.detach().cpu().numpy()
        final_scene_scale = max(float(np.linalg.norm(np.std(final_means, axis=0))), 1e-6)
        scale_correction = init_scene_scale / final_scene_scale
        if not np.isfinite(scale_correction) or scale_correction <= 0.0:
            scale_correction = 1.0
        if scale_correction != 1.0:
            with torch.no_grad():
                gaussians.means.data = gaussians.means.data * float(scale_correction)
                gaussians.scales.data = gaussians.scales.data + float(np.log(float(scale_correction)))
    except Exception:
        pass

    _save_layerpano_compatible_ply(out_ply_path, gaussians, labels_np=labels_out, sh_degree=3)
    return out_ply_path


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

def _train_mlx(
    traindata: Dict,
    out_ply_path: str,
    num_iterations: int,
    rasterizer: str,
    adaptive: bool,
    densify_interval: int,
    prune_threshold: float,
    clone_fraction: float,
    max_points: Optional[int],
    downsample_ratio: float,
    repulsion_weight: float,
    mean_lr_scale: float = 1.0,
    opacity_init: float = 0.1,
    opacity_reg_weight: float = 0.005,
    opacity_mean_reg_weight: float = 0.0,
    blur_reg_weight: float = 0.02,
    scale_log_min: float = -7.0,
    scale_log_max: float = -2.8,
    initial_gaussian_params: Optional[Dict[str, np.ndarray]] = None,
    initial_gaussian_labels: Optional[np.ndarray] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    lr_plateau_patience: Optional[int] = None,
    lr_plateau_factor: float = 0.5,
    lr_plateau_min_lr: float = 1e-6,
    freeze_sh_coeffs: bool = False,
) -> str:
    _ensure_mlx_gs_importable()

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_gs.core.gaussians import init_gaussians_from_pcd
    from mlx_gs.training.trainer import train_step

    xyz, rgb, labels, frozen_mask, cameras, targets = _build_training_batch_mlx(traindata, downsample_ratio)

    if initial_gaussian_params is not None:
        init_xyz = np.asarray(initial_gaussian_params.get("means"), dtype=np.float32)
        init_sh = np.asarray(initial_gaussian_params.get("sh_coeffs"), dtype=np.float32)
        if init_xyz.size == 0:
            raise ValueError("initial_gaussian_params has no points")
        if init_sh.ndim == 3 and init_sh.shape[-1] == 3:
            init_rgb = np.clip(init_sh[:, 0, :], 0.0, 1.0).astype(np.float32)
        else:
            init_rgb = np.zeros((init_xyz.shape[0], 3), dtype=np.float32)
        xyz = init_xyz
        rgb = init_rgb
        labels = initial_gaussian_labels

    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.any():
        raise ValueError("Point cloud entirely NaN/Inf.")
    if not np.all(finite_mask):
        xyz = xyz[finite_mask]; rgb = rgb[finite_mask]
        if labels is not None:
            labels = labels[finite_mask]

    init_point_count = int(len(xyz))
    small_object_layer = init_point_count < 25_000
    if small_object_layer:
        adaptive = False
        opacity_reg_weight = 0.0
        opacity_mean_reg_weight = 0.0
        blur_reg_weight = 0.0

    init_scene_scale = max(float(np.linalg.norm(np.std(xyz, axis=0))), 1e-3)
    scale_init = _estimate_log_scales(xyz)
    scale_init = np.clip(scale_init, scale_log_min, scale_log_max)

    gaussians = init_gaussians_from_pcd(
        np.asarray(xyz, dtype=np.float32),
        np.asarray(rgb, dtype=np.float32),
        sh_degree=3,
        scale_init=scale_init,
        opacity_init=opacity_init,
    )

    params = {
        "means":       gaussians.means,
        "scales":      gaussians.scales,
        "quaternions": gaussians.quaternions,
        "opacities":   gaussians.opacities,
        "sh_coeffs":   gaussians.sh_coeffs,
    }

    if initial_gaussian_params is not None:
        params = {k: mx.array(v, dtype=mx.float32) for k, v in initial_gaussian_params.items()}
        if labels is None and initial_gaussian_labels is not None:
            labels = np.asarray(initial_gaussian_labels, dtype=np.int32).reshape(-1)

    def make_optimizers(lr_mult: float = 1.0):
        optimizers = {
            "means":       optim.Adam(learning_rate=(0.00016 * float(mean_lr_scale) * float(lr_mult))),
            "scales":      optim.Adam(learning_rate=(0.0015 * float(lr_mult))),
            "quaternions": optim.Adam(learning_rate=(0.005 * float(lr_mult))),
            "opacities":   optim.Adam(learning_rate=(0.05 * float(lr_mult))),
        }
        if not freeze_sh_coeffs:
            optimizers["sh_coeffs"] = optim.Adam(learning_rate=(0.0025 * float(lr_mult)))
        return optimizers

    current_lr_mult = 1.0
    optimizers = make_optimizers(current_lr_mult)
    max_points_cap = 3_000_000
    if max_points is None:
        init_points = int(gaussians.means.shape[0])
        max_points = int(min(max_points_cap, max(120_000, init_points * 1.75)))
        densify_interval = max(densify_interval, 200)
        clone_fraction = min(clone_fraction, 0.02)
    else:
        max_points = int(max_points)
    max_points = int(min(max_points, max_points_cap))

    if frozen_mask is None:
        frozen_mask = np.zeros((len(xyz),), dtype=bool)
    else:
        frozen_mask = np.asarray(frozen_mask, dtype=bool).reshape(-1)

    progress = tqdm(range(num_iterations), desc="Splat-Apple MLX training")
    t0 = time.time()

    best_loss = float("inf")
    no_improve = 0
    no_improve_lr = 0

    for i in progress:
        cam_idx = random.randint(0, len(cameras) - 1)
        active_sh_degree = 0 if i < max(1, num_iterations // 4) else 1
        loss, _, psnr, _ = train_step(
            params, optimizers, targets[cam_idx], cameras[cam_idx],
            lambda_ssim=0.25, rasterizer_type=rasterizer,
            scale_reg_weight=max(opacity_reg_weight * 1.5, 0.01),
            opacity_reg_weight=opacity_reg_weight,
            opacity_mean_reg_weight=float(opacity_mean_reg_weight),
            blur_reg_weight=blur_reg_weight,
            scale_log_min=scale_log_min,
            scale_log_max=scale_log_max,
            repulsion_weight=repulsion_weight,
            repulsion_min_dist=0.05,
            repulsion_max_samples=512,
            active_sh_degree=active_sh_degree,
        )

        if i % 20 == 0 or i % 100 == 0:
            mx.eval(loss, psnr)

        loss_value = float(loss)
        if loss_value < best_loss - float(early_stop_min_delta or 0.0):
            best_loss = loss_value
            no_improve = 0
            no_improve_lr = 0
        else:
            no_improve += 1
            no_improve_lr += 1

        if lr_plateau_patience is not None and lr_plateau_patience > 0:
            if no_improve_lr >= lr_plateau_patience:
                # Reduce LR by recreating optimizers with a multiplicative factor
                current_lr_mult *= float(lr_plateau_factor)
                # Recreate optimizers with scaled base learning rates
                optimizers = make_optimizers(current_lr_mult)
                no_improve_lr = 0
                print(f"[splat-apple-mlx] lr reduced on plateau (mult={current_lr_mult:.4f})", flush=True)

        if early_stop_patience is not None and early_stop_patience > 0:
            if no_improve >= early_stop_patience:
                print(f"[splat-apple-mlx] early stopping at iter={i} (best_loss={best_loss:.6f})", flush=True)
                break

        if i % 20 == 0:
            progress.set_postfix({"loss": f"{float(loss):.4f}", "psnr": f"{float(psnr):.2f}"})

        if i % 100 == 0:
            print(f"[splat-apple-mlx] iter={i} loss={float(loss):.5f} psnr={float(psnr):.2f} elapsed={time.time()-t0:.1f}s", flush=True)

        if i > 0 and i % 3000 == 0:
            reset_val = float(_logit(opacity_init))
            params["opacities"] = mx.full(params["opacities"].shape, reset_val, dtype=mx.float32)
            optimizers = make_optimizers()
            print(f"[splat-apple-mlx] iter={i} opacity reset", flush=True)

        if scale_log_min is not None or scale_log_max is not None:
            params["scales"] = mx.clip(params["scales"], scale_log_min, scale_log_max)

        # Densify only during the first 60% of iterations.
        if adaptive and i > 0 and i % densify_interval == 0 and i <= int(num_iterations * 0.6):
            mx.eval(params)
            np_params = {k: np.asarray(v) for k, v in params.items()}
            effective_prune = prune_threshold
            np_params, labels, frozen_mask = _adaptive_topology_update_mlx(
                np_params, labels,
                frozen_mask=frozen_mask,
                prune_threshold=effective_prune,
                clone_fraction=clone_fraction,
                max_points=max_points,
                scale_log_max=scale_log_max,
                split_scale_factor=2.0,
            )
            params     = {k: mx.array(v, dtype=mx.float32) for k, v in np_params.items()}
            optimizers = make_optimizers()

    # Preserve the original point cloud scale to avoid MLX shrinking/expanding the scene.
    final_means = np.asarray(params["means"], dtype=np.float32)
    final_scene_scale = max(float(np.linalg.norm(np.std(final_means, axis=0))), 1e-6)
    scale_correction = init_scene_scale / final_scene_scale
    if not np.isfinite(scale_correction) or scale_correction <= 0.0:
        scale_correction = 1.0

    params["means"] = params["means"] * float(scale_correction)
    if scale_correction != 1.0:
        params["scales"] = params["scales"] + mx.log(mx.array(float(scale_correction), dtype=mx.float32))
    params["scales"] = mx.clip(params["scales"], scale_log_min, scale_log_max)

    final_params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    if small_object_layer and "opacities" in final_params:
        final_params["opacities"] = np.maximum(final_params["opacities"], _logit(0.12)).astype(np.float32)
    _save_layerpano_compatible_ply_mlx(
        out_ply_path,
        final_params,
        labels=labels,
        scale_log_min=scale_log_min,
        scale_log_max=scale_log_max,
        sh_degree=3,
    )
    return out_ply_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_with_splat_apple(
    traindata: Dict,
    out_ply_path: str,
    num_iterations: int,
    rasterizer: str = "cpp",
    device: str = "mps",
    training_backend: str = "mlx",
    adaptive: bool = True,
    densify_interval: int = 200,
    prune_threshold: float = 0.05,
    clone_fraction: float = 0.02,
    max_points: Optional[int] = None,
    downsample_ratio: float = 0.1,
    repulsion_weight: float = 1e-4,
    mean_lr_scale: float = 1.0,
    prev_gaussian_params: Optional[Dict[str, np.ndarray]] = None,
    prev_gaussian_labels: Optional[np.ndarray] = None,
    training_profile: Optional[str] = None,
    initial_gaussian_params: Optional[Dict[str, np.ndarray]] = None,
    initial_gaussian_labels: Optional[np.ndarray] = None,
    opacity_init: float = 0.1,
    opacity_reg_weight: float = 0.005,
    opacity_mean_reg_weight: float = 0.0,
    blur_reg_weight: float = 0.02,
    scale_log_min: float = -7.0,
    scale_log_max: float = 0.15,
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    lr_plateau_patience: Optional[int] = None,
    lr_plateau_factor: float = 0.5,
    lr_plateau_min_lr: float = 1e-6,
    freeze_sh_coeffs: bool = False,
) -> str:
    """
    Train gaussian splatting with support for frozen gaussian re-activation from previous layer.
    
    Args:
        traindata: Training data dict (points, colors, cameras, etc.)
        out_ply_path: Output PLY file path
        num_iterations: Number of training iterations
        prev_gaussian_params: Optional gaussian parameters from previous layer (for ray-matching transfer)
        prev_gaussian_labels: Optional labels from previous layer gaussians
    """
    # Transfer frozen gaussians from previous layer if available
    traindata["frozen_mask"] = np.zeros((len(traindata.get("pcd_points", [])),), dtype=bool)

    if prev_gaussian_params is not None and "pcd_points" in traindata:
        new_xyz = np.asarray(traindata["pcd_points"], dtype=np.float32)
        new_rgb = np.asarray(traindata["pcd_colors"], dtype=np.float32)
        new_labels = traindata.get("pcd_labels", None)
        
        merged_xyz, merged_rgb, merged_labels, merged_frozen_mask = _transfer_frozen_gaussians(
            frozen_params=prev_gaussian_params,
            frozen_labels=prev_gaussian_labels,
            new_xyz=new_xyz,
            new_rgb=new_rgb,
            new_labels=new_labels,
            beta3=8.0,
        )
        
        traindata["pcd_points"] = merged_xyz
        traindata["pcd_colors"] = merged_rgb
        traindata["frozen_mask"] = merged_frozen_mask
        if merged_labels is not None:
            traindata["pcd_labels"] = merged_labels

    densify_interval, prune_threshold, clone_fraction, opacity_init, opacity_reg_weight, scale_log_min, scale_log_max = _apply_training_profile(
        training_profile,
        densify_interval,
        prune_threshold,
        clone_fraction,
        opacity_init,
        opacity_reg_weight,
        scale_log_min,
        scale_log_max,
    )

    if max_points is None and training_profile in ("per_layer", "deva_instances"):
        max_points = 3_000_000

    if training_backend == "mlx":
        # MLX is very fast, but needs stricter scale control than torch_gs.
        # PLY scale_* fields are log-scales; values near 0 become unit-sized splats.
        mlx_scale_cap = -0.6 if training_profile == "final_fill" else -0.8
        scale_log_max = min(float(scale_log_max), mlx_scale_cap)
        opacity_reg_weight = min(max(float(opacity_reg_weight), 0.001), 0.003)
        opacity_mean_reg_weight = 0.0
        blur_reg_weight = max(float(blur_reg_weight), 0.015)
        densify_interval = min(int(densify_interval), 140)
        clone_fraction = max(float(clone_fraction), 0.08)
    
    if training_backend == "mlx":
        return _train_mlx(
            traindata=traindata,
            out_ply_path=out_ply_path,
            num_iterations=num_iterations,
            rasterizer=rasterizer,
            adaptive=adaptive,
            densify_interval=densify_interval,
            prune_threshold=prune_threshold,
            clone_fraction=clone_fraction,
            max_points=max_points,
            downsample_ratio=downsample_ratio,
            repulsion_weight=repulsion_weight,
            mean_lr_scale=mean_lr_scale,
            opacity_init=opacity_init,
            opacity_reg_weight=opacity_reg_weight,
            opacity_mean_reg_weight=opacity_mean_reg_weight,
            blur_reg_weight=blur_reg_weight,
            scale_log_min=scale_log_min,
            scale_log_max=scale_log_max,
            initial_gaussian_params=initial_gaussian_params,
            initial_gaussian_labels=initial_gaussian_labels,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            lr_plateau_patience=lr_plateau_patience,
            lr_plateau_factor=lr_plateau_factor,
            lr_plateau_min_lr=lr_plateau_min_lr,
            freeze_sh_coeffs=freeze_sh_coeffs,
        )
    else:
        return _train_torch_gs(
            traindata=traindata,
            out_ply_path=out_ply_path,
            num_iterations=num_iterations,
            rasterizer=rasterizer,
            device=device,
            adaptive=adaptive,
            densify_interval=densify_interval,
            prune_threshold=prune_threshold,
            clone_fraction=clone_fraction,
            max_points=max_points,
            downsample_ratio=downsample_ratio,
            mean_lr_scale=mean_lr_scale,
            opacity_init=opacity_init,
            opacity_reg_weight=opacity_reg_weight,
            opacity_mean_reg_weight=opacity_mean_reg_weight,
            scale_log_min=scale_log_min,
            scale_log_max=scale_log_max,
            initial_gaussian_params=initial_gaussian_params,
            initial_gaussian_labels=initial_gaussian_labels,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            lr_plateau_patience=lr_plateau_patience,
            lr_plateau_factor=lr_plateau_factor,
            lr_plateau_min_lr=lr_plateau_min_lr,
        )


def _coerce_vertex_dtype(vertex: np.ndarray, dtype) -> np.ndarray:
    if vertex.dtype == dtype:
        return vertex
    coerced = np.zeros(vertex.shape[0], dtype=dtype)
    for name in dtype.names:
        if name in vertex.dtype.names:
            coerced[name] = vertex[name]
        else:
            coerced[name] = 0
    return coerced


def merge_ply_layers(
    ply_paths: List[str],
    out_path: str,
    voxel_size: float = 0.0005,
    min_opacity: float = 0.001,
    max_points: Optional[int] = None,
) -> str:
    if not ply_paths:
        raise ValueError("No PLY paths provided")

    base_vertex = PlyData.read(ply_paths[0])["vertex"].data
    base_dtype = base_vertex.dtype
    merged_parts = []

    for path in ply_paths:
        ply = PlyData.read(path)
        vertex = ply["vertex"].data
        vertex = _coerce_vertex_dtype(vertex, base_dtype)

        finite = np.isfinite(vertex["x"]) & np.isfinite(vertex["y"]) & np.isfinite(vertex["z"])
        if "opacity" in vertex.dtype.names:
            finite &= np.isfinite(vertex["opacity"])
            if min_opacity is not None:
                finite &= vertex["opacity"] >= float(min_opacity)
        vertex = vertex[finite]
        if vertex.size:
            merged_parts.append(vertex)

    if not merged_parts:
        raise ValueError("No vertices left after filtering")

    merged = np.concatenate(merged_parts, axis=0)

    if voxel_size and voxel_size > 0:
        coords = np.stack([merged["x"], merged["y"], merged["z"]], axis=1)
        min_pt = coords.min(axis=0)
        vox = np.floor((coords - min_pt) / float(voxel_size)).astype(np.int64)

        buckets = {}
        for i in range(len(merged)):
            key = (int(vox[i, 0]), int(vox[i, 1]), int(vox[i, 2]))
            buckets.setdefault(key, []).append(i)

        out = np.zeros(len(buckets), dtype=merged.dtype)
        for j, idxs in enumerate(buckets.values()):
            idxs = np.asarray(idxs, dtype=np.int64)
            if "opacity" in merged.dtype.names:
                opacity = np.asarray(merged["opacity"][idxs], dtype=np.float32).reshape(-1)
                if "scale_0" in merged.dtype.names and "scale_1" in merged.dtype.names and "scale_2" in merged.dtype.names:
                    scale_vals = np.stack([
                        np.asarray(merged["scale_0"][idxs], dtype=np.float32).reshape(-1),
                        np.asarray(merged["scale_1"][idxs], dtype=np.float32).reshape(-1),
                        np.asarray(merged["scale_2"][idxs], dtype=np.float32).reshape(-1),
                    ], axis=1)
                    scale_mean = np.exp(scale_vals).mean(axis=1)
                    score = opacity * scale_mean
                else:
                    score = opacity
                k = idxs[int(np.argmax(score))]
            else:
                print(f"[merge_ply_layers] warning: missing opacity field, using first sample for voxel with {len(idxs)} points", flush=True)
                k = idxs[0]

            out[j] = merged[k]
        merged = out

    if max_points is not None and len(merged) > int(max_points):
        opacity = merged["opacity"] if "opacity" in merged.dtype.names else np.zeros((len(merged),), dtype=np.float32)
        top_idx = np.argsort(opacity)[-int(max_points):]
        merged = merged[top_idx]

    PlyData([PlyElement.describe(merged, "vertex")]).write(out_path)
    return out_path


def global_refine_after_merge(
    traindata: Dict,
    merged_ply_path: str,
    out_ply_path: str,
    num_iterations: int = 300,
    rasterizer: str = "cpp",
    device: str = "mps",
    training_backend: str = "mlx",
    adaptive: bool = False,
    max_points: Optional[int] = None,
    downsample_ratio: float = 1.0,
    repulsion_weight: float = 1e-4,
) -> str:
    params, labels = extract_gaussian_params_from_ply(merged_ply_path)
    return train_with_splat_apple(
        traindata=traindata,
        out_ply_path=out_ply_path,
        num_iterations=num_iterations,
        rasterizer=rasterizer,
        device=device,
        training_backend=training_backend,
        adaptive=adaptive,
        densify_interval=180,
        prune_threshold=0.01,
        clone_fraction=0.04,
        max_points=max_points,
        downsample_ratio=downsample_ratio,
        repulsion_weight=repulsion_weight,
        training_profile="final_fill",
        initial_gaussian_params=params,
        initial_gaussian_labels=labels,
        freeze_sh_coeffs=True,
    )
