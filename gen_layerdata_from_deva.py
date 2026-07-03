"""Generate object-aware LayerPano training data from segmentation masks.

This script:
- runs GroundingDINO + SAM on the panorama, or DEVA + SAM on perspective frames
- groups masks by instance id
- projects instance ids to the 3D point cloud
- writes per-instance traindata/layerK folders with masked frames and PLY files
- optionally adds a background layer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UTILITY_PATH = PROJECT_ROOT / "submodules" / "360monodepth" / "code" / "python" / "src" / "utility"
if str(UTILITY_PATH) not in sys.path:
    sys.path.insert(0, str(UTILITY_PATH))

import spherical_coordinates as sc

from utils.depth_alignment import Pano_depth_estimation
import utils.pano_utils.Equirec2Perspec as E2P
from utils.trajectory import gcd_pose_gs
from utils.deva_instance_segmentation import (
    DEVA_AVAILABLE,
    segment_perspective_frames,
    propagate_frame_instances_to_3d,
)
from utils.semantic_instance_detection import detect_objects_grounding_then_sam_on_panorama


def _numeric_suffix_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return (int(digits) if digits else -1, stem)


def _frame_idx_to_theta_phi(
    frame_idx: int,
    n: int = 8,
    phi_bands: Optional[List[float]] = None,
) -> Optional[Tuple[float, float]]:
    if frame_idx < 0:
        return None
    if phi_bands is None:
        phi_bands = [80.0, 67.5, 45.0, 0.0, -45.0, -67.5, -80.0]

    main_count = int(n) * len(phi_bands)
    if frame_idx == main_count:
        return 0.0, 90.0
    if frame_idx == main_count + 1:
        return 0.0, -90.0
    if frame_idx > main_count + 1:
        return None

    band = frame_idx // n
    pos = frame_idx % n
    if band < 0 or band >= len(phi_bands):
        return None
    phi = float(phi_bands[band])
    theta = (360.0 / float(n)) * float(pos)
    return theta, phi


def _project_frame_labels_to_equirect(
    frame_labels: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    out_h: int,
    out_w: int,
    fov_deg: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame_labels.shape[:2]
    w_fov = float(fov_deg)
    h_fov = float(h) / float(w) * w_fov
    w_len = np.tan(np.radians(w_fov / 2.0))
    h_len = np.tan(np.radians(h_fov / 2.0))

    x_deg, y_deg = np.meshgrid(
        np.linspace(-180.0, 180.0, out_w),
        np.linspace(90.0, -90.0, out_h),
    )

    x_map = np.cos(np.radians(x_deg)) * np.cos(np.radians(y_deg))
    y_map = np.sin(np.radians(x_deg)) * np.cos(np.radians(y_deg))
    z_map = np.sin(np.radians(y_deg))
    xyz = np.stack((x_map, y_map, z_map), axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    r1, _ = cv2.Rodrigues(z_axis * np.radians(theta_deg))
    r2, _ = cv2.Rodrigues(np.dot(r1, y_axis) * np.radians(-phi_deg))
    r1 = np.linalg.inv(r1)
    r2 = np.linalg.inv(r2)

    xyz = xyz.reshape([out_h * out_w, 3]).T
    xyz = np.dot(r2, xyz)
    xyz = np.dot(r1, xyz).T.reshape([out_h, out_w, 3])

    front = xyz[:, :, 0] > 0
    x0 = np.where(front, xyz[:, :, 0], 1.0)
    xyz = xyz / np.repeat(x0[:, :, np.newaxis], 3, axis=2)
    inside = (
        (xyz[:, :, 1] > -w_len) & (xyz[:, :, 1] < w_len) &
        (xyz[:, :, 2] > -h_len) & (xyz[:, :, 2] < h_len)
    )

    valid = front & inside

    lon_map = np.where(
        inside,
        (xyz[:, :, 1] + w_len) / (2.0 * w_len) * float(w),
        0.0,
    ).astype(np.float32)
    lat_map = np.where(
        inside,
        (-xyz[:, :, 2] + h_len) / (2.0 * h_len) * float(h),
        0.0,
    ).astype(np.float32)

    proj = cv2.remap(
        frame_labels.astype(np.float32),
        lon_map,
        lat_map,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int32)
    proj[~valid] = 0
    return proj, valid


def _postprocess_equirect_labels(
    label_map: np.ndarray,
    kernel_size: int = 15,
    fill_holes: bool = True,
    dilate_iters: int = 1,
) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.int32)
    out = label_map.copy()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    labels = [int(x) for x in np.unique(label_map) if int(x) > 0]

    for lab in labels:
        m = (label_map == lab).astype(np.uint8) * 255

        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        kernel_size_large = max(15, int(kernel_size) * 3)
        if kernel_size_large > kernel_size:
            kernel_large = np.ones((kernel_size_large, kernel_size_large), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_large)

        if fill_holes:
            h, w = m.shape
            flood = m.copy()
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            holes = cv2.bitwise_not(flood)
            m = cv2.bitwise_or(m, holes)

        if dilate_iters > 0:
            m = cv2.dilate(m, kernel, iterations=int(dilate_iters))

        out[(m > 0) & (out == 0)] = lab

    return out


def _resize_mask_to_shape(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _mask_has_training_content(mask: np.ndarray, min_pixels: int = 1) -> bool:
    return int(np.asarray(mask, dtype=bool).sum()) >= int(min_pixels)


def _project_erp_mask_to_frame(
    erp_mask: np.ndarray,
    frame_idx: int,
    out_h: int,
    out_w: int,
    n: int = 8,
    phi_bands: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    ang = _frame_idx_to_theta_phi(int(frame_idx), n=n, phi_bands=phi_bands)
    if ang is None:
        return None
    theta_deg, phi_deg = ang
    erp_u8 = (erp_mask.astype(np.uint8) * 255)
    erp_rgb = np.repeat(erp_u8[..., None], 3, axis=2)
    equ = E2P.Equirectangular(erp_rgb)
    pers = equ.GetPerspective(90, theta_deg, phi_deg, int(out_h), int(out_w))
    if pers.ndim == 3:
        pers = pers[..., 0]
    return pers > 127


def _frames_to_equirect_instance_map(
    instance_maps: Dict[int, np.ndarray],
    out_h: int,
    out_w: int,
    n: int = 8,
    phi_bands: Optional[List[float]] = None,
    min_votes: int = 2,
    postprocess: bool = True,
    kernel_size: int = 3,
) -> np.ndarray:
    counts_by_label: Dict[int, np.ndarray] = {}
    total_valid = np.zeros((out_h, out_w), dtype=np.uint16)

    for fidx, fmap in instance_maps.items():
        ang = _frame_idx_to_theta_phi(int(fidx), n=n, phi_bands=phi_bands)
        if ang is None:
            continue

        theta_deg, phi_deg = ang
        proj, valid = _project_frame_labels_to_equirect(
            np.asarray(fmap, dtype=np.int32),
            theta_deg,
            phi_deg,
            out_h,
            out_w,
            fov_deg=90.0,
        )

        total_valid[valid] += 1

        labels = np.unique(proj[valid])
        labels = labels[labels > 0]

        for lab in labels:
            mask = (proj == int(lab)) & valid
            if not mask.any():
                continue
            arr = counts_by_label.get(int(lab))
            if arr is None:
                arr = np.zeros((out_h, out_w), dtype=np.uint16)
                counts_by_label[int(lab)] = arr
            arr[mask] += 1

    if not counts_by_label:
        return np.zeros((out_h, out_w), dtype=np.int32)

    best_label = np.zeros((out_h, out_w), dtype=np.int32)
    best_count = np.zeros((out_h, out_w), dtype=np.uint16)

    for lab, cnt in counts_by_label.items():
        g = cnt > best_count
        if g.any():
            best_label[g] = int(lab)
            best_count[g] = cnt[g]

    best_label[best_count < int(min_votes)] = 0
    best_label[total_valid == 0] = 0

    if postprocess:
        best_label = _postprocess_equirect_labels(
            best_label,
            kernel_size=kernel_size,
            fill_holes=True,
            dilate_iters=1,
        )

    best_label[total_valid == 0] = 0
    return best_label


@dataclass
class InstanceStats:
    instance_id: int
    frame_count: int
    total_pixels: int
    points_3d: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_rgb(input_dir: Path) -> np.ndarray:
    rgb_path = input_dir / "rgb.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing {rgb_path}")
    return np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)


def _load_depth(input_dir: Path, depth_model: str, force: bool) -> np.ndarray:
    candidates = [
        input_dir / "depth.npy",
        input_dir / "layering" / "depth.npy",
    ]

    if not force:
        for path in candidates:
            if path.exists():
                return np.load(path)

    rgb = _load_rgb(input_dir)
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    estimator = Pano_depth_estimation(
        rgb.shape[0], rgb.shape[1], str(input_dir), device, depth_model=depth_model
    )
    depth = np.asarray(estimator.get_panodepth(rgb))
    _ensure_dir(input_dir / "layering")
    np.save(input_dir / "layering" / "depth.npy", depth)
    return depth


def _generate_frames(
    pano_rgb: np.ndarray,
    frames_dir: Path,
    n: int = 8,
    phi_bands: Optional[List[float]] = None,
) -> None:
    _ensure_dir(frames_dir)
    pano_h = int(pano_rgb.shape[0])
    pers_size = int((pano_h / 1024.0) * 512)
    pers_size = max(128, pers_size)

    if phi_bands is None:
        phi_bands = [80.0, 67.5, 45.0, 0.0, -45.0, -67.5, -80.0]
    theta = [(360.0 / n) * i for i in range(n)]
    theta = theta * len(phi_bands)
    phi = []
    for band in phi_bands:
        phi += [float(band)] * n

    equ = E2P.Equirectangular(pano_rgb)
    for i, (th, ph) in enumerate(zip(theta, phi)):
        pers_img = equ.GetPerspective(90, th, ph, pers_size, pers_size)
        pers_img = np.clip(pers_img, 0, 255).astype(np.uint8)
        Image.fromarray(pers_img).save(frames_dir / f"rgb_{i}.png")
        np.save(frames_dir / f"transform_matrix_{i}.npy", gcd_pose_gs(th, ph))

    zenith_idx = len(theta)
    zenith_img = equ.GetPerspective(90, 0.0, 90.0, pers_size, pers_size)
    zenith_img = np.clip(zenith_img, 0, 255).astype(np.uint8)
    Image.fromarray(zenith_img).save(frames_dir / f"rgb_{zenith_idx}.png")
    np.save(frames_dir / f"transform_matrix_{zenith_idx}.npy", gcd_pose_gs(0.0, 90.0))

    nadir_idx = zenith_idx + 1
    nadir_img = equ.GetPerspective(90, 0.0, -90.0, pers_size, pers_size)
    nadir_img = np.clip(nadir_img, 0, 255).astype(np.uint8)
    Image.fromarray(nadir_img).save(frames_dir / f"rgb_{nadir_idx}.png")
    np.save(frames_dir / f"transform_matrix_{nadir_idx}.npy", gcd_pose_gs(0.0, -90.0))


def _ensure_frames_dir(
    input_dir: Path,
    preferred: Optional[Path],
    n_views: int = 8,
    phi_bands: Optional[List[float]] = None,
) -> Path:
    if preferred is not None and preferred.exists():
        return preferred

    if phi_bands is None:
        phi_bands = [80.0, 67.5, 45.0, 0.0, -45.0, -67.5, -80.0]
    expected_count = int(n_views) * len(phi_bands) + 2

    default_dir = input_dir / "traindata" / "deva_frames" / "frames"
    existing_frames = sorted(default_dir.glob("rgb_*.png"), key=_numeric_suffix_key)
    existing_poses = sorted(default_dir.glob("transform_matrix_*.npy"), key=_numeric_suffix_key)
    if len(existing_frames) == expected_count and len(existing_poses) >= expected_count:
        return default_dir
    if existing_frames:
        print(
            "[frames] Existing generated frames do not match requested view grid "
            f"({len(existing_frames)} found, {expected_count} expected); regenerating."
        )
        for path in existing_frames + existing_poses:
            path.unlink(missing_ok=True)

    pano_rgb = _load_rgb(input_dir)
    _generate_frames(pano_rgb, default_dir, n=n_views, phi_bands=phi_bands)
    return default_dir


def _erp_pointcloud(depth: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pixel_x, pixel_y = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
    theta, phi = sc.erp2sph([pixel_x, pixel_y])

    x = (depth * np.cos(phi) * np.sin(theta)).flatten()
    y = -(depth * np.sin(phi)).flatten()
    z = (depth * np.cos(phi) * np.cos(theta)).flatten()

    r = rgb[pixel_y, pixel_x, 0].flatten()
    g = rgb[pixel_y, pixel_x, 1].flatten()
    b = rgb[pixel_y, pixel_x, 2].flatten()

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = np.stack([r, g, b], axis=1).astype(np.uint8)
    return xyz, colors


def _write_point_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> None:
    if xyz.size == 0:
        raise ValueError(f"No points to write: {path}")

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels.shape[0] != xyz.shape[0]:
            raise ValueError("labels length mismatch")

    dtype_list = [
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
    ]
    if labels is not None:
        dtype_list.append(("label", np.int32))

    vertex = np.zeros(xyz.shape[0], dtype=dtype_list)
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]
    if labels is not None:
        vertex["label"] = labels

    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def _filter_instances(
    instance_maps: Dict[int, np.ndarray],
    min_frames: int,
    min_total_pixels: int,
) -> Dict[int, InstanceStats]:
    stats: Dict[int, InstanceStats] = {}
    for fmap in instance_maps.values():
        ids, counts = np.unique(fmap, return_counts=True)
        for inst_id, cnt in zip(ids.tolist(), counts.tolist()):
            if inst_id <= 0:
                continue
            if inst_id not in stats:
                stats[inst_id] = InstanceStats(inst_id, 0, 0, 0)
            stats[inst_id].frame_count += 1
            stats[inst_id].total_pixels += int(cnt)

    return {
        k: v for k, v in stats.items()
        if v.frame_count >= min_frames and v.total_pixels >= min_total_pixels
    }


def _filter_erp_instances(
    erp_masks: Dict[int, np.ndarray],
    instance_maps: Dict[int, np.ndarray],
    min_total_pixels: int,
) -> Dict[int, InstanceStats]:
    stats: Dict[int, InstanceStats] = {}
    for inst_id, mask in erp_masks.items():
        inst_id = int(inst_id)
        if inst_id <= 0:
            continue
        total_pixels = int(np.asarray(mask, dtype=bool).sum())
        if total_pixels < int(min_total_pixels):
            continue
        frame_count = 0
        for fmap in instance_maps.values():
            if np.any(np.asarray(fmap) == inst_id):
                frame_count += 1
        stats[inst_id] = InstanceStats(
            instance_id=inst_id,
            frame_count=frame_count,
            total_pixels=total_pixels,
            points_3d=0,
        )
    return stats


def _parse_label_filter(value: Optional[str]) -> set[str]:
    if value is None:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _label_matches_filter(label: Optional[str], filters: set[str]) -> bool:
    if not label or not filters:
        return False
    tokens = {part.strip().lower() for part in str(label).replace("/", " ").split() if part.strip()}
    label_lower = str(label).strip().lower()
    return label_lower in filters or bool(tokens & filters)


def _normalize_group_label(label: Optional[str], fallback: str) -> str:
    raw = str(label or "").strip().lower()
    if not raw:
        return fallback
    raw = raw.replace("/", " ").replace("-", " ").replace("_", " ")
    tokens = [part for part in raw.split() if part]
    if not tokens:
        return fallback
    aliases = {
        "trees": "tree",
        "bushes": "bush",
        "plants": "plant",
        "persons": "person",
        "people": "person",
        "roads": "road",
        "pavements": "pavement",
        "sidewalk": "pavement",
        "sidewalks": "pavement",
        "leaves": "leaves",
        "leaf": "leaves",
    }
    return aliases.get(tokens[0], tokens[0])


def _group_is_semantic(group: dict, names: set[str]) -> bool:
    label = str(group.get("group_label", "")).strip().lower()
    tokens = {part for part in label.replace("/", " ").replace("-", " ").replace("_", " ").split() if part}
    return label in names or bool(tokens & names)


def _nearest_layer_for_component(
    layer_map: np.ndarray,
    component_mask: np.ndarray,
    exclude_layers: Optional[set[int]] = None,
    max_radius: int = 96,
    min_fraction: float = 0.0,
) -> Optional[int]:
    exclude_layers = exclude_layers or set()
    comp = np.asarray(component_mask, dtype=bool)
    if not comp.any():
        return None

    h, w = comp.shape
    max_radius = max(3, min(int(max_radius), max(h, w)))
    for radius in (3, 5, 9, 15, 25, 41, 65, max_radius):
        kernel = np.ones((radius, radius), np.uint8)
        dilated = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & (~comp)
        neighbor_values = layer_map[ring]
        neighbor_values = neighbor_values[neighbor_values >= 0]
        if exclude_layers:
            neighbor_values = np.array([v for v in neighbor_values.tolist() if int(v) not in exclude_layers], dtype=np.int32)
        if neighbor_values.size:
            values, counts = np.unique(neighbor_values.astype(np.int32), return_counts=True)
            best_idx = int(np.argmax(counts))
            if float(counts[best_idx]) / float(max(1, int(counts.sum()))) < float(min_fraction):
                return None
            return int(values[best_idx])
    return None


def _fill_unassigned_by_nearest_layer(layer_map: np.ndarray) -> Tuple[np.ndarray, int]:
    out = np.asarray(layer_map, dtype=np.int32).copy()
    unknown = out < 0
    if not unknown.any() or not (out >= 0).any():
        return out, 0

    n_labels, cc = cv2.connectedComponents(unknown.astype(np.uint8), connectivity=8)
    filled_pixels = 0
    for comp_idx in range(1, n_labels):
        comp = cc == comp_idx
        target = _nearest_layer_for_component(out, comp)
        if target is None:
            continue
        out[comp] = int(target)
        filled_pixels += int(comp.sum())
    return out, filled_pixels


def _nearest_instance_for_component(
    label_map: np.ndarray,
    component_mask: np.ndarray,
    allowed_ids: np.ndarray,
    max_radius: int = 128,
) -> Optional[int]:
    comp = np.asarray(component_mask, dtype=bool)
    if not comp.any():
        return None
    allowed = set(int(x) for x in np.asarray(allowed_ids, dtype=np.int32).tolist())
    if not allowed:
        return None

    h, w = comp.shape
    max_radius = max(3, min(int(max_radius), max(h, w)))
    for radius in (3, 5, 9, 15, 25, 41, 65, 97, max_radius):
        kernel = np.ones((radius, radius), np.uint8)
        dilated = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & (~comp)
        neighbor_values = label_map[ring]
        neighbor_values = np.array([v for v in neighbor_values.tolist() if int(v) in allowed], dtype=np.int32)
        if neighbor_values.size:
            values, counts = np.unique(neighbor_values, return_counts=True)
            return int(values[int(np.argmax(counts))])
    return None


def _labels_from_consolidated_layer_map(
    original_labels: np.ndarray,
    layer_map: np.ndarray,
    layer_groups: List[dict],
) -> np.ndarray:
    labels_out = np.zeros_like(original_labels, dtype=np.int32)
    for layer_idx, group in enumerate(layer_groups):
        group_ids = np.array(group["instance_ids"], dtype=np.int32)
        target_mask = layer_map == int(layer_idx)
        if not target_mask.any():
            continue

        keep_original = target_mask & np.isin(original_labels, group_ids)
        labels_out[keep_original] = original_labels[keep_original]

        needs_label = target_mask & (~keep_original)
        if not needs_label.any():
            continue

        fallback_id = int(group_ids[0])
        n_labels, cc = cv2.connectedComponents(needs_label.astype(np.uint8), connectivity=8)
        for comp_idx in range(1, n_labels):
            comp = cc == comp_idx
            inst_id = _nearest_instance_for_component(original_labels, comp, group_ids)
            labels_out[comp] = int(inst_id if inst_id is not None else fallback_id)

    return labels_out


def _consolidate_layer_label_map(
    labels_2d: np.ndarray,
    layer_groups: List[dict],
    pano_rgb: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Make layer assignment dense and remove obvious semantic/fragment errors."""
    h, w = labels_2d.shape[:2]
    layer_map = np.full((h, w), -1, dtype=np.int32)
    for layer_idx, group in enumerate(layer_groups):
        ids = np.array(group["instance_ids"], dtype=np.int32)
        layer_map[np.isin(labels_2d, ids)] = int(layer_idx)

    cleanup = {
        "sky_pruned_pixels": 0,
        "small_component_reassigned_pixels": 0,
        "residual_reassigned_pixels": 0,
    }

    sky_layers = {
        idx for idx, group in enumerate(layer_groups)
        if _group_is_semantic(group, {"sky"})
    }
    if sky_layers:
        horizon_cutoff = int(round(h * 0.62))
        upper_anchor = int(round(h * 0.20))
        for sky_idx in sorted(sky_layers):
            sky_mask = layer_map == sky_idx
            if not sky_mask.any():
                continue

            remove = np.zeros_like(sky_mask, dtype=bool)
            remove[horizon_cutoff:, :] |= sky_mask[horizon_cutoff:, :]

            n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(sky_mask.astype(np.uint8), connectivity=8)
            for comp_idx in range(1, n_labels):
                y_top = int(stats[comp_idx, cv2.CC_STAT_TOP])
                centroid_y = float(centroids[comp_idx][1])
                if y_top > upper_anchor and centroid_y > h * 0.48:
                    remove |= cc == comp_idx

            if remove.any():
                n_rm, rm_cc = cv2.connectedComponents(remove.astype(np.uint8), connectivity=8)
                for comp_idx in range(1, n_rm):
                    comp = rm_cc == comp_idx
                    target = _nearest_layer_for_component(layer_map, comp, exclude_layers={sky_idx}, min_fraction=0.30)
                    if target is None:
                        layer_map[comp] = -1
                    else:
                        layer_map[comp] = int(target)
                    cleanup["sky_pruned_pixels"] += int(comp.sum())

    total_pixels = int(h * w)
    small_abs = max(256, min(6000, int(round(total_pixels * 0.0008))))
    for layer_idx in range(len(layer_groups)):
        mask = layer_map == layer_idx
        layer_area = int(mask.sum())
        if layer_area <= 0:
            continue
        rel_limit = max(1, int(round(layer_area * 0.025)))
        area_limit = max(small_abs, rel_limit)
        n_labels, cc, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            continue
        largest_comp_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for comp_idx in range(1, n_labels):
            if comp_idx == largest_comp_idx:
                continue
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area > area_limit:
                continue
            comp = cc == comp_idx
            target = _nearest_layer_for_component(layer_map, comp, exclude_layers={layer_idx}, min_fraction=0.60)
            if target is None:
                continue
            layer_map[comp] = int(target)
            cleanup["small_component_reassigned_pixels"] += area

    layer_map, filled = _fill_unassigned_by_nearest_layer(layer_map)
    cleanup["residual_reassigned_pixels"] = int(filled)

    if (layer_map < 0).any() and (layer_map >= 0).any():
        # Last resort for rare isolated pixels: iteratively grow assigned labels.
        for _ in range(max(h, w)):
            unknown = layer_map < 0
            if not unknown.any():
                break
            changed = False
            for layer_idx in range(len(layer_groups)):
                mask = layer_map == layer_idx
                if not mask.any():
                    continue
                grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                claim = grown & unknown
                if claim.any():
                    layer_map[claim] = int(layer_idx)
                    cleanup["residual_reassigned_pixels"] += int(claim.sum())
                    changed = True
            if not changed:
                break

    return layer_map, cleanup


def _masked_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    out = image.copy()
    out[~mask] = 0
    return out


def _save_erp_outputs(
    layer_dir: Path,
    pano_rgb: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> None:
    _ensure_dir(layer_dir)
    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8).save(layer_dir / f"{prefix}_erp_mask.png")
    Image.fromarray(_masked_rgb(pano_rgb, mask)).save(layer_dir / f"{prefix}_erp_rgb.png")


def _save_layer_panorama_overlay(
    pano_rgb: np.ndarray,
    layer_masks: List[Tuple[int, np.ndarray]],
    out_path: Path,
    alpha: float = 0.42,
) -> None:
    base = np.asarray(pano_rgb, dtype=np.uint8)
    overlay = base.astype(np.float32)
    if not layer_masks:
        Image.fromarray(base).save(out_path)
        return

    colors = np.array(
        [
            [255, 99, 71],
            [30, 144, 255],
            [60, 179, 113],
            [238, 130, 238],
            [255, 215, 0],
            [70, 130, 180],
            [255, 105, 180],
            [46, 139, 87],
            [255, 140, 0],
            [123, 104, 238],
            [255, 160, 122],
            [32, 178, 170],
        ],
        dtype=np.float32,
    )

    for idx, (_, mask) in enumerate(sorted(layer_masks, key=lambda item: item[0])):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        color = colors[idx % len(colors)]
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def _clear_generated_deva_outputs(save_path: Path) -> None:
    traindata_dir = save_path / "traindata"
    if not traindata_dir.exists():
        return

    for child in traindata_dir.glob("layer*"):
        if child.is_dir():
            shutil.rmtree(child)
    for name in ["deva_instances.json", "layer_mask_visualization.png"]:
        candidate = traindata_dir / name
        if candidate.exists():
            candidate.unlink()
    deva_instances_dir = traindata_dir / "deva_instances"
    if deva_instances_dir.exists():
        shutil.rmtree(deva_instances_dir)


def generate_layers_from_deva(
    input_dir: str,
    save_dir: Optional[str] = None,
    depth_model: str = "DepthAnythingv2",
    depth_scale: float = 1.0,
    device: str = "mps",
    deva_checkpoint: str = "checkpoints/DEVA-propagation.pth",
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
    use_grounding_dino: bool = False,
    grounding_dino_checkpoint: str = "checkpoints/groundingdino_swinb_cogvlm.pth",
    sam_variant: str = "sam2",
    sam2_checkpoint: str = "checkpoints/SAM 2.1 Hiera Large.pt",
    min_frame_area: int = 2000,
    min_frames: int = 3,
    min_total_pixels: int = 10000,
    min_points_3d: int = 5000,
    add_background: bool = True,
    frames_dir: Optional[str] = None,
    n_views: int = 8,
    phi_bands: Optional[List[float]] = None,
    auto_depth_scale: bool = False,
    target_scene_scale: float = 0.5,
    temporal_setting: str = "online",
    sam_pred_iou_threshold: Optional[float] = 0.88,
    sam_stability_score_threshold: Optional[float] = 0.95,
    mask_min_area: Optional[int] = 1500,
    detection_every: Optional[int] = None,
    max_num_objects: Optional[int] = None,
    use_full_scene_background: bool = True,
    equirect_min_votes: int = 1,
    equirect_kernel_size: int = 15,
    grounding_first: bool = False,
    grounding_prompts: Optional[str] = None,
    grounding_box_threshold: float = 0.25,
    grounding_text_threshold: float = 0.20,
    grounding_max_detections: Optional[int] = None,
    grounding_mask_min_area: int = 1500,
    grounding_sam_multimask: bool = True,
    grounding_box_padding: float = 0.15,
    grounding_infer_max_side: int = 1024,
    grounding_exclude_labels: Optional[str] = None,
    grounding_min_component_area_ratio: float = 0.02,
    grounding_morph_open_kernel: int = 5,
    aggregate_by_label: bool = False,
) -> Path:
    input_path = Path(input_dir)
    save_path = Path(save_dir) if save_dir else input_path

    if not grounding_first and not DEVA_AVAILABLE:
        raise RuntimeError("DEVA not available; check submodules/deva installation")

    depth = _load_depth(input_path, depth_model, force=False)
    if depth_scale and float(depth_scale) != 1.0:
        depth = depth * float(depth_scale)
    pano_rgb = _load_rgb(input_path)
    xyz, colors = _erp_pointcloud(depth, pano_rgb)

    final_depth_scale = float(depth_scale)
    auto_scaled = False
    if auto_depth_scale:
        try:
            scene_std = float(np.linalg.norm(np.std(xyz, axis=0)))
            if scene_std > 1e-9:
                scale_factor = float(target_scene_scale) / scene_std
                if abs(scale_factor - 1.0) > 1e-3:
                    xyz = xyz * float(scale_factor)
                    final_depth_scale = final_depth_scale * float(scale_factor)
                    auto_scaled = True
        except Exception:
            auto_scaled = False

    frames_path = _ensure_frames_dir(
        input_path,
        Path(frames_dir) if frames_dir else None,
        n_views=n_views,
        phi_bands=phi_bands,
    )

    # Optional final refine layer not created by default. Set to None unless explicit refine requested.
    final_refine_layer_idx = None

    grounding_tags: Dict[int, str] = {}
    grounding_scores: Dict[int, float] = {}
    grounding_detections: List[dict] = []
    grounding_erp_masks: Dict[int, np.ndarray] = {}
    grounding_erp_instance_map: Optional[np.ndarray] = None
    grounding_excluded_ids: set[int] = set()

    if grounding_first:
        print(f"[Grounding SAM] Running grounding-first detection on {input_path / 'rgb.png'}")
        prompt_sam_checkpoint = sam2_checkpoint if str(sam_variant).lower() == "sam2" else sam_checkpoint
        res = detect_objects_grounding_then_sam_on_panorama(
            input_path / "rgb.png",
            sam_checkpoint=prompt_sam_checkpoint,
            device=device,
            use_grounding=use_grounding_dino,
            grounding_checkpoint=grounding_dino_checkpoint,
            sam_variant=sam_variant,
            grounding_prompts=grounding_prompts,
            box_threshold=grounding_box_threshold,
            text_threshold=grounding_text_threshold,
            max_detections=grounding_max_detections,
            multimask_output=grounding_sam_multimask,
            min_mask_area=grounding_mask_min_area,
            box_padding_ratio=grounding_box_padding,
            grounding_infer_max_side=grounding_infer_max_side,
            min_component_area_ratio=grounding_min_component_area_ratio,
            morph_open_kernel=grounding_morph_open_kernel,
        )

        erp_instance_map = res.get("instance_map")
        erp_masks = res.get("masks", {})
        grounding_erp_masks = {
            int(inst_id): np.asarray(mask, dtype=bool)
            for inst_id, mask in erp_masks.items()
        }
        if erp_instance_map is not None:
            grounding_erp_instance_map = np.asarray(erp_instance_map, dtype=np.int32)
        grounding_tags = {int(k): str(v) for k, v in res.get("tags", {}).items()}
        grounding_scores = {int(k): float(v) for k, v in res.get("scores", {}).items()}
        grounding_detections = list(res.get("detections", []))
        label_filters = _parse_label_filter(grounding_exclude_labels)
        grounding_excluded_ids = {
            int(inst_id)
            for inst_id, label in grounding_tags.items()
            if _label_matches_filter(label, label_filters)
        }
        if grounding_excluded_ids:
            excluded_labels = sorted(
                f"{inst_id}:{grounding_tags.get(inst_id, '')}"
                for inst_id in grounding_excluded_ids
            )
            print(f"[Grounding SAM] Excluding stuff/layout labels from object layers: {', '.join(excluded_labels)}")
        if erp_instance_map is not None and int(np.asarray(erp_instance_map).max()) > 0:
            print(f"[Grounding SAM] Found {int(np.asarray(erp_instance_map).max())} panorama instances")
        # Build per-frame instance maps by projecting ERP masks to each frame
        frame_paths = sorted(frames_path.glob("rgb_*.png"), key=_numeric_suffix_key)
        if not frame_paths:
            frame_paths = sorted(frames_path.glob("*.png"), key=_numeric_suffix_key) + sorted(
                frames_path.glob("*.jpg"), key=_numeric_suffix_key
            )
        instance_maps = {}
        for frame_idx, frame_path in enumerate(frame_paths):
            rgb = np.array(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            h_f, w_f = rgb.shape[:2]
            fmap = np.zeros((h_f, w_f), dtype=np.int32)
            for inst_id, mask in erp_masks.items():
                erp_mask = mask
                proj = _project_erp_mask_to_frame(erp_mask, int(frame_idx), h_f, w_f, n=n_views, phi_bands=phi_bands)
                if proj is None:
                    continue
                fmap[proj > 0] = int(inst_id)
            instance_maps[frame_idx] = fmap
    else:
        print(f"[DEVA] Segmenting frames in {frames_path}")
        instance_maps = segment_perspective_frames(
            frames_path,
            deva_checkpoint=deva_checkpoint,
            sam_checkpoint=sam_checkpoint,
            device=device,
            temporal_setting=temporal_setting,
            min_area=min_frame_area,
            sam_pred_iou_threshold=sam_pred_iou_threshold,
            sam_stability_score_threshold=sam_stability_score_threshold,
            mask_min_area=mask_min_area,
            detection_every=detection_every,
            max_num_objects=max_num_objects,
            use_grounding_dino=use_grounding_dino,
            grounding_dino_checkpoint=grounding_dino_checkpoint,
            sam_variant=sam_variant,
            sam2_checkpoint=sam2_checkpoint,
        )

    if not instance_maps:
        raise RuntimeError("Segmentation returned no instance maps")

    if grounding_first:
        grounding_min_total_pixels = min(int(min_total_pixels), int(grounding_mask_min_area))
        raw_stats = _filter_erp_instances(
            grounding_erp_masks,
            instance_maps,
            min_total_pixels=grounding_min_total_pixels,
        )
        for inst_id in grounding_excluded_ids:
            raw_stats.pop(int(inst_id), None)
    else:
        raw_stats = _filter_instances(
            instance_maps,
            min_frames=min_frames,
            min_total_pixels=min_total_pixels,
        )
    if not raw_stats:
        raise RuntimeError(
            "No instances passed the 2D filtering thresholds. "
            "For --grounding_first, try lowering --min_total_pixels or --grounding_mask_min_area."
        )

    backend_label = "Grounding SAM" if grounding_first else "DEVA"
    print(f"[{backend_label}] Projecting instance ids to 3D")
    if grounding_first and grounding_erp_instance_map is not None:
        imap_eq = grounding_erp_instance_map.copy()
    else:
        imap_eq = _frames_to_equirect_instance_map(
            instance_maps,
            pano_rgb.shape[0],
            pano_rgb.shape[1],
            n=n_views,
            phi_bands=phi_bands,
            min_votes=equirect_min_votes,
            postprocess=True,
            kernel_size=equirect_kernel_size,
        )

    prop_labels = propagate_frame_instances_to_3d(frames_path, instance_maps, xyz)

    if imap_eq.max() > 0:
        labels_from_imap = imap_eq.reshape(-1).astype(np.int32)
        prop_labels = prop_labels.astype(np.int32)
        use_prop = (labels_from_imap == 0) & (prop_labels > 0)
        if use_prop.any():
            labels_from_imap[use_prop] = prop_labels[use_prop]
        labels_3d = labels_from_imap
    else:
        labels_3d = prop_labels.astype(np.int32)

    for inst_id, stat in list(raw_stats.items()):
        points = int((labels_3d == inst_id).sum())
        stat.points_3d = points
        if points < min_points_3d:
            raw_stats.pop(inst_id)

    if not raw_stats:
        raise RuntimeError("No instances passed the 3D point threshold")

    _ensure_dir(save_path / "traindata")
    meta_dir = save_path / "traindata" / "deva_instances"
    _ensure_dir(meta_dir)

    selected_ids = sorted(raw_stats.keys())
    instance_to_layer = {inst_id: idx for idx, inst_id in enumerate(selected_ids)}
    overlay_masks: List[Tuple[int, np.ndarray]] = []

    instance_frames: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for inst_id in selected_ids:
        frames_for_inst: List[Tuple[int, np.ndarray]] = []
        for frame_idx, fmap in instance_maps.items():
            mask = (fmap == inst_id)
            if mask.any():
                frames_for_inst.append((frame_idx, mask))
            elif grounding_first:
                frames_for_inst.append((frame_idx, np.zeros_like(fmap, dtype=bool)))
        if frames_for_inst:
            instance_frames[inst_id] = frames_for_inst

    if not instance_frames:
        raise RuntimeError(
            "All instance masks were empty across frames; check --n_views/--phi_bands "
            "or remove the cached traindata/deva_frames directory."
        )

    selected_ids = [inst_id for inst_id in selected_ids if inst_id in instance_frames]
    selected_ids_np = np.array(selected_ids, dtype=np.int32)

    labels_2d = labels_3d.reshape(pano_rgb.shape[:2])

    layer_groups: List[dict] = []
    if aggregate_by_label and grounding_first:
        grouped: Dict[str, List[int]] = {}
        for inst_id in selected_ids:
            group_label = _normalize_group_label(grounding_tags.get(int(inst_id)), f"instance_{inst_id}")
            grouped.setdefault(group_label, []).append(int(inst_id))
        for group_label in sorted(grouped.keys()):
            layer_groups.append({"group_label": group_label, "instance_ids": sorted(grouped[group_label])})
    else:
        layer_groups = [
            {"group_label": _normalize_group_label(grounding_tags.get(int(inst_id)), f"instance_{inst_id}"),
             "instance_ids": [int(inst_id)]}
            for inst_id in selected_ids
        ]

    instance_to_layer = {}
    for layer_idx, group in enumerate(layer_groups):
        for inst_id in group["instance_ids"]:
            instance_to_layer[int(inst_id)] = int(layer_idx)

    layer_map_2d, cleanup_stats = _consolidate_layer_label_map(labels_2d, layer_groups, pano_rgb)
    labels_2d = _labels_from_consolidated_layer_map(labels_2d, layer_map_2d, layer_groups)
    labels_3d = labels_2d.reshape(-1).astype(np.int32)
    for inst_id, stat in raw_stats.items():
        stat.points_3d = int((labels_3d == int(inst_id)).sum())

    cleanup_summary = ", ".join(f"{k}={v}" for k, v in cleanup_stats.items())
    print(f"[LayerData] Segmentation cleanup: {cleanup_summary}")

    print(
        f"[LayerData] Writing {len(layer_groups)} training layers "
        f"from {len(selected_ids)} instances"
    )
    frame_items = sorted(instance_maps.items(), key=lambda item: int(item[0]))
    for layer_idx, group in enumerate(layer_groups):
        group_ids = np.array(group["instance_ids"], dtype=np.int32)
        layer_dir = save_path / "traindata" / f"layer{layer_idx}"
        frames_out = layer_dir / "frames"
        _ensure_dir(frames_out)

        mask_points = np.isin(labels_3d, group_ids)
        inst_xyz = xyz[mask_points]
        inst_rgb = colors[mask_points]
        inst_labels = labels_3d[mask_points].astype(np.int32)

        _write_point_ply(layer_dir / f"pcd_rgb_layer{layer_idx}.ply", inst_xyz, inst_rgb, labels=inst_labels)
        _write_point_ply(
            layer_dir / f"pcd_mask_layer{layer_idx}.ply",
            inst_xyz,
            np.full_like(inst_rgb, 255, dtype=np.uint8),
            labels=inst_labels,
        )

        erp_mask = np.isin(labels_2d, group_ids)
        for frame_idx, fmap in frame_items:
            rgb_path = frames_path / f"rgb_{frame_idx}.png"
            pose_path = frames_path / f"transform_matrix_{frame_idx}.npy"
            if not rgb_path.exists() or not pose_path.exists():
                continue
            rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            frame_mask = _project_erp_mask_to_frame(
                erp_mask,
                int(frame_idx),
                rgb.shape[0],
                rgb.shape[1],
                n=n_views,
                phi_bands=phi_bands,
            )

            if frame_mask is None:
                frame_mask = np.isin(fmap, group_ids)
            frame_mask = _resize_mask_to_shape(frame_mask, rgb.shape[:2])
            if not _mask_has_training_content(frame_mask):
                continue
            Image.fromarray(_masked_rgb(rgb, frame_mask)).save(frames_out / f"rgb_{frame_idx}.png")
            np.save(frames_out / f"transform_matrix_{frame_idx}.npy", np.load(pose_path))

        _save_erp_outputs(layer_dir, pano_rgb, erp_mask, f"layer{layer_idx}")
        overlay_masks.append((layer_idx, erp_mask))

    selected_union_2d = np.isin(labels_2d, np.array(selected_ids, dtype=np.int32))
    selected_union_3d = np.isin(labels_3d, selected_ids_np)
    effective_full_scene_background = bool(use_full_scene_background)

    background_layer_idx = None
    if add_background:
        background_layer_idx = len(layer_groups)

        if effective_full_scene_background:
            # Full-scene background keeps all points and all valid panorama pixels.
            bg_point_mask = np.ones((labels_3d.shape[0],), dtype=bool)
            bg_erp_mask = np.ones(labels_2d.shape, dtype=bool)
            bg_labels_full = np.where(selected_union_3d, labels_3d, 0).astype(np.int32)
        else:
            # Non-full background: keep the *uncovered* points as background
            # (the previous logic used the covered points which inverted background/residual).
            bg_point_mask = ~selected_union_3d
            bg_erp_mask = ~selected_union_2d
            # Background labels are zero for background points
            bg_labels_full = np.zeros_like(labels_3d, dtype=np.int32)

        if not bg_point_mask.any():
            background_layer_idx = None
        else:
            layer_dir = save_path / "traindata" / f"layer{background_layer_idx}"
            frames_out = layer_dir / "frames"
            _ensure_dir(frames_out)

            bg_xyz = xyz[bg_point_mask]
            bg_rgb = colors[bg_point_mask]
            bg_labels = bg_labels_full[bg_point_mask]

            _write_point_ply(layer_dir / f"pcd_rgb_layer{background_layer_idx}.ply", bg_xyz, bg_rgb, labels=bg_labels)
            _write_point_ply(
                layer_dir / f"pcd_mask_layer{background_layer_idx}.ply",
                bg_xyz,
                np.full_like(bg_rgb, 255, dtype=np.uint8),
                labels=bg_labels,
            )

            for frame_idx, fmap in instance_maps.items():
                rgb_path = frames_path / f"rgb_{frame_idx}.png"
                pose_path = frames_path / f"transform_matrix_{frame_idx}.npy"
                if not rgb_path.exists() or not pose_path.exists():
                    continue
                rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
                bg_mask = _project_erp_mask_to_frame(
                    bg_erp_mask,
                    int(frame_idx),
                    rgb.shape[0],
                    rgb.shape[1],
                    n=n_views,
                    phi_bands=phi_bands,
                )

                if bg_mask is None:
                    frame_union = np.isin(fmap, np.array(selected_ids, dtype=np.int32))
                    bg_mask = np.ones_like(frame_union, dtype=bool) if effective_full_scene_background else ~frame_union
                bg_mask = _resize_mask_to_shape(bg_mask, rgb.shape[:2])
                if not _mask_has_training_content(bg_mask):
                    continue
                Image.fromarray(_masked_rgb(rgb, bg_mask)).save(frames_out / f"rgb_{frame_idx}.png")
                np.save(frames_out / f"transform_matrix_{frame_idx}.npy", np.load(pose_path))

            _save_erp_outputs(layer_dir, pano_rgb, bg_erp_mask, f"layer{background_layer_idx}")
            overlay_masks.append((background_layer_idx, bg_erp_mask))

    residual_layer_idx = None
    # If we used non-full background (uncovered points were assigned to background),
    # there is no residual left. Otherwise, residual is the complement of selected_union_3d.
    if add_background and not effective_full_scene_background:
        residual_mask_3d = np.zeros((labels_3d.shape[0],), dtype=bool)
    else:
        residual_mask_3d = ~selected_union_3d

    if residual_mask_3d.any():
        residual_layer_idx = (background_layer_idx + 1) if background_layer_idx is not None else len(layer_groups)
        layer_dir = save_path / "traindata" / f"layer{residual_layer_idx}"
        frames_out = layer_dir / "frames"
        _ensure_dir(frames_out)

        residual_xyz = xyz[residual_mask_3d]
        residual_rgb = colors[residual_mask_3d]
        residual_labels = np.zeros((residual_xyz.shape[0],), dtype=np.int32)

        _write_point_ply(layer_dir / f"pcd_rgb_layer{residual_layer_idx}.ply", residual_xyz, residual_rgb, labels=residual_labels)
        _write_point_ply(
            layer_dir / f"pcd_mask_layer{residual_layer_idx}.ply",
            residual_xyz,
            np.full_like(residual_rgb, 255, dtype=np.uint8),
            labels=residual_labels,
        )

        residual_erp_mask = ~selected_union_2d
        _save_erp_outputs(layer_dir, pano_rgb, residual_erp_mask, f"layer{residual_layer_idx}")
        overlay_masks.append((residual_layer_idx, residual_erp_mask))

        for frame_idx, fmap in instance_maps.items():
            rgb_path = frames_path / f"rgb_{frame_idx}.png"
            pose_path = frames_path / f"transform_matrix_{frame_idx}.npy"
            if not rgb_path.exists() or not pose_path.exists():
                continue
            rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            residual_frame_mask = _project_erp_mask_to_frame(
                residual_erp_mask,
                int(frame_idx),
                rgb.shape[0],
                rgb.shape[1],
                n=n_views,
                phi_bands=phi_bands,
            )

            if residual_frame_mask is None:
                residual_frame_mask = ~np.isin(fmap, selected_ids_np)
            residual_frame_mask = _resize_mask_to_shape(residual_frame_mask, rgb.shape[:2])
            if not _mask_has_training_content(residual_frame_mask):
                continue
            Image.fromarray(_masked_rgb(rgb, residual_frame_mask)).save(frames_out / f"rgb_{frame_idx}.png")
            np.save(frames_out / f"transform_matrix_{frame_idx}.npy", np.load(pose_path))

    _save_layer_panorama_overlay(
        pano_rgb,
        overlay_masks,
        save_path / "traindata" / "layer_mask_visualization.png",
    )

    coverage = float(np.mean(np.isin(labels_3d, selected_ids_np))) if labels_3d.size else 0.0

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_path),
        "frames_dir": str(frames_path),
        "depth_scale": float(depth_scale),
        "auto_depth_scaled": bool(auto_scaled),
        "final_depth_scale": float(final_depth_scale),
        "instance_count": len(selected_ids),
        "training_layer_count": len(layer_groups),
        "coverage_3d": coverage,
        "instances": [
            {
                "instance_id": int(inst_id),
                "layer_idx": int(instance_to_layer[inst_id]),
                "frame_count": int(raw_stats[inst_id].frame_count),
                "total_pixels": int(raw_stats[inst_id].total_pixels),
                "points_3d": int(raw_stats[inst_id].points_3d),
                "tag": grounding_tags.get(int(inst_id)),
                "score": grounding_scores.get(int(inst_id)),
            }
            for inst_id in selected_ids
        ],
        "layer_groups": [
            {
                "layer_idx": int(layer_idx),
                "group_label": str(group["group_label"]),
                "instance_ids": [int(inst_id) for inst_id in group["instance_ids"]],
            }
            for layer_idx, group in enumerate(layer_groups)
        ],
        "segmentation_backend": "grounding_sam" if grounding_first else "deva_sam",
        "grounding": {
            "enabled": bool(use_grounding_dino),
            "grounding_first": bool(grounding_first),
            "prompts": grounding_prompts,
            "box_threshold": float(grounding_box_threshold),
            "text_threshold": float(grounding_text_threshold),
            "max_detections": grounding_max_detections,
            "mask_min_area": int(grounding_mask_min_area),
            "box_padding": float(grounding_box_padding),
            "infer_max_side": int(grounding_infer_max_side),
            "min_component_area_ratio": float(grounding_min_component_area_ratio),
            "morph_open_kernel": int(grounding_morph_open_kernel),
            "detections": grounding_detections,
        },
        "segmentation_cleanup": cleanup_stats,
        "background_layer_idx": background_layer_idx,
        "residual_layer_idx": residual_layer_idx,
        "final_refine_layer_idx": final_refine_layer_idx,
        "filters": {
            "min_frame_area": min_frame_area,
            "min_frames": min_frames,
            "min_total_pixels": min_total_pixels,
            "min_points_3d": min_points_3d,
        },
    }

    metadata_path = save_path / "traindata" / "deva_instances.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[LayerData] Metadata written to {metadata_path}")
    print(f"[LayerData] 3D coverage: {coverage:.4f}")
    print(f"[LayerData] residual layer: {residual_layer_idx}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object-aware LayerPano training data")
    parser.add_argument("--input_dir", required=True, help="Input directory (e.g. outputs_lgs)")
    parser.add_argument("--save_dir", default=None, help="Output root (defaults to input_dir)")
    parser.add_argument("--depth_model", default="DepthAnythingv2")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--deva_checkpoint", default="checkpoints/DEVA-propagation.pth")
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--min_frame_area", type=int, default=2000)
    parser.add_argument("--min_frames", type=int, default=3)
    parser.add_argument("--min_total_pixels", type=int, default=10000)
    parser.add_argument("--min_points_3d", type=int, default=5000)
    parser.add_argument("--no_background", action="store_true")
    parser.add_argument("--frames_dir", default=None, help="Use an existing frames directory")
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--phi_bands", default="80,67.5,45,0,-45,-67.5,-80", help="Comma-separated phi bands in degrees")
    parser.add_argument("--temporal_setting", default="semionline", choices=["online", "semionline"])
    parser.add_argument("--sam_pred_iou_threshold", type=float, default=0.88)
    parser.add_argument("--sam_stability_score_threshold", type=float, default=0.95)
    parser.add_argument("--mask_min_area", type=int, default=1500)
    parser.add_argument("--detection_every", type=int, default=None)
    parser.add_argument("--max_num_objects", type=int, default=None)
    parser.add_argument("--equirect_min_votes", type=int, default=1)
    parser.add_argument("--equirect_kernel_size", type=int, default=15)
    parser.add_argument("--use_grounding_dino", action="store_true", help="Enable GroundingDINO proposals + tagging")
    parser.add_argument("--grounding_dino_checkpoint", default="checkpoints/groundingdino_swinb_cogvlm.pth")
    parser.add_argument("--grounding_first", action="store_true", help="Run GroundingDINO on panorama then SAM per-box, instead of DEVA frame segmentation")
    parser.add_argument("--grounding_prompts", default=None, help="GroundingDINO prompt string, e.g. 'person . chair . table'")
    parser.add_argument("--grounding_box_threshold", type=float, default=0.25)
    parser.add_argument("--grounding_text_threshold", type=float, default=0.20)
    parser.add_argument("--grounding_max_detections", type=int, default=None)
    parser.add_argument("--grounding_mask_min_area", type=int, default=1500)
    parser.add_argument("--grounding_single_mask", action="store_true", help="Use SAM's best single mask per GroundingDINO box")
    parser.add_argument("--grounding_box_padding", type=float, default=0.15, help="Padding ratio used to clip SAM masks around each GroundingDINO box")
    parser.add_argument("--grounding_infer_max_side", type=int, default=1024, help="Max panorama side used for GroundingDINO inference")
    parser.add_argument("--grounding_exclude_labels", default=None, help="Optional comma-separated labels to exclude from object layers")
    parser.add_argument("--grounding_min_component_area_ratio", type=float, default=0.02, help="Drop detached SAM mask components smaller than this fraction of total mask area")
    parser.add_argument("--grounding_morph_open_kernel", type=int, default=5, help="Opening kernel used to remove thin detached mask artifacts; set 0 to disable")
    parser.add_argument("--aggregate_by_label", action="store_true", help="Group same-label Grounding-SAM instances into shared training layers while preserving instance labels")
    parser.add_argument("--sam_variant", default="sam2", choices=["original", "mobile", "sam2"])
    parser.add_argument("--sam2_checkpoint", default="checkpoints/SAM 2.1 Hiera Large.pt")
    parser.add_argument("--use_full_scene_background", dest="use_full_scene_background", action="store_true")
    parser.add_argument("--no-full-scene-background", dest="use_full_scene_background", action="store_false")
    parser.set_defaults(use_full_scene_background=True)
    args = parser.parse_args()

    phi_bands = [float(x) for x in args.phi_bands.split(",") if x.strip()]

    generate_layers_from_deva(
        input_dir=args.input_dir,
        save_dir=args.save_dir,
        depth_model=args.depth_model,
        depth_scale=args.depth_scale,
        device=args.device,
        deva_checkpoint=args.deva_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        min_frame_area=args.min_frame_area,
        min_frames=args.min_frames,
        min_total_pixels=args.min_total_pixels,
        min_points_3d=args.min_points_3d,
        add_background=not args.no_background,
        frames_dir=args.frames_dir,
        n_views=args.n_views,
        phi_bands=phi_bands,
        temporal_setting=args.temporal_setting,
        sam_pred_iou_threshold=args.sam_pred_iou_threshold,
        sam_stability_score_threshold=args.sam_stability_score_threshold,
        mask_min_area=args.mask_min_area,
        detection_every=args.detection_every,
        max_num_objects=args.max_num_objects,
        use_full_scene_background=args.use_full_scene_background,
        equirect_min_votes=args.equirect_min_votes,
        equirect_kernel_size=args.equirect_kernel_size,
        use_grounding_dino=args.use_grounding_dino,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        sam_variant=args.sam_variant,
        sam2_checkpoint=args.sam2_checkpoint,
        grounding_first=args.grounding_first,
        grounding_prompts=args.grounding_prompts,
        grounding_box_threshold=args.grounding_box_threshold,
        grounding_text_threshold=args.grounding_text_threshold,
        grounding_max_detections=args.grounding_max_detections,
        grounding_mask_min_area=args.grounding_mask_min_area,
        grounding_sam_multimask=not args.grounding_single_mask,
        grounding_box_padding=args.grounding_box_padding,
        grounding_infer_max_side=args.grounding_infer_max_side,
        grounding_exclude_labels=args.grounding_exclude_labels,
        grounding_min_component_area_ratio=args.grounding_min_component_area_ratio,
        grounding_morph_open_kernel=args.grounding_morph_open_kernel,
        aggregate_by_label=args.aggregate_by_label,
    )


if __name__ == "__main__":
    main()
