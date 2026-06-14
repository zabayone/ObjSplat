#!/usr/bin/env python3
"""MPS-safe preprocessing bridge for the local LayerPano3D pipeline.

Normalises LayerPano3D outputs into a LabelGS-style preprocess layout.

Instance labelling pipeline (in priority order):
  1. DEVA + SAM  (--use_deva)   — video instance segmentation on perspective frames,
                                  produces temporally consistent IDs across views,
                                  writes layer{i}_instance_labels.npy (overrides SAM-only)
  2. SAM only    (--detect_objects) — automatic mask generation on the equirectangular
                                  panorama per layer, writes layer{i}_instance_labels.npy
  3. Connected-components fallback — scipy label on the visible_mask (always runs as
                                  initial pass, overwritten by SAM/DEVA if requested)

All three paths write to the same canonical path:
    <output_dir>/instances/layer{i}_instance_labels.npy
so that labelgs_instance_bridge.py and gen_traindata.py can find them
without knowing which method was used.
"""

from __future__ import annotations

import argparse
import cv2
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_closing

# Make local project imports robust when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Optional dependencies ─────────────────────────────────────────────────────
DEVA_IMPORT_ERROR = None
try:
    import utils.deva_instance_segmentation as deva_seg
    segment_perspective_frames = deva_seg.segment_perspective_frames
    propagate_frame_instances_to_3d = deva_seg.propagate_frame_instances_to_3d
    # Mirror effective availability from DEVA wrapper, not only import success.
    DEVA_AVAILABLE = bool(getattr(deva_seg, "DEVA_AVAILABLE", False))
except Exception as exc:
    DEVA_AVAILABLE = False
    DEVA_IMPORT_ERROR = str(exc)

try:
    from utils.semantic_instance_detection import detect_objects_in_layer
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class LayerSummary:
    layer_index: int
    mask_path: str
    smooth_mask_path: str
    visible_mask_path: str
    occluded_mask_path: str
    total_pixels: int
    visible_pixels: int
    occluded_pixels: int
    coverage: float
    mean_depth: Optional[float]
    median_depth: Optional[float]
    labelling_method: str = "connected_components"
    objects: list = field(default_factory=list)


# ── Utilities ─────────────────────────────────────────────────────────────────
def _device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_bool_mask(path: Path, size: tuple | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return np.array(img, dtype=np.uint8) > 0


def _save_bool_mask(path: Path, mask: np.ndarray) -> None:
    _ensure_dir(path.parent)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def _compact_instance_ids(imap: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remap positive instance IDs to a compact range [1..K]."""
    flat = np.asarray(imap, dtype=np.int64).reshape(-1)
    uniq = np.unique(flat)
    uniq = uniq[uniq > 0]
    if uniq.size == 0:
        return np.asarray(imap, dtype=np.int32), {}

    mapping = {int(old): idx + 1 for idx, old in enumerate(uniq)}
    compact = np.zeros_like(flat, dtype=np.int32)
    for old, new in mapping.items():
        compact[flat == old] = new
    return compact.reshape(np.asarray(imap).shape), mapping


def _save_canonical_3d_labels(
    instances_dir: Path,
    layer_idx_str: str,
    labels_3d: np.ndarray,
) -> Path:
    labels_3d = np.asarray(labels_3d, dtype=np.int32).reshape(-1)
    labels_3d_compact, remap = _compact_instance_ids(labels_3d)
    canonical_path = instances_dir / f"layer{layer_idx_str}_labels_3d.npy"
    np.save(canonical_path, labels_3d_compact)
    return canonical_path


def _save_label_overlay(rgb_path: Path, imap: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    import matplotlib.pyplot as plt

    if not rgb_path.exists():
        return

    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != imap.shape:
        rgb = np.array(Image.fromarray(rgb).resize((imap.shape[1], imap.shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8)

    uniq = np.unique(imap)
    uniq = uniq[uniq > 0]
    overlay = rgb.copy().astype(np.float32)
    if uniq.size > 0:
        cmap = plt.get_cmap('hsv')
        remap = {int(old): idx + 1 for idx, old in enumerate(uniq)}
        vis = np.zeros_like(imap, dtype=np.int32)
        for old, new in remap.items():
            vis[imap == old] = new
        vis_norm = vis.astype(np.float32) / float(vis.max()) if vis.max() > 0 else vis.astype(np.float32)
        colors = (cmap(vis_norm)[:, :, :3] * 255).astype(np.float32)
        mask = imap > 0
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * colors[mask]

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)





def _connected_components_map(
    mask: np.ndarray, start_id: int = 1, min_pixels: int = 512
) -> tuple[np.ndarray, int, list]:
    """Run scipy connected-components on *mask* as fallback/debug only."""
    structure = np.ones((3, 3), dtype=np.int32)
    # Clean small gaps/noise to reduce fragmentation before connected-components
    mask = binary_closing(mask.astype(bool), structure=np.ones((5, 5), dtype=bool))
    comp_ids, n_comp = scipy_label(mask.astype(bool), structure=structure)
    imap = np.zeros_like(comp_ids, dtype=np.int32)
    summary = []
    obj_id = start_id
    for c in range(1, n_comp + 1):
        cmask = comp_ids == c
        npix = int(cmask.sum())
        if npix < min_pixels:
            continue
        imap[cmask] = obj_id
        summary.append({"object_id": obj_id, "pixels": npix})
        obj_id += 1
    return imap, obj_id, summary


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if src.exists():
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return str(dst)
    return None


def _discover_layer_dirs(layering_dir: Path) -> list[tuple[int, Path]]:
    items = []
    for child in layering_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("layer"):
            continue
        suffix = child.name[5:]
        if suffix.isdigit():
            items.append((int(suffix), child))
    return sorted(items)


def _select_mask_path(layer_dir: Path, idx: int) -> Path:
    for name in [
        f"layer{idx}_mask_new.png",
        f"layer{idx}_mask_smooth_new.png",
        f"layer{idx}_mask_smooth.png",
        f"layer{idx}_mask.png",
    ]:
        p = layer_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No mask found in {layer_dir} for layer {idx}")


def _select_smooth_path(layer_dir: Path, idx: int) -> Optional[Path]:
    for name in [
        f"layer{idx}_mask_smooth_new.png",
        f"layer{idx}_mask_smooth.png",
    ]:
        p = layer_dir / name
        if p.exists():
            return p
    return None


def _load_depth(
    input_dir: Path, layering_dir: Path, depth_model: str, force_rebuild: bool
) -> np.ndarray:
    if not force_rebuild:
        for candidate in [
            layering_dir / "depth.npy",
            input_dir / "depth.npy",
            input_dir / "layering" / "depth.npy",
        ]:
            if candidate.exists():
                return np.load(candidate)

    rgb_path = input_dir / "rgb.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB panorama at {rgb_path}")

    from utils.depth_alignment import Pano_depth_estimation
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    device = _device_name()
    estimator = Pano_depth_estimation(
        rgb.shape[0], rgb.shape[1], str(input_dir), device, depth_model=depth_model
    )
    depth = np.asarray(estimator.get_panodepth(rgb))
    _ensure_dir(layering_dir)
    np.save(layering_dir / "depth.npy", depth)
    return depth


# ── Step 1: Base layout + connected-components instances ──────────────────────
def build_preprocess_layout(
    input_dir: Path,
    output_dir: Path,
    depth_model: str,
    force_rebuild_depth: bool,
) -> tuple[dict, list[LayerSummary]]:
    layering_dir = input_dir / "layering"
    if not layering_dir.exists():
        raise FileNotFoundError(f"Missing layering directory: {layering_dir}")

    _ensure_dir(output_dir)
    depth = np.asarray(_load_depth(input_dir, layering_dir, depth_model, force_rebuild_depth))

    for sub in ["depth", "masks", "unoccluded_masks", "occlusion", "instances", "reference"]:
        _ensure_dir(output_dir / sub)

    np.save(output_dir / "depth" / "depth.npy", depth)
    if depth.ndim == 2:
        dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
        norm = ((depth - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(norm).save(output_dir / "depth" / "depth.png")

    _copy_if_exists(input_dir / "rgb.png", output_dir / "reference" / "rgb.png")
    _copy_if_exists(
        layering_dir / "layer_mask_visualization.png",
        output_dir / "layer_mask_visualization.png",
    )

    layer_summaries: list[LayerSummary] = []
    cumulative_mask = np.zeros(depth.shape[:2], dtype=bool)
    layer_dirs = _discover_layer_dirs(layering_dir)
    next_obj_id = 1

    if not layer_dirs:
        raise FileNotFoundError(f"No layer folders found under {layering_dir}")

    for layer_idx, layer_dir in layer_dirs:
        mask_path = _select_mask_path(layer_dir, layer_idx)
        smooth_path = _select_smooth_path(layer_dir, layer_idx)

        sharp_mask = _load_bool_mask(mask_path, size=(depth.shape[1], depth.shape[0]))
        visible_mask = sharp_mask & ~cumulative_mask
        occluded_mask = sharp_mask & cumulative_mask
        cumulative_mask |= sharp_mask

        # ── Copy masks ──
        mtdir = output_dir / "masks" / f"layer{layer_idx}"
        vtdir = output_dir / "unoccluded_masks" / f"layer{layer_idx}"
        _ensure_dir(mtdir); _ensure_dir(vtdir)

        sharp_tgt = mtdir / mask_path.name
        smooth_tgt = mtdir / (smooth_path.name if smooth_path else f"layer{layer_idx}_mask_smooth.png")
        vis_tgt = vtdir / f"layer{layer_idx}_visible_mask.png"
        occ_tgt = vtdir / f"layer{layer_idx}_occluded_mask.png"

        shutil.copy2(mask_path, sharp_tgt)
        if smooth_path:
            shutil.copy2(smooth_path, smooth_tgt)
        else:
            _save_bool_mask(smooth_tgt, visible_mask)
        _save_bool_mask(vis_tgt, visible_mask)
        _save_bool_mask(occ_tgt, occluded_mask)

        depth_vals = depth[sharp_mask]
        mean_d = float(np.nanmean(depth_vals)) if depth_vals.size else None
        med_d = float(np.nanmedian(depth_vals)) if depth_vals.size else None

        # ── Connected-components (always done as base / fallback) ──
        imap, next_obj_id, obj_summary = _connected_components_map(
            visible_mask, start_id=next_obj_id
        )
        print(f"  connected-components fallback -> layer{layer_idx} ({int(visible_mask.sum())} visible px)")
        inst_path = output_dir / "instances" / f"layer{layer_idx}_instance_labels.npy"
        np.save(inst_path, np.round(imap).astype(np.int32))

        layer_summaries.append(
            LayerSummary(
                layer_index=layer_idx,
                mask_path=str(sharp_tgt),
                smooth_mask_path=str(smooth_tgt),
                visible_mask_path=str(vis_tgt),
                occluded_mask_path=str(occ_tgt),
                total_pixels=int(sharp_mask.size),
                visible_pixels=int(visible_mask.sum()),
                occluded_pixels=int(occluded_mask.sum()),
                coverage=float(sharp_mask.mean()),
                mean_depth=mean_d,
                median_depth=med_d,
                labelling_method="connected_components_fallback",
                objects=obj_summary,
            )
        )

    # ── Occlusion map JSON ──
    layer_numbers = [s.layer_index for s in layer_summaries]
    occlusion_map = {
        "layout": "numeric-layer-order-front-to-back",
        "front_to_back_layers": layer_numbers,
        "layer_relations": {
            f"layer{s.layer_index}": {
                "occludes": [f"layer{i}" for i in layer_numbers if i > s.layer_index],
                "occluded_by": [f"layer{i}" for i in layer_numbers if i < s.layer_index],
            }
            for s in layer_summaries
        },
    }
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "depth_model": depth_model,
        "depth_shape": list(depth.shape),
        "layer_count": len(layer_summaries),
        "layers": [asdict(s) for s in layer_summaries],
        "occlusion_map": occlusion_map,
    }
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(output_dir / "occlusion" / "occlusion_map.json", "w") as fh:
        json.dump(occlusion_map, fh, indent=2)

    return summary, layer_summaries


# ── Step 2 (optional): SAM per-layer instance detection ──────────────────────
def refine_instances_with_deva(
    input_dir: Path,
    output_dir: Path,
    layer_summaries: list[LayerSummary],
    deva_checkpoint: str = "checkpoints/DEVA-propagation.pth",
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
    device: str = "mps",
) -> None:
    """Refine instance labels using DEVA on perspective frames.

    For each layer:
      1. Runs DEVA+SAM on traindata/layer{i}/frames/rgb_*.png
      2. Projects 3D point cloud (pcd_rgb_layer{i}.ply) into each frame
         using the saved 4x4 c2w transform matrices
      3. Majority-votes instance IDs per 3D point
      4. Re-projects votes back to equirectangular to build instance_map
      5. Writes layer{i}_instance_labels.npy (DEVA always overrides previous maps)
    """
    if not DEVA_AVAILABLE:
        print("⚠ DEVA not available — skipping DEVA refinement")
        if DEVA_IMPORT_ERROR:
            print(f"  Import error: {DEVA_IMPORT_ERROR}")
        return

    instances_dir = output_dir / "instances"
    _ensure_dir(instances_dir)

    layer_idx = 3
    layer_idx_str = "3"
    frames_dir, pcd_path = _resolve_layer3_deva_inputs(input_dir, output_dir)
    frame_files = list(frames_dir.glob("rgb_*.png")) if frames_dir is not None else []
    if frames_dir is None or pcd_path is None or not frame_files:
        print("  ⚠ Unable to resolve layer3 DEVA inputs (frames/PLY) — skipping DEVA")
        print("    Expected one of: traindata/layer3 or fallback layering/pcd_rgb.ply + rgb.png")
        return

    if "traindata" in str(frames_dir):
        print(f"\n  DEVA → layer3 ({len(frame_files)} frames from traindata)")
    else:
        print(f"\n  DEVA → layer3 ({len(frame_files)} generated fallback frames)")

    # ── DEVA inference ──
    instance_maps = segment_perspective_frames(
        frames_dir,
        deva_checkpoint=deva_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device=device,
        temporal_setting="online",
    )
    if not instance_maps:
        print(f"    ⚠ DEVA returned empty maps for layer{layer_idx}")
        return

    # ── Load point cloud for this layer ──
    if not pcd_path.exists():
        print(f"    ⚠ Point cloud not found: {pcd_path}")
        return

    try:
        xyz = _load_ply_xyz(pcd_path)
    except Exception as exc:
        print(f"    ⚠ Failed to read PLY: {exc}")
        return

    # ── 3D vote ──
    labels_3d = propagate_frame_instances_to_3d(frames_dir, instance_maps, xyz)
    canonical_3d_path = _save_canonical_3d_labels(instances_dir, layer_idx_str, labels_3d)
    print(
        f"    ✓ 3D labels saved to {canonical_3d_path.name} "
        f"({int((labels_3d > 0).sum())}/{len(labels_3d)} labelled)"
    )

    # ── Project labels to equirectangular and save (DEVA is authoritative) ──
    # Primary path: inverse project DEVA frame masks back to ERP using the same
    # camera convention as frame generation.
    imap_eq = frames_to_equirect_instance_map(instance_maps, output_dir)
    # Fallback: project compact 3D labels if frame inversion fails.
    if imap_eq is None or imap_eq.max() == 0:
        imap_eq = labels_3d_to_equirect(xyz, labels_3d, output_dir)

    if imap_eq is not None and imap_eq.max() > 0:
        imap_eq_compact, _ = _compact_instance_ids(imap_eq)
        overlay_rgb_path = input_dir / "rgb.png"
        inst_path = instances_dir / f"layer{layer_idx_str}_instance_labels.npy"

        # Save canonical projected labels and overwrite instance_labels.npy unconditionally
        np.save(inst_path, np.round(imap_eq_compact).astype(np.int32))

        # Save a single equirectangular debug overlay.
        _save_label_overlay(
            overlay_rgb_path,
            imap_eq_compact,
            instances_dir / f"layer{layer_idx_str}_labels_3d_overlay.png",
        )

        print(f"    ✓ 2D projected labels saved to {inst_path.name} derived from DEVA 3D labels")

        # Update summary metadata
        for s in layer_summaries:
            if s.layer_index == layer_idx:
                s.labelling_method = "3d_deva"
                n = int(np.unique(imap_eq_compact[imap_eq_compact > 0]).size)
                s.objects = [
                    {"object_id": int(o), "pixels": int((imap_eq_compact == o).sum())}
                    for o in np.unique(imap_eq_compact)
                    if o > 0
                ]
                print(
                    f"    ✓ Equirect map: {n} objects written to "
                    f"layer{layer_idx_str}_instance_labels.npy"
                )
                break
    else:
        print("    ⚠ Equirect re-projection empty — keeping previous labels")


def _generate_layer3_frames_from_equirect(input_dir: Path, output_dir: Path, n: int = 8) -> Optional[Path]:
    """Generate layer3 perspective frames directly from equirect rgb for DEVA.

    This mirrors gen_traindata's frame sampling convention (24 views at
    phi={45,0,-45}, theta={0..360}). It breaks the preprocess<->traindata
    dependency cycle for DEVA on layer3.
    """
    pano_path = input_dir / "rgb.png"
    if not pano_path.exists():
        return None

    try:
        import utils.pano_utils.Equirec2Perspec as E2P
        from utils.trajectory import gcd_pose_gs
    except Exception as exc:
        print(f"  ⚠ Cannot import panorama projection utilities: {exc}")
        return None

    try:
        pano = np.array(Image.open(pano_path).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        print(f"  ⚠ Cannot read {pano_path.name}: {exc}")
        return None

    pano_h = int(pano.shape[0])
    pers_size = int((pano_h / 1024.0) * 512)
    pers_size = max(128, pers_size)

    frames_dir = output_dir / "deva_frames" / "layer3" / "frames"
    _ensure_dir(frames_dir)

    theta = [(360.0 / n) * i for i in range(n)]
    theta = theta + theta + theta
    phi = [45.0] * n + [0.0] * n + [-45.0] * n

    equ = E2P.Equirectangular(pano)
    for i, (th, ph) in enumerate(zip(theta, phi)):
        pers_img = equ.GetPerspective(90, th, ph, pers_size, pers_size)
        pers_img = np.clip(pers_img, 0, 255).astype(np.uint8)
        Image.fromarray(pers_img).save(frames_dir / f"rgb_{i}.png")
        np.save(frames_dir / f"transform_matrix_{i}.npy", gcd_pose_gs(th, ph))

    return frames_dir


def _resolve_layer3_deva_inputs(input_dir: Path, output_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve frames+PLY for DEVA on layer3 without requiring traindata first."""
    td_layer3 = input_dir / "traindata" / "layer3"
    td_frames = td_layer3 / "frames"
    td_pcd = td_layer3 / "pcd_rgb_layer3.ply"
    if td_frames.exists() and list(td_frames.glob("rgb_*.png")) and td_pcd.exists():
        return td_frames, td_pcd

    # Fallback path that avoids circular dependency with gen_traindata.
    layering_pcd = input_dir / "layering" / "pcd_rgb.ply"
    if not layering_pcd.exists():
        return None, None

    gen_frames = _generate_layer3_frames_from_equirect(input_dir, output_dir)
    if gen_frames is None or not list(gen_frames.glob("rgb_*.png")):
        return None, None
    return gen_frames, layering_pcd


def _frame_idx_to_theta_phi(frame_idx: int, n: int = 8) -> Optional[tuple[float, float]]:
    """Frame convention used by gen_traindata.gen_frames_data."""
    if frame_idx < 0:
        return None
    band = frame_idx // n
    pos = frame_idx % n
    if band == 0:
        phi = 45.0
    elif band == 1:
        phi = 0.0
    elif band == 2:
        phi = -45.0
    else:
        return None
    theta = (360.0 / float(n)) * float(pos)
    return theta, phi


def _project_frame_labels_to_equirect(
    frame_labels: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    out_h: int,
    out_w: int,
    fov_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project perspective labels to ERP with Perspec2Equirec geometry + NN sampling."""
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
    R1, _ = cv2.Rodrigues(z_axis * np.radians(theta_deg))
    R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi_deg))
    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([out_h * out_w, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T.reshape([out_h, out_w, 3])

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


def frames_to_equirect_instance_map(instance_maps: dict, output_dir: Path, n: int = 8) -> Optional[np.ndarray]:
    """Rebuild ERP labels from DEVA frame masks with inverse projection + per-pixel voting."""
    depthpath = output_dir / "depth" / "depth.npy"
    if not depthpath.exists():
        return None
    depth = np.load(depthpath)
    H, W = depth.shape

    counts_by_label: dict[int, np.ndarray] = {}
    for fidx, fmap in instance_maps.items():
        ang = _frame_idx_to_theta_phi(int(fidx), n=n)
        if ang is None:
            continue
        theta_deg, phi_deg = ang
        proj, valid = _project_frame_labels_to_equirect(
            np.asarray(fmap, dtype=np.int32), theta_deg, phi_deg, H, W, fov_deg=90.0
        )
        labels = np.unique(proj[valid])
        labels = labels[labels > 0]
        for lab in labels:
            mask = (proj == int(lab)) & valid
            if not mask.any():
                continue
            arr = counts_by_label.get(int(lab))
            if arr is None:
                arr = np.zeros((H, W), dtype=np.uint16)
                counts_by_label[int(lab)] = arr
            arr[mask] += 1

    if not counts_by_label:
        return np.zeros((H, W), dtype=np.int32)

    best_label = np.zeros((H, W), dtype=np.int32)
    best_count = np.zeros((H, W), dtype=np.uint16)
    for lab, cnt in counts_by_label.items():
        g = cnt > best_count
        if g.any():
            best_label[g] = int(lab)
            best_count[g] = cnt[g]
    return best_label


def _load_ply_xyz(pcd_path: Path) -> np.ndarray:
    """Load xyz from a PLY using plyfile (always available in LayerPano env)."""
    from plyfile import PlyData
    ply = PlyData.read(str(pcd_path))
    v = ply["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)


def labels_3d_to_equirect(
    xyz: np.ndarray,
    labels: np.ndarray,
    outputdir: Path,
) -> Optional[np.ndarray]:
    depthpath = outputdir / "depth" / "depth.npy"
    if not depthpath.exists():
        print(f"depth.npy not found at {depthpath} cannot back-project labels to equirect")
        return None

    depth = np.load(depthpath)
    H, W = depth.shape

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return None
    if xyz.shape[0] != labels.shape[0]:
        print(f"xyz/labels size mismatch: {xyz.shape[0]} vs {labels.shape[0]}")
        return None

    # Fast path for LayerPano point clouds generated from ERP depthmap raster:
    # point order is pixel order, so reshaping is exact and avoids reprojection drift.
    if labels.shape[0] == H * W:
        return labels.reshape(H, W).astype(np.int32)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    norm = np.sqrt(x * x + y * y + z * z)
    norm = np.where(norm < 1e-8, 1e-8, norm)

    # Match training-time bridge convention for world-space points:
    # world axes are treated as x=right, y=up, z=forward.
    # This keeps labels consistent between overlays and frame projections.
    lon = np.arctan2(x, z)                         # [-pi, pi]
    lat = np.arcsin(np.clip(y / norm, -1.0, 1.0)) # [-pi/2, pi/2]

    u = ((lon / (2 * np.pi) + 0.5) * W).astype(np.int32)
    v = ((0.5 - lat / np.pi) * H).astype(np.int32)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    
    valid = labels > 0
    if not valid.any():
        return np.zeros((H, W), dtype=np.int32)

    flatidx = (v[valid].astype(np.int64) * int(W) + u[valid].astype(np.int64))
    lbls = labels[valid].astype(np.int64)
    npixels = int(H) * int(W)

    bestlabel = np.zeros(npixels, dtype=np.int32)
    bestcount = np.zeros(npixels, dtype=np.int32)

    for lab in np.unique(lbls):
        mask = lbls == lab
        if not mask.any():
            continue
        counts = np.bincount(flatidx[mask], minlength=npixels)
        greater = counts > bestcount
        if greater.any():
            bestlabel[greater] = int(lab)
            bestcount[greater] = counts[greater]

    return bestlabel.reshape(H, W)

# ── Finalise summary JSON after optional refinement steps ────────────────────
def _write_summary(output_dir: Path, summary: dict, layer_summaries: list[LayerSummary]) -> None:
    summary["layers"] = [asdict(s) for s in layer_summaries]
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bridge LayerPano3D outputs into a LabelGS-style preprocess layout"
    )
    parser.add_argument("--input_dir", default="outputs", help="Root pipeline output directory")
    parser.add_argument("--output_dir", default=None, help="Target preprocess directory (default: <input>/preprocess/labelgs)")
    parser.add_argument("--depth_model", default="DepthAnythingv2", help="Fallback depth model")
    parser.add_argument("--rebuild_depth", action="store_true", help="Force depth recomputation")
    parser.add_argument("--detect_objects", action="store_true", help="Refine instances with SAM (automatic masking)")
    parser.add_argument("--use_deva", action="store_true", help="Refine instances with DEVA+SAM on perspective frames (best quality)")
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--deva_checkpoint", default="checkpoints/DEVA-propagation.pth")
    parser.add_argument("--device", default="mps", help="mps | cuda | cpu")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir / "preprocess" / "labelgs"
    )

    # Step 1 — always
    print("\n── Step 1: building base preprocess layout (connected-components) ──")
    summary, layer_summaries = build_preprocess_layout(
        input_dir, output_dir, args.depth_model, args.rebuild_depth
    )
    print(f"  ✓ {summary['layer_count']} layers processed")

    # Step 2 — SAM (optional, superseded by DEVA if both flags set)
    if args.detect_objects and not args.use_deva:
        print("\n── Step 2: refining instances with SAM ──")
        refine_instances_with_sam(
            input_dir, output_dir, layer_summaries,
            sam_checkpoint=args.sam_checkpoint, device=args.device
        )

    # Step 3 — DEVA (optional, highest quality)
    if args.use_deva:
        print("\n── Step 3: refining instances with DEVA+SAM ──")
        if DEVA_AVAILABLE:
            refine_instances_with_deva(
                input_dir, output_dir, layer_summaries,
                deva_checkpoint=args.deva_checkpoint,
                sam_checkpoint=args.sam_checkpoint,
                device=args.device,
            )
        else:
            print("⚠ DEVA non disponibile: fallback automatico a SAM")
            if DEVA_IMPORT_ERROR:
                print(f"  Motivo import DEVA: {DEVA_IMPORT_ERROR}")
            refine_instances_with_sam(
                input_dir, output_dir, layer_summaries,
                sam_checkpoint=args.sam_checkpoint, device=args.device
            )

    # Persist updated labelling_method in summary.json
    _write_summary(output_dir, summary, layer_summaries)

    methods = {s.layer_index: s.labelling_method for s in layer_summaries}
    print("\n── Done ──")
    print(json.dumps({"output_dir": str(output_dir), "layer_count": summary["layer_count"], "labelling_methods": methods}, indent=2))


if __name__ == "__main__":
    main()
