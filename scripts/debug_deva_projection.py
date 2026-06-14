#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


def save_overlay(rgb_path: Path, labels: np.ndarray, out_path: Path, alpha: float = 0.45) -> None:
    import matplotlib.pyplot as plt

    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != labels.shape:
        rgb = np.array(
            Image.fromarray(rgb).resize((labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    overlay = rgb.astype(np.float32).copy()
    if uniq.size > 0:
        remap = {int(old): idx + 1 for idx, old in enumerate(uniq)}
        vis = np.zeros_like(labels, dtype=np.int32)
        for old, new in remap.items():
            vis[labels == old] = new
        vis = vis.astype(np.float32) / float(max(vis.max(), 1))
        cmap = plt.get_cmap("hsv")
        colors = (cmap(vis)[:, :, :3] * 255).astype(np.float32)
        mask = labels > 0
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * colors[mask]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def project_points_to_frame(xyz: np.ndarray, c2w: np.ndarray, width: int, height: int):
    import numpy.linalg as LA

    w2c = LA.inv(c2w)
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)])
    pts_gs = (w2c @ xyz_h.T).T[:, :3]

    # Match utils.deva_instance_segmentation.propagate_frame_instances_to_3d
    pts_cam = np.stack([pts_gs[:, 2], pts_gs[:, 0], -pts_gs[:, 1]], axis=1)
    forward = pts_cam[:, 0]
    valid = forward > 1e-4

    focal = float(width) / 2.0  # FOV=90
    x_ndc = pts_cam[:, 1] / np.where(valid, forward, 1.0)
    y_ndc = pts_cam[:, 2] / np.where(valid, forward, 1.0)

    u = (x_ndc * focal + float(width) / 2.0).astype(np.int32)
    v = (-y_ndc * focal + float(height) / 2.0).astype(np.int32)

    inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, inside


def majority_splat_frame(height: int, width: int, u: np.ndarray, v: np.ndarray, labels: np.ndarray, inside: np.ndarray) -> np.ndarray:
    idx = np.nonzero(inside & (labels > 0))[0]
    out = np.zeros((height, width), dtype=np.int32)
    if idx.size == 0:
        return out

    flat = (v[idx].astype(np.int64) * int(width) + u[idx].astype(np.int64))
    labs = labels[idx].astype(np.int64)
    npix = int(height) * int(width)

    best = np.zeros(npix, dtype=np.int32)
    best_count = np.zeros(npix, dtype=np.int32)

    for lab in np.unique(labs):
        m = labs == lab
        c = np.bincount(flat[m], minlength=npix)
        g = c > best_count
        if g.any():
            best[g] = int(lab)
            best_count[g] = c[g]

    return best.reshape(height, width)


def load_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)


def remap_ids(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.int32)
    if not mapping:
        return out
    for old, new in mapping.items():
        out[arr == int(old)] = int(new)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Debug DEVA frame/3D projection consistency")
    p.add_argument("--root", default="outputs_lgs", help="Pipeline output root")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--max_frames", type=int, default=24)
    args = p.parse_args()

    root = Path(args.root)
    layer = int(args.layer)

    instances = root / "preprocess" / "labelgs" / "instances"
    frames_dir = root / "traindata" / f"layer{layer}" / "frames"
    pcd_path = root / "traindata" / f"layer{layer}" / f"pcd_rgb_layer{layer}.ply"

    labels3d_path = instances / f"layer{layer}_labels_3d.npy"
    remap_path = instances / f"layer{layer}_label_remap.json"
    if not labels3d_path.exists() or not pcd_path.exists() or not frames_dir.exists():
        raise FileNotFoundError("Missing required inputs (labels3d/pcd/frames)")

    labels3d = np.load(labels3d_path).astype(np.int32).reshape(-1)
    xyz = load_xyz(pcd_path)
    if xyz.shape[0] != labels3d.shape[0]:
        raise ValueError(f"xyz/labels mismatch: {xyz.shape[0]} vs {labels3d.shape[0]}")

    mapping: dict[int, int] = {}
    if remap_path.exists():
        raw = json.loads(remap_path.read_text())
        mapping = {int(k): int(v) for k, v in raw.get("old_to_new", {}).items()}

    out_dir = root / "preprocess" / "labelgs" / f"debug_projection_layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = sorted(
        int(p.stem.split("_")[1])
        for p in frames_dir.glob("rgb_*.png")
        if p.stem.split("_")[-1].isdigit()
    )[: args.max_frames]

    rows = []
    for fid in frame_ids:
        rgb_path = frames_dir / f"rgb_{fid}.png"
        pose_path = frames_dir / f"transform_matrix_{fid}.npy"
        fmap_path = instances / f"layer{layer}_frame{fid}_instance_labels_deva.npy"

        if not rgb_path.exists() or not pose_path.exists() or not fmap_path.exists():
            continue

        fmap_raw = np.load(fmap_path).astype(np.int32)
        fmap_comp = remap_ids(fmap_raw, mapping)
        h, w = fmap_raw.shape

        c2w = np.load(pose_path).astype(np.float64)
        u, v, inside = project_points_to_frame(xyz, c2w, w, h)
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            continue

        pred = labels3d[idx]
        deva_raw_s = fmap_raw[v[idx], u[idx]]
        deva_comp_s = fmap_comp[v[idx], u[idx]]
        on_obj = deva_raw_s > 0

        acc_raw = float((pred[on_obj] == deva_raw_s[on_obj]).mean()) if on_obj.any() else 0.0
        acc_comp = float((pred[on_obj] == deva_comp_s[on_obj]).mean()) if on_obj.any() else 0.0
        coverage = float(on_obj.mean())

        pred_map = majority_splat_frame(h, w, u, v, labels3d, inside)

        save_overlay(rgb_path, fmap_comp, out_dir / f"frame{fid:02d}_deva_comp_overlay.png")
        save_overlay(rgb_path, pred_map, out_dir / f"frame{fid:02d}_pred3d_overlay.png")
        mismatch = (fmap_comp > 0) & (pred_map > 0) & (fmap_comp != pred_map)
        mismatch_img = np.zeros((h, w), dtype=np.uint8)
        mismatch_img[mismatch] = 255
        Image.fromarray(mismatch_img).save(out_dir / f"frame{fid:02d}_mismatch_mask.png")

        rows.append(
            {
                "frame": int(fid),
                "inside_points": int(idx.size),
                "deva_positive_ratio": coverage,
                "acc_raw_ids": acc_raw,
                "acc_compact_ids": acc_comp,
                "deva_compact_unique": int(np.unique(fmap_comp[fmap_comp > 0]).size),
                "pred3d_unique": int(np.unique(pred_map[pred_map > 0]).size),
            }
        )

    report = {
        "root": str(root),
        "layer": layer,
        "num_frames": len(rows),
        "id_remap_size": len(mapping),
        "mean_acc_raw_ids": float(np.mean([r["acc_raw_ids"] for r in rows])) if rows else 0.0,
        "mean_acc_compact_ids": float(np.mean([r["acc_compact_ids"] for r in rows])) if rows else 0.0,
        "min_acc_compact_ids": float(np.min([r["acc_compact_ids"] for r in rows])) if rows else 0.0,
        "max_acc_compact_ids": float(np.max([r["acc_compact_ids"] for r in rows])) if rows else 0.0,
        "frames": rows,
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved visual debug artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
