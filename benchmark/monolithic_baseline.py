#!/usr/bin/env python3
"""Train the optional same-input monolithic MLX baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mps_splat_backend import train_with_splat_apple


def _read_points(path: Path):
    vertex = PlyData.read(path, mmap="r")["vertex"]
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(np.float32) / 255
    labels = np.asarray(vertex["label"], dtype=np.int32) if "label" in vertex.data.dtype.names else np.zeros(len(points), np.int32)
    return points, colors, labels


def build_traindata(scene_root: Path, image_size: int) -> dict:
    clouds = [_read_points(path) for path in sorted((scene_root / "traindata").glob("layer*/pcd_rgb_layer*.ply"))]
    if not clouds:
        raise RuntimeError("No layer point clouds found")
    frame_dir = scene_root / "traindata" / "perspective_frames" / "frames"
    metadata = __import__("json").loads((scene_root / "traindata" / "layer_instances.json").read_text())
    excluded = set((metadata.get("benchmark_view_split") or {}).get("evaluation_indices", []))
    frames = []
    for rgb_path in sorted(frame_dir.glob("rgb_*.png")):
        index = int(rgb_path.stem.rsplit("_", 1)[1])
        pose_path = frame_dir / f"transform_matrix_{index}.npy"
        if index in excluded or not pose_path.exists():
            continue
        image = Image.open(rgb_path).convert("RGB")
        scale = min(1.0, image_size / max(image.size))
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        frames.append({"image": image, "transform_matrix": np.load(pose_path)})
    if not frames:
        raise RuntimeError("No monolithic training frames found")
    return {
        "fov": 90, "W": frames[0]["image"].width, "H": frames[0]["image"].height,
        "pcd_points": np.concatenate([x[0] for x in clouds]),
        "pcd_colors": np.concatenate([x[1] for x in clouds]),
        "pcd_labels": np.concatenate([x[2] for x in clouds]),
        "pcd_masks": np.ones((sum(len(x[0]) for x in clouds), 3), dtype=np.float32),
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--rasterizer", default="cpp", choices=["cpp", "python"])
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    data = build_traindata(Path(args.scene_root), args.image_size)
    train_with_splat_apple(
        data, args.output, num_iterations=args.iterations, rasterizer=args.rasterizer,
        adaptive=args.adaptive, downsample_ratio=1.0, max_points=0,
        training_profile="layer_instances",
    )


if __name__ == "__main__":
    main()
