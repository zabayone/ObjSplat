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


def build_traindata(
    scene_root: Path, image_size: int, max_points: int = 0, seed: int = 42
) -> dict:
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
    points = np.concatenate([x[0] for x in clouds])
    colors = np.concatenate([x[1] for x in clouds])
    labels = np.concatenate([x[2] for x in clouds])
    if int(max_points or 0) > 0 and len(points) > int(max_points):
        rng = np.random.default_rng(int(seed))
        selected = np.sort(
            rng.choice(len(points), size=int(max_points), replace=False)
        )
        points, colors, labels = points[selected], colors[selected], labels[selected]
    return {
        "fov": 90, "W": frames[0]["image"].width, "H": frames[0]["image"].height,
        "pcd_points": points,
        "pcd_colors": colors,
        "pcd_labels": labels,
        "pcd_masks": np.ones((len(points), 3), dtype=np.float32),
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
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = build_traindata(
        Path(args.scene_root), args.image_size, args.max_points, args.seed
    )
    train_with_splat_apple(
        data, args.output, num_iterations=args.iterations, rasterizer=args.rasterizer,
        adaptive=args.adaptive, downsample_ratio=1.0, max_points=args.max_points,
        training_profile="layer_instances",
    )


if __name__ == "__main__":
    main()
