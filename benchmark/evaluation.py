from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.io_utils import read_csv, write_csv
from benchmark.metrics.image_metrics import absolute_error_visualization, lpips_score, mae, psnr, ssim
from benchmark.rendering import MLXSceneRenderer
from benchmark.schemas import RECONSTRUCTION_COLUMNS, RENDERING_COLUMNS


def _numeric(path: Path) -> int:
    digits = "".join(ch for ch in path.stem.rsplit("_", 1)[-1] if ch.isdigit())
    return int(digits) if digits else -1


def evaluation_indices(scene_root: Path) -> tuple[list[int], str]:
    metadata_path = scene_root / "traindata" / "layer_instances.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split = metadata.get("benchmark_view_split") or {}
    indices = [int(value) for value in split.get("evaluation_indices", [])]
    return indices, "held_out" if indices else "unavailable_not_held_out"


@lru_cache(maxsize=8)
def _scene_evaluation_context(scene_root_value: str):
    scene_root = Path(scene_root_value)
    metadata = json.loads(
        (scene_root / "traindata" / "layer_instances.json").read_text(
            encoding="utf-8"
        )
    )
    object_union = None
    for group in metadata.get("layer_groups", []):
        if str(group.get("group_label", "")).lower() in {
            "sky",
            "background",
            "residual",
        }:
            continue
        layer_index = int(group["layer_idx"])
        mask_path = (
            scene_root
            / "traindata"
            / f"layer{layer_index}"
            / f"layer{layer_index}_erp_mask.png"
        )
        if mask_path.exists():
            with Image.open(mask_path) as handle:
                value = np.asarray(handle.convert("L")) >= 128
            object_union = value if object_union is None else (object_union | value)
    return metadata, object_union


@lru_cache(maxsize=128)
def _evaluation_frame(scene_root_value: str, index: int, max_side: int):
    scene_root = Path(scene_root_value)
    frame_dir = scene_root / "traindata" / "perspective_frames" / "frames"
    rgb_path = frame_dir / f"rgb_{index}.png"
    pose_path = frame_dir / f"transform_matrix_{index}.npy"
    if not rgb_path.exists() or not pose_path.exists():
        return None
    with Image.open(rgb_path) as handle:
        reference_image = handle.convert("RGB")
        scale = min(1.0, int(max_side) / max(reference_image.size))
        size = (
            max(1, round(reference_image.width * scale)),
            max(1, round(reference_image.height * scale)),
        )
        reference = np.asarray(
            reference_image.resize(size, Image.Resampling.LANCZOS)
        )
    metadata, object_union = _scene_evaluation_context(scene_root_value)
    foreground_mask = None
    if object_union is not None:
        try:
            from generate_layer_data import _project_erp_mask_to_frame

            grid = metadata.get("view_grid") or {}
            foreground_mask = _project_erp_mask_to_frame(
                object_union,
                index,
                size[1],
                size[0],
                n=int(grid.get("n_views", 8)),
                phi_bands=[float(x) for x in grid.get("phi_bands", [])],
            )
        except Exception:
            foreground_mask = None
    return reference, np.load(pose_path), foreground_mask


def evaluate_reconstruction(
    scene_root: str | Path, ply_path: str | Path, output_dir: str | Path,
    context: dict, variant: str, max_side: int = 512, rasterizer: str = "cpp",
) -> list[dict]:
    scene_root, output_dir = Path(scene_root), Path(output_dir)
    indices, split_status = evaluation_indices(scene_root)
    if not indices:
        return [{
            **context, "variant": variant, "view_id": None, "split": split_status,
            "status": "unavailable",
            "note": "No views were excluded from training; post-hoc training-view scores are intentionally not reported.",
        }]
    renderer = MLXSceneRenderer(ply_path, rasterizer=rasterizer)
    image_dir = output_dir / "images" / "reconstruction" / variant
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scene_cache_key = str(scene_root.resolve())
    for index in indices:
        frame = _evaluation_frame(scene_cache_key, index, int(max_side))
        if frame is None:
            rows.append({**context, "variant": variant, "view_id": index, "split": "held_out",
                         "status": "failed", "note": "Missing evaluation RGB or pose"})
            continue
        reference, pose, foreground_mask = frame
        size = (reference.shape[1], reference.shape[0])
        started = time.perf_counter()
        rendered = renderer.render(pose, size[0], size[1])
        render_seconds = time.perf_counter() - started
        error = absolute_error_visualization(reference, rendered)
        Image.fromarray(reference).save(image_dir / f"view_{index:04d}_ground_truth.png")
        Image.fromarray(rendered).save(image_dir / f"view_{index:04d}_rendered.png")
        Image.fromarray(error).save(image_dir / f"view_{index:04d}_absolute_error.png")
        rows.append({
            **context, "variant": variant, "view_id": index, "split": "held_out",
            "width": size[0], "height": size[1], "psnr_db": psnr(reference, rendered),
            "ssim": ssim(reference, rendered), "lpips": lpips_score(reference, rendered),
            "mae": mae(reference, rendered), "render_seconds": render_seconds,
            "foreground_psnr_db": (
                psnr(reference, rendered, foreground_mask)
                if foreground_mask is not None and foreground_mask.any() else None
            ),
            "foreground_ssim": (
                ssim(reference, rendered, foreground_mask)
                if foreground_mask is not None and foreground_mask.any() else None
            ),
            "background_psnr_db": (
                psnr(reference, rendered, ~foreground_mask)
                if foreground_mask is not None and (~foreground_mask).any() else None
            ),
            "background_ssim": (
                ssim(reference, rendered, ~foreground_mask)
                if foreground_mask is not None and (~foreground_mask).any() else None
            ),
            "status": "success",
            "note": "Same-source-panorama fidelity; not true novel-view geometry.",
        })
    metrics_path = output_dir / "reconstruction_metrics.csv"
    write_csv(metrics_path, read_csv(metrics_path) + rows, RECONSTRUCTION_COLUMNS)
    return rows


def benchmark_rendering(
    scene_root: str | Path, variants: dict[str, Path], output_dir: str | Path,
    context: dict, width: int, height: int, warmup: int, measured: int,
    rasterizer: str = "cpp",
) -> list[dict]:
    scene_root, output_dir = Path(scene_root), Path(output_dir)
    pose_paths = sorted(
        (scene_root / "traindata" / "perspective_frames" / "frames").glob("transform_matrix_*.npy"),
        key=_numeric,
    )
    poses = [np.load(path) for path in pose_paths]
    rows = []
    for name, ply_path in variants.items():
        try:
            renderer = MLXSceneRenderer(ply_path, rasterizer=rasterizer)
            metrics = renderer.benchmark(poses, width, height, warmup, measured)
            row = {
                **context, "variant": name, "target": Path(ply_path).name,
                "width": width, "height": height, "warmup_frames": warmup,
                "measured_frames": measured, **metrics, "status": "success",
                "ply_size_bytes": Path(ply_path).stat().st_size,
            }
        except Exception as exc:
            row = {
                **context, "variant": name, "target": Path(ply_path).name,
                "width": width, "height": height, "warmup_frames": warmup,
                "measured_frames": measured, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "ply_size_bytes": (
                    Path(ply_path).stat().st_size
                    if Path(ply_path).exists() else None
                ),
            }
        rows.append(row)
    write_csv(output_dir / "rendering_metrics.csv", rows, RENDERING_COLUMNS)
    return rows
