from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.io_utils import read_csv, write_csv
from benchmark.metrics.mood import compare_topology
from benchmark.metrics.segmentation import mask_metrics, seam_crossing
from benchmark.ply_utils import inspect_ply
from benchmark.schemas import LAYER_COLUMNS, MOOD_COLUMNS, SEGMENTATION_COLUMNS


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_indices(scene_root: Path, metadata: dict) -> list[int]:
    result = set()
    for group in metadata.get("layer_groups", []):
        result.add(int(group["layer_idx"]))
    for key in ("background_layer_idx", "residual_layer_idx"):
        if metadata.get(key) is not None:
            result.add(int(metadata[key]))
    for path in (scene_root / "traindata").glob("layer*"):
        match = re.fullmatch(r"layer(\d+)", path.name)
        if match:
            result.add(int(match.group(1)))
    return sorted(result)


def _connected_components(mask: np.ndarray) -> int | None:
    try:
        import cv2
        count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return max(0, int(count - 1))
    except ImportError:
        return None


def _count_ply(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    info = inspect_ply(path)
    return info["vertex_count"], info["size_bytes"]


def analyse_layers(scene_root: str | Path, context: dict) -> tuple[list[dict], dict]:
    scene_root = Path(scene_root)
    metadata_path = scene_root / "traindata" / "layer_instances.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    metadata = _load_json(metadata_path)
    groups = {int(g["layer_idx"]): g for g in metadata.get("layer_groups", [])}
    instances_by_layer: dict[int, list[dict]] = {}
    for instance in metadata.get("instances", []):
        instances_by_layer.setdefault(int(instance["layer_idx"]), []).append(instance)
    merged_candidates = [
        scene_root / "scene" / "gsplat_scene_merged_refined.ply",
        scene_root / "scene" / "gsplat_scene_merged.ply",
    ]
    merged = next((path for path in merged_candidates if path.exists()), None)
    merged_count = inspect_ply(merged)["vertex_count"] if merged else None
    rows = []
    for index in _layer_indices(scene_root, metadata):
        layer_dir = scene_root / "traindata" / f"layer{index}"
        mask_path = layer_dir / f"layer{index}_erp_mask.png"
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128 if mask_path.exists() else None
        frame_masks = sorted((layer_dir / "frames").glob("mask_*.png"))
        supervised = []
        for path in frame_masks:
            with Image.open(path) as image:
                supervised.append(int((np.asarray(image.convert("L")) >= 128).sum()))
        initial_count, _ = _count_ply(layer_dir / f"pcd_rgb_layer{index}.ply")
        final_count, ply_size = _count_ply(scene_root / "scene" / f"gsplat_layer{index}.ply")
        members = instances_by_layer.get(index, [])
        scores = [float(item["score"]) for item in members if item.get("score") is not None]
        group = groups.get(index, {})
        label = group.get("group_label")
        if label is None:
            if index == metadata.get("background_layer_idx"):
                label = "background"
            elif index == metadata.get("residual_layer_idx"):
                label = "residual"
            else:
                label = "unknown"
        rows.append({
            **context, "variant": "layered", "layer_index": index,
            "semantic_label": label,
            "instance_ids": json.dumps(group.get("instance_ids", [x.get("instance_id") for x in members])),
            "confidence_count": len(scores), "confidence_mean": np.mean(scores) if scores else None,
            "confidence_min": min(scores) if scores else None,
            "confidence_max": max(scores) if scores else None,
            "mask_area_pixels": int(mask.sum()) if mask is not None else None,
            "mask_coverage_percent": float(mask.mean() * 100) if mask is not None else None,
            "connected_components": _connected_components(mask) if mask is not None else None,
            "projected_3d_points": sum(int(x.get("points_3d") or 0) for x in members) or initial_count,
            "training_frames": len(frame_masks),
            "total_supervised_pixels": sum(supervised),
            "mean_supervised_pixels_per_frame": np.mean(supervised) if supervised else None,
            "training_iterations": None, "training_time_seconds": None,
            "initial_gaussians": initial_count, "final_gaussians": final_count,
            "ply_size_bytes": ply_size,
            "percent_final_scene_gaussians": (
                final_count / merged_count * 100 if final_count is not None and merged_count else None
            ),
            "status": "success" if final_count is not None else ("skipped" if initial_count is not None else "failed"),
            "reason": None if final_count is not None else "No trained layer PLY found",
        })
    values = [row["final_gaussians"] for row in rows if row["final_gaussians"] is not None]
    summary = {
        "layer_count": len(rows), "trained_layer_count": len(values),
        "final_layer_gaussians": {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "largest_to_smallest_ratio": max(values) / min(values) if values and min(values) else None,
        },
        "correlations": correlation_summary(rows),
    }
    return rows, summary


def correlation_summary(rows: list[dict]) -> dict:
    pairs = [
        ("mask_area_pixels", "projected_3d_points"),
        ("mask_area_pixels", "final_gaussians"),
        ("projected_3d_points", "final_gaussians"),
        ("training_time_seconds", "final_gaussians"),
    ]
    result = {}
    for left, right in pairs:
        data = [(row.get(left), row.get(right)) for row in rows]
        data = [(float(a), float(b)) for a, b in data if a is not None and b is not None]
        key = f"{left}_vs_{right}"
        result[key] = float(np.corrcoef(np.asarray(data).T)[0, 1]) if len(data) >= 3 else None
    return result


def analyse_segmentation(
    scene_root: str | Path, context: dict, ground_truth_root: str | Path | None = None,
) -> tuple[list[dict], dict]:
    scene_root = Path(scene_root)
    metadata = _load_json(scene_root / "traindata" / "layer_instances.json")
    rows, masks = [], []
    groups = {int(g["layer_idx"]): g for g in metadata.get("layer_groups", [])}
    for layer_path in sorted((scene_root / "traindata").glob("layer*/layer*_erp_mask.png")):
        match = re.search(r"layer(\d+)", layer_path.parent.name)
        if not match:
            continue
        index = int(match.group(1))
        mask = np.asarray(Image.open(layer_path).convert("L")) >= 128
        masks.append(mask)
        group = groups.get(index, {})
        target = str(group.get("group_label", f"layer{index}"))
        base = {
            **context, "target": target, "metric_scope": "intrinsic",
            "coverage_percent": float(mask.mean() * 100),
            "seam_crossing": seam_crossing(mask), "source": str(layer_path),
        }
        rows.append(base)
        if ground_truth_root:
            gt_candidates = [
                Path(ground_truth_root) / context["scene"] / f"{target}.png",
                Path(ground_truth_root) / context["scene"] / f"layer{index}.png",
            ]
            gt_path = next((p for p in gt_candidates if p.exists()), None)
            if gt_path:
                gt = np.asarray(Image.open(gt_path).convert("L").resize(
                    (mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST
                )) >= 128
                rows.append({
                    **context, "target": target, "metric_scope": "ground_truth",
                    **mask_metrics(mask, gt), "seam_crossing": seam_crossing(mask),
                    "source": str(gt_path),
                })
    if masks:
        stack = np.stack(masks)
        coverage = np.any(stack, axis=0)
        overlap = np.sum(stack, axis=0) > 1
        unassigned = ~coverage
    else:
        coverage = overlap = unassigned = np.zeros((1, 1), dtype=bool)
    background_index = metadata.get("background_layer_idx")
    background_mask = None
    if background_index is not None:
        path = scene_root / "traindata" / f"layer{background_index}" / f"layer{background_index}_erp_mask.png"
        if path.exists():
            background_mask = np.asarray(Image.open(path).convert("L")) >= 128
    summary = {
        "intrinsic_not_accuracy": True,
        "total_coverage_percent": float(coverage.mean() * 100),
        "unassigned_percent": float(unassigned.mean() * 100),
        "background_percent": float(background_mask.mean() * 100) if background_mask is not None else None,
        "overlap_after_pixels": int(overlap.sum()),
        "overlap_before_pixels": None,
        "discarded_detections": max(
            0, len((metadata.get("grounding") or {}).get("detections") or []) - int(metadata.get("instance_count") or 0)
        ),
        "discarded_small_masks": None,
        "discarded_insufficient_3d_layers": None,
        "seam_crossing_mask_count": sum(bool(row.get("seam_crossing")) for row in rows if row["metric_scope"] == "intrinsic"),
        "sky_coverage_percent": next(
            (row["coverage_percent"] for row in rows if row["target"] == "sky" and row["metric_scope"] == "intrinsic"), None
        ),
    }
    return rows, summary


def analyse_moods(scene_root: str | Path, context: dict) -> list[dict]:
    scene_root = Path(scene_root)
    day_candidates = [
        scene_root / "scene" / "gsplat_scene_merged_refined.ply",
        scene_root / "scene" / "gsplat_scene_merged.ply",
    ]
    day = next((p for p in day_candidates if p.exists()), None)
    if day is None:
        return []
    generation_path = scene_root / "traindata" / "sky" / "night_generation.json"
    generation = _load_json(generation_path) if generation_path.exists() else {}
    rows = []
    for mood in sorted((scene_root / "scene").glob("gsplat_scene_*.ply")):
        if mood in day_candidates or mood.name == "gsplat_scene_active.ply":
            continue
        comparison = compare_topology(day, mood)
        rows.append({
            **context, "day_variant": day.name, "mood_variant": mood.name,
            **comparison, "analytic_fit_seconds": None, "refinement_seconds": None,
            "mood_ply_size_bytes": mood.stat().st_size,
            "circular_seam_mae": generation.get("seam_mae_after"),
            "status": "success" if comparison.get("correspondence_compatible") else "failed",
            "reason": comparison.get("reason"),
        })
    return rows


def analyse_existing_scene(
    scene_root: str | Path, output_dir: str | Path, context: dict,
    ground_truth_root: str | Path | None = None, run_mood_evaluation: bool = True,
) -> dict:
    scene_root, output_dir = Path(scene_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers, layer_summary = analyse_layers(scene_root, context)
    timing_rows = read_csv(output_dir / "stage_timings.csv")
    timing_by_layer = {}
    for timing in timing_rows:
        try:
            index = int(timing.get("layer_index", ""))
        except (TypeError, ValueError):
            continue
        if timing.get("stage") in {"per_layer_training", "background_training", "sky_training"}:
            timing_by_layer[index] = timing
    for row in layers:
        timing = timing_by_layer.get(int(row["layer_index"]))
        if timing:
            row["training_time_seconds"] = timing.get("wall_seconds")
            row["training_iterations"] = timing.get("iterations")
    layer_summary["correlations"] = correlation_summary(layers)
    segmentation, segmentation_summary = analyse_segmentation(scene_root, context, ground_truth_root)
    moods = analyse_moods(scene_root, context) if run_mood_evaluation else []
    analytic_timings = [
        row for row in timing_rows
        if row.get("stage") == "analytic_day_to_mood_gaussian_fitting"
    ]
    refinement_timings = [
        row for row in timing_rows if row.get("stage") == "optional_mood_refinement"
    ]
    for index, row in enumerate(moods):
        if index < len(analytic_timings):
            row["analytic_fit_seconds"] = analytic_timings[index].get("wall_seconds")
        if index < len(refinement_timings):
            row["refinement_seconds"] = refinement_timings[index].get("wall_seconds")
    write_csv(output_dir / "layer_metrics.csv", layers, LAYER_COLUMNS)
    write_csv(output_dir / "segmentation_metrics.csv", segmentation, SEGMENTATION_COLUMNS)
    write_csv(output_dir / "mood_metrics.csv", moods, MOOD_COLUMNS)
    files = list(scene_root.rglob("*"))
    disk = {
        "scene_bytes": sum(p.stat().st_size for p in files if p.is_file() and "scene" in p.parts),
        "traindata_bytes": sum(p.stat().st_size for p in files if p.is_file() and "traindata" in p.parts),
        "ply_bytes": sum(p.stat().st_size for p in files if p.is_file() and p.suffix.lower() == ".ply"),
        "training_frame_bytes": sum(
            p.stat().st_size for p in files if p.is_file() and "frames" in p.parts
        ),
        "mask_bytes": sum(
            p.stat().st_size for p in files
            if p.is_file() and ("mask" in p.name.lower() or "mask" in p.parts)
        ),
        "mood_bytes": sum(
            p.stat().st_size for p in files if p.is_file() and "moods" in p.parts
        ),
        "total_scene_root_bytes": sum(p.stat().st_size for p in files if p.is_file()),
    }
    return {
        "scene_root": str(scene_root.resolve()), "layer_summary": layer_summary,
        "segmentation_summary": segmentation_summary, "disk_usage": disk,
        "mood_variant_count": len(moods),
        "generated_training_views": len(list(
            (scene_root / "traindata" / "perspective_frames" / "frames").glob("rgb_*.png")
        )),
        "input_point_count": sum(
            int(row.get("initial_gaussians") or 0) for row in layers
        ),
        "final_gaussian_count": sum(
            int(row.get("final_gaussians") or 0) for row in layers
        ),
        "usable_merged_ply": any((scene_root / "scene" / name).exists() for name in (
            "gsplat_scene_merged.ply", "gsplat_scene_merged_refined.ply"
        )),
    }
