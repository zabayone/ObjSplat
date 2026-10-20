from __future__ import annotations

import json
import math
from pathlib import Path

from benchmark.io_utils import read_csv, write_csv
from benchmark.schemas import COMPARISON_COLUMNS


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict], key: str):
    values = [_number(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _pipeline_arg(config: dict, scene: dict, flag: str, default: str) -> str:
    args = list(config.get("pipeline_args", [])) + list(scene.get("pipeline_args", []))
    value = default
    for index, item in enumerate(args[:-1]):
        if str(item) == flag:
            value = str(args[index + 1])
    return value


def _quality(config: dict, scene: dict) -> str:
    value = _pipeline_arg(config, scene, "--quality", "standard")
    return str((scene.get("ablation") or {}).get("quality", value))


def write_comparison_metrics(
    scene_root: str | Path,
    output_dir: str | Path,
    context: dict,
    config: dict,
    scene: dict,
) -> list[dict]:
    """Create one audit-friendly row for each comparable representation."""
    scene_root, output_dir = Path(scene_root), Path(output_dir)
    reconstruction = read_csv(output_dir / "reconstruction_metrics.csv")
    rendering = read_csv(output_dir / "rendering_metrics.csv")
    timings = read_csv(output_dir / "stage_timings.csv")
    resources = read_csv(output_dir / "resource_samples.csv")

    metadata = {}
    metadata_path = scene_root / "traindata" / "layer_instances.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass

    def stage_time(stages: tuple[str, ...]):
        preferred = next(
            (stage for stage in stages if any(row.get("stage") == stage for row in timings)),
            stages[-1],
        )
        values = [
            _number(row.get("wall_seconds"))
            for row in timings
            if row.get("stage") == preferred and row.get("status") == "success"
        ]
        values = [value for value in values if value is not None]
        return max(values) if values else None

    def stage_peak(stages: tuple[str, ...]):
        preferred = next(
            (stage for stage in stages if any(row.get("stage") == stage for row in resources)),
            stages[-1],
        )
        values = [
            _number(row.get("process_rss_bytes"))
            for row in resources
            if row.get("stage") == preferred
        ]
        values = [value for value in values if value is not None]
        return max(values) if values else None

    rows = []
    base = str(scene.get("base_scene_name") or scene.get("name") or scene_root.name)
    for variant, training_stages in (
        ("layered", ("all_layer_training",)),
        ("monolithic", ("monolithic_training_process", "monolithic_baseline")),
    ):
        if variant == "monolithic" and not config.get("run_monolithic_baseline"):
            continue
        fidelity = [
            row for row in reconstruction
            if row.get("variant") == variant and row.get("status") == "success"
        ]
        render = next((
            row for row in rendering
            if row.get("variant") == variant and row.get("status") == "success"
        ), {})
        training_time = stage_time(training_stages)
        peak_rss = stage_peak(training_stages)
        missing = []
        if config.get("run_quality_evaluation") and not fidelity:
            missing.append("fidelity")
        if config.get("run_rendering_benchmark") and not render:
            missing.append("rendering")
        if (config.get("retrain") or variant == "monolithic") and training_time is None:
            missing.append("training time")
        if (config.get("retrain") or variant == "monolithic") and peak_rss is None:
            missing.append("stage-local peak RSS")
        rows.append({
            **context,
            "base_scene": base,
            "quality": _quality(config, scene),
            "variant": variant,
            "psnr_db": _mean(fidelity, "psnr_db"),
            "ssim": _mean(fidelity, "ssim"),
            "lpips": _mean(fidelity, "lpips"),
            "mae": _mean(fidelity, "mae"),
            "held_out_views": len(fidelity),
            "training_time_seconds": training_time,
            "peak_training_rss_bytes": peak_rss,
            "gaussian_count": _number(render.get("gaussian_count")),
            "ply_size_bytes": _number(render.get("ply_size_bytes")),
            "cold_start_seconds": _number(render.get("cold_start_seconds")),
            "mean_ms": _number(render.get("mean_ms")),
            "median_ms": _number(render.get("median_ms")),
            "p95_ms": _number(render.get("p95_ms")),
            "average_fps": _number(render.get("average_fps")),
            "megapixels_per_second": _number(render.get("megapixels_per_second")),
            "training_layer_count": (
                metadata.get("training_layer_count") if variant == "layered" else 1
            ),
            "coverage_3d": metadata.get("coverage_3d"),
            "downsample_ratio": _number(
                _pipeline_arg(config, scene, "--downsample_ratio", "1.0")
            ),
            "status": "success" if not missing else "partial",
            "reason": ", ".join(missing) if missing else None,
        })
    write_csv(output_dir / "comparison_metrics.csv", rows, COMPARISON_COLUMNS)
    return rows
