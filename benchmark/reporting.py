from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from benchmark.io_utils import atomic_json, read_csv, write_csv
from benchmark.schemas import (
    COMPARISON_COLUMNS, EDITING_COLUMNS, LAYER_COLUMNS, MOOD_COLUMNS, QUALITY_SCALING_COLUMNS, RECONSTRUCTION_COLUMNS,
    RENDERING_COLUMNS, RESOURCE_COLUMNS, SEGMENTATION_COLUMNS, STAGE_COLUMNS,
)

TABLES = {
    "stage_timings.csv": STAGE_COLUMNS, "resource_samples.csv": RESOURCE_COLUMNS,
    "layer_metrics.csv": LAYER_COLUMNS, "segmentation_metrics.csv": SEGMENTATION_COLUMNS,
    "reconstruction_metrics.csv": RECONSTRUCTION_COLUMNS,
    "rendering_metrics.csv": RENDERING_COLUMNS, "editing_metrics.csv": EDITING_COLUMNS,
    "mood_metrics.csv": MOOD_COLUMNS,
    "comparison_metrics.csv": COMPARISON_COLUMNS,
}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def statistical_summary(values: list[float]) -> dict:
    data = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(data):
        return {"count": 0, **{key: None for key in ("mean", "median", "std", "min", "max", "p25", "p75", "ci95_low", "ci95_high")}}
    mean, std = float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else None
    half = None
    if len(data) > 1:
        try:
            from scipy.stats import t
            half = float(t.ppf(0.975, df=len(data) - 1)) * std / math.sqrt(len(data))
        except ImportError:
            pass
    return {
        "count": int(len(data)), "mean": mean, "median": float(np.median(data)),
        "std": std, "min": float(data.min()), "max": float(data.max()),
        "p25": float(np.percentile(data, 25)), "p75": float(np.percentile(data, 75)),
        "ci95_low": mean - half if half is not None else None,
        "ci95_high": mean + half if half is not None else None,
    }


def grouped_metric_summary(
    rows: list[dict], group_key: str, metric_keys: tuple[str, ...]
) -> dict:
    result = {}
    for group in sorted({str(row.get(group_key, "")) for row in rows}):
        group_rows = [row for row in rows if str(row.get(group_key, "")) == group]
        result[group] = {
            metric: statistical_summary(
                [
                    value
                    for row in group_rows
                    if (value := _number(row.get(metric))) is not None
                ]
            )
            for metric in metric_keys
        }
    return result


def aggregate_results(input_root: str | Path, output_dir: str | Path) -> dict:
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = input_root / "selected_runs.json"
    selected_dirs = None
    if manifest.exists():
        selected_dirs = {Path(item["output"]).resolve() for item in json.loads(manifest.read_text())}
    else:
        # Historical attempts are not independent scene replicates. Without an
        # explicit selection manifest, use the latest attempt, including failures.
        latest = {}
        for summary in input_root.rglob("run_summary.json"):
            if summary.parent == output_dir or "report" in summary.relative_to(input_root).parts:
                continue
            key = summary.parent.parent
            if key not in latest or summary.stat().st_mtime_ns > latest[key].stat().st_mtime_ns:
                latest[key] = summary
        if latest:
            selected_dirs = {path.parent.resolve() for path in latest.values()}

    def is_aggregate_artifact(path: Path) -> bool:
        return (
            (selected_dirs is not None and path.parent.resolve() not in selected_dirs)
            or output_dir in path.parents
            or any(parent.name == "report" for parent in path.parents)
        )

    aggregated = {}
    for filename, columns in TABLES.items():
        rows = []
        for path in input_root.rglob(filename):
            if is_aggregate_artifact(path):
                continue
            rows.extend(read_csv(path))
        write_csv(output_dir / filename, rows, columns)
        aggregated[filename] = len(rows)
    reconstruction_rows = read_csv(output_dir / "reconstruction_metrics.csv")
    scene_reconstruction_rows = scene_mean_rows(reconstruction_rows)
    rendering_rows = read_csv(output_dir / "rendering_metrics.csv")
    editing_rows = read_csv(output_dir / "editing_metrics.csv")
    mood_rows = read_csv(output_dir / "mood_metrics.csv")
    stage_rows = read_csv(output_dir / "stage_timings.csv")
    comparison_rows = read_csv(output_dir / "comparison_metrics.csv")
    quality_scaling_rows = []
    for experiment, base_scene in sorted({(row.get("experiment", ""), row.get("base_scene", "")) for row in comparison_rows}):
        scene_rows = [row for row in comparison_rows if row.get("base_scene") == base_scene and row.get("experiment", "") == experiment]
        qualities = {row.get("quality") for row in scene_rows}
        if len(qualities) < 2:
            continue
        by_quality_variant = {
            (row.get("quality"), row.get("variant")): row for row in scene_rows
        }
        standard_layered = by_quality_variant.get(("standard", "layered"), {})
        standard_monolithic = by_quality_variant.get(("standard", "monolithic"), {})

        def percent_change(value, baseline):
            return 100 * (value - baseline) / baseline if value is not None and baseline else None

        for quality in ("low", "standard", "high", "ultra"):
            if quality not in qualities:
                continue
            layered = by_quality_variant.get((quality, "layered"), {})
            monolithic = by_quality_variant.get((quality, "monolithic"), {})
            layered_time = _number(layered.get("training_time_seconds"))
            monolithic_time = _number(monolithic.get("training_time_seconds"))
            layered_memory = _number(layered.get("peak_training_rss_bytes"))
            monolithic_memory = _number(monolithic.get("peak_training_rss_bytes"))
            complete = all(value is not None for value in (
                layered_time, monolithic_time, layered_memory, monolithic_memory,
            ))
            quality_scaling_rows.append({
                "experiment": layered.get("experiment") or monolithic.get("experiment"),
                "base_scene": base_scene,
                "quality": quality,
                "downsample_ratio": layered.get("downsample_ratio") or monolithic.get("downsample_ratio"),
                "layered_training_time_seconds": layered_time,
                "monolithic_training_time_seconds": monolithic_time,
                "layered_peak_training_rss_bytes": layered_memory,
                "monolithic_peak_training_rss_bytes": monolithic_memory,
                "layered_speedup_percent": (
                    100 * (monolithic_time - layered_time) / monolithic_time
                    if layered_time is not None and monolithic_time else None
                ),
                "layered_peak_memory_reduction_percent": (
                    100 * (monolithic_memory - layered_memory) / monolithic_memory
                    if layered_memory is not None and monolithic_memory else None
                ),
                "baseline_quality": "standard",
                "layered_time_change_vs_standard_percent": percent_change(
                    layered_time, _number(standard_layered.get("training_time_seconds"))
                ),
                "monolithic_time_change_vs_standard_percent": percent_change(
                    monolithic_time, _number(standard_monolithic.get("training_time_seconds"))
                ),
                "layered_memory_change_vs_standard_percent": percent_change(
                    layered_memory, _number(standard_layered.get("peak_training_rss_bytes"))
                ),
                "monolithic_memory_change_vs_standard_percent": percent_change(
                    monolithic_memory, _number(standard_monolithic.get("peak_training_rss_bytes"))
                ),
                "status": "success" if complete else "partial",
                "reason": None if complete else "Missing paired training time or stage-local peak RSS",
            })
    write_csv(output_dir / "quality_scaling.csv", quality_scaling_rows, QUALITY_SCALING_COLUMNS)
    aggregated["quality_scaling.csv"] = len(quality_scaling_rows)
    summaries = {
        "stage_wall_seconds": statistical_summary([
            value for row in stage_rows
            if (value := _number(row.get("wall_seconds"))) is not None
        ]),
        "layer_final_gaussians": statistical_summary([
            value for row in read_csv(output_dir / "layer_metrics.csv")
            if (value := _number(row.get("final_gaussians"))) is not None
        ]),
        "reconstruction_psnr_db": statistical_summary([
            value for row in reconstruction_rows
            if (value := _number(row.get("psnr_db"))) is not None
        ]),
        "rendering_fps": statistical_summary([
            value for row in rendering_rows
            if (value := _number(row.get("average_fps"))) is not None
        ]),
        "reconstruction_by_variant": grouped_metric_summary(
            scene_reconstruction_rows, "variant", ("psnr_db", "ssim", "lpips", "mae")
        ),
        "comparison_by_variant": grouped_metric_summary(
            comparison_rows,
            "variant",
            (
                "psnr_db", "ssim", "lpips", "mae", "training_time_seconds",
                "peak_training_rss_bytes", "average_fps", "mean_ms", "p95_ms",
                "gaussian_count", "ply_size_bytes",
            ),
        ),
        "rendering_by_variant": grouped_metric_summary(
            rendering_rows,
            "variant",
            (
                "average_fps", "mean_ms", "p95_ms", "gaussian_count",
                "ply_size_bytes",
            ),
        ),
        "editing_layered": {
            metric: statistical_summary(
                [
                    value
                    for row in editing_rows
                    if row.get("variant") == "layered"
                    and (value := _number(row.get(metric))) is not None
                ]
            )
            for metric in (
                "edit_leakage_ratio",
                "edit_locality_score",
                "outside_changed_percent",
                "creation_seconds",
            )
        },
        "mood_topology": {
            metric: statistical_summary(
                [
                    value
                    for row in mood_rows
                    if (value := _number(row.get(metric))) is not None
                ]
            )
            for metric in (
                "gaussian_count_difference",
                "position_max_abs",
                "scale_max_abs",
                "rotation_max_abs",
                "opacity_max_abs",
                "nonappearance_changed_percent",
                "analytic_fit_seconds",
                "circular_seam_mae",
            )
        },
        "pipeline_stage_by_name": grouped_metric_summary(
            stage_rows, "stage", ("wall_seconds", "peak_sampled_rss_bytes")
        ),
        "aggregation_unit": "Scene-run means; paired comparisons require identical view IDs. CI uses Student t, without a significance test.",
        "scope_warning": (
            "The legacy reconstruction_psnr_db and rendering_fps summaries mix "
            "variants and are diagnostic only. Use the by-variant summaries for claims."
        ),
    }
    summaries["paired_monolithic_vs_layered"] = paired_variant_summary(
        read_csv(output_dir / "reconstruction_metrics.csv"), "psnr_db"
    )
    run_summaries = []
    for path in input_root.rglob("run_summary.json"):
        if is_aggregate_artifact(path):
            continue
        try:
            run_summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    failures = [row for row in run_summaries if row.get("status") not in ("success", "partial_success")]
    summaries["run_resources"] = {
        "final_gaussian_count": statistical_summary(
            [_number(row.get("final_gaussian_count")) for row in run_summaries]
        ),
        "peak_process_rss_bytes": statistical_summary(
            [_number(row.get("peak_process_rss_bytes")) for row in run_summaries]
        ),
    }
    status = {
        "scene_runs": len(run_summaries),
        "successful": sum(row.get("status") == "success" for row in run_summaries),
        "partial": sum(row.get("status") == "partial_success" for row in run_summaries),
        "failed": len(failures),
        "success_rate_percent": (
            100 * sum(row.get("status") == "success" for row in run_summaries) / len(run_summaries)
            if run_summaries else None
        ),
        "stage_failure_counts": _counts(row.get("failed_stage") for row in failures),
        "failure_reasons": _counts(row.get("exception_message") for row in failures),
        "usable_merged_ply_percent": _percent(
            run_summaries, lambda row: bool(row.get("usable_merged_ply"))
        ),
        "valid_semantic_layers_percent": _percent(
            run_summaries,
            lambda row: int((row.get("layer_summary") or {}).get("trained_layer_count") or 0) > 0,
        ),
        "valid_mood_variant_percent": _percent(
            run_summaries, lambda row: int(row.get("mood_variant_count") or 0) > 0
        ),
    }
    payload = {
        "row_counts": aggregated, "statistics": summaries,
        "robustness": status, "runs": run_summaries,
    }
    atomic_json(output_dir / "aggregated_summary.json", payload)
    generate_plots(output_dir)
    generate_report(output_dir, payload)
    return payload


def _counts(values) -> dict:
    result = {}
    for value in values:
        if value:
            result[str(value)] = result.get(str(value), 0) + 1
    return result


def scene_mean_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        if row.get("status") not in (None, "", "success"):
            continue
        key = tuple(row.get(k, "") for k in ("experiment", "scene", "run_id", "variant"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in groups.items():
        row = dict(zip(("experiment", "scene", "run_id", "variant"), key))
        for metric in ("psnr_db", "ssim", "lpips", "mae"):
            numbers = [_number(value.get(metric)) for value in values]
            numbers = [n for n in numbers if n is not None]
            row[metric] = float(np.mean(numbers)) if numbers else None
        result.append(row)
    return result


def paired_variant_summary(rows: list[dict], metric: str) -> dict:
    groups = {}
    for row in rows:
        value = _number(row.get(metric))
        variant = row.get("variant")
        if row.get("status") not in (None, "", "success"):
            continue
        if value is not None and variant in {"layered", "monolithic"}:
            key = tuple(row.get(k, "") for k in ("experiment", "scene", "run_id"))
            groups.setdefault(key, {}).setdefault(variant, {})[str(row.get("view_id", ""))] = value
    pairs = []
    for key, variants in groups.items():
        layered_views = variants.get("layered", {})
        monolithic_views = variants.get("monolithic", {})
        # Never compare different view subsets or different run attempts.
        if layered_views and layered_views.keys() == monolithic_views.keys():
            layered = float(np.mean(list(layered_views.values())))
            monolithic = float(np.mean(list(monolithic_views.values())))
            pairs.append({
                **dict(zip(("experiment", "scene", "run_id"), key)),
                "matched_views": len(layered_views),
                "layered": layered, "monolithic": monolithic,
                "difference_monolithic_minus_layered": monolithic - layered,
            })
    return {
        "pair_count": len(pairs), "pairs": pairs,
        "difference_summary": statistical_summary([p["difference_monolithic_minus_layered"] for p in pairs]),
        "significance_test": None,
        "note": "Paired within the same run and identical view sets; no automatic significance test.",
    }


def _percent(rows: list[dict], predicate) -> float | None:
    return 100.0 * sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None


def generate_plots(output_dir: str | Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stages = read_csv(output_dir / "stage_timings.csv")
    layers = read_csv(output_dir / "layer_metrics.csv")
    renders = read_csv(output_dir / "rendering_metrics.csv")
    reconstruction = read_csv(output_dir / "reconstruction_metrics.csv")
    edits = read_csv(output_dir / "editing_metrics.csv")
    moods = read_csv(output_dir / "mood_metrics.csv")
    resources = read_csv(output_dir / "resource_samples.csv")
    comparisons = read_csv(output_dir / "comparison_metrics.csv")
    comparison_plot_rows = [
        {
            **row,
            "training_minutes": (
                _number(row.get("training_time_seconds")) / 60
                if _number(row.get("training_time_seconds")) is not None else None
            ),
            "peak_training_gib": (
                _number(row.get("peak_training_rss_bytes")) / 1024 ** 3
                if _number(row.get("peak_training_rss_bytes")) is not None else None
            ),
        }
        for row in comparisons
    ]
    quality_plot_rows = [
        {
            **row,
            "scene_quality": (
                f"{row.get('base_scene')}:{row.get('quality')}:{row.get('variant')}"
            ),
        }
        for row in comparison_plot_rows
        if row.get("variant") in {"layered", "monolithic"} and row.get("quality")
    ]
    try:
        aggregate_payload = json.loads((output_dir / "aggregated_summary.json").read_text())
        robustness = aggregate_payload.get("robustness", {})
    except (OSError, json.JSONDecodeError):
        aggregate_payload = {}
        robustness = {}
    status_rows = [
        {"status": label, "count": robustness.get(key)}
        for label, key in (("Success", "successful"), ("Partial", "partial"), ("Failed", "failed"))
    ]
    total_runtime_rows = []
    for scene in sorted({row.get("scene", "") for row in stages}):
        scene_rows = [row for row in stages if row.get("scene", "") == scene]
        complete = [row for row in scene_rows if row.get("stage") == "complete_end_to_end"]
        selected = complete or [
            row for row in scene_rows
            if not row.get("parent_stage") and row.get("stage") not in {
                "all_layer_training", "generation_of_layer_training_data"
            }
        ]
        values = [_number(row.get("wall_seconds")) for row in selected]
        total_runtime_rows.append({
            "scene": scene, "wall_seconds": sum(value for value in values if value is not None)
        })
    breakdown_stages = [
        row for row in stages if row.get("stage") not in {
            "complete_end_to_end", "all_layer_training",
            "generation_of_layer_training_data", "object_detection_and_segmentation",
        }
    ]
    edit_plot_rows = [
        {
            **row,
            "scene_target": (
                f"{row.get('scene', '')}:{row.get('target_type', '')}"
                f"_{row.get('target_id', '')}"
            ),
        }
        for row in edits
    ]
    monolithic_layered = [
        row for row in reconstruction
        if row.get("variant") in {"layered", "monolithic"}
    ]
    definitions = [
        ("total_runtime_by_scene", "Total runtime by scene", "Scene", "Seconds", total_runtime_rows, "scene", "wall_seconds", "bar"),
        ("runtime_breakdown_by_stage", "Runtime breakdown by stage", "Stage", "Seconds", breakdown_stages, "stage", "wall_seconds", "bar"),
        ("peak_memory_by_scene", "Peak process memory by scene", "Scene", "GiB", resources, "scene", "process_rss_bytes", "max_gib"),
        ("training_time_vs_final_gaussians", "Training time vs final Gaussian count", "Final Gaussians", "Seconds", layers, "final_gaussians", "training_time_seconds", "scatter"),
        ("reconstruction_quality_comparison", "Reconstruction quality comparison", "Variant", "Mean PSNR (dB)", reconstruction, "variant", "psnr_db", "mean_bar"),
        ("monolithic_vs_layered", "Monolithic vs layered comparison", "Variant", "Mean PSNR (dB)", monolithic_layered, "variant", "psnr_db", "mean_bar"),
        ("training_time_by_variant", "Training time by representation", "Variant", "Minutes", comparison_plot_rows, "variant", "training_minutes", "mean_bar"),
        ("training_memory_by_variant", "Peak training memory by representation", "Variant", "GiB", comparison_plot_rows, "variant", "peak_training_gib", "mean_bar"),
        ("quality_vs_training_time", "Per-scene quality vs training time", "Scene:quality:variant", "Minutes", quality_plot_rows, "scene_quality", "training_minutes", "bar"),
        ("quality_vs_peak_memory", "Per-scene quality vs peak training memory", "Scene:quality:variant", "GiB", quality_plot_rows, "scene_quality", "peak_training_gib", "bar"),
        ("rendering_fps_vs_gaussians", "Rendering FPS vs Gaussian count", "Gaussians", "FPS", renders, "gaussian_count", "average_fps", "scatter"),
        ("edit_leakage_by_target", "Edit leakage by scene and target", "Scene:target", "Leakage ratio", edit_plot_rows, "scene_target", "edit_leakage_ratio", "mean_bar"),
        ("day_night_property_differences", "Day/night non-appearance differences", "Mood", "Changed Gaussians (%)", moods, "mood_variant", "nonappearance_changed_percent", "mean_bar"),
        ("success_failure_summary", "Success/failure summary", "Status", "Runs", status_rows, "status", "count", "bar"),
    ]
    outputs = []
    for slug, title, xlabel, ylabel, rows, xkey, ykey, kind in definitions:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        x, y = [], []
        for row in rows:
            value = _number(row.get(ykey))
            if value is not None:
                if kind == "max_gib":
                    value /= 1024 ** 3
                x.append(row.get(xkey, ""))
                y.append(value)
        if x and kind in ("bar", "mean_bar", "max_gib"):
            labels = list(dict.fromkeys(x))
            totals = [
                (
                    sum(v for key, v in zip(x, y) if key == label)
                    / sum(1 for key in x if key == label)
                    if kind == "mean_bar"
                    else sum(v for key, v in zip(x, y) if key == label)
                )
                for label in labels
            ]
            if kind == "max_gib":
                totals = [max(v for key, v in zip(x, y) if key == label) for label in labels]
            ax.bar(range(len(labels)), totals)
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        elif x and kind == "scatter":
            numeric_x = [_number(value) for value in x]
            valid = [(a, b) for a, b in zip(numeric_x, y) if a is not None]
            if valid:
                ax.scatter([a for a, _ in valid], [b for _, b in valid])
        elif y and kind == "hist":
            ax.hist(y, bins=min(15, max(3, int(math.sqrt(len(y))))))
        else:
            ax.text(0.5, 0.5, "Metric unavailable for this run", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = plot_dir / f"{slug}.{suffix}"
            fig.savefig(path, dpi=180)
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def generate_report(output_dir: str | Path, summary: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    if summary is None:
        path = output_dir / "aggregated_summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
    robustness = summary.get("robustness", {})
    statistics = summary.get("statistics", {})

    def mean(group: dict, metric: str):
        value = ((group or {}).get(metric) or {}).get("mean")
        return float(value) if value is not None else None

    def median(group: dict, metric: str):
        value = ((group or {}).get(metric) or {}).get("median")
        return float(value) if value is not None else None

    def formatted(value, digits=3, scale=1.0, suffix=""):
        return "n/a" if value is None else f"{value / scale:.{digits}f}{suffix}"

    reconstruction = statistics.get("reconstruction_by_variant", {})
    rendering = statistics.get("rendering_by_variant", {})
    editing = statistics.get("editing_layered", {})
    mood = statistics.get("mood_topology", {})
    stages = statistics.get("pipeline_stage_by_name", {})
    comparison = statistics.get("comparison_by_variant", {})
    paired = statistics.get("paired_monolithic_vs_layered", {})
    quality_rows = []
    for variant in ("layered", "refined", "monolithic"):
        if variant not in reconstruction:
            continue
        quality_rows.append(
            f"| {variant} | "
            f"{formatted(mean(reconstruction[variant], 'psnr_db'), 2)} | "
            f"{formatted(mean(reconstruction[variant], 'ssim'), 3)} | "
            f"{formatted(mean(reconstruction[variant], 'lpips'), 3)} | "
            f"{formatted(mean(reconstruction[variant], 'mae'), 4)} |"
        )
    rendering_rows = []
    for variant in ("layered", "refined", "monolithic", "night"):
        if variant not in rendering:
            continue
        rendering_rows.append(
            f"| {variant} | "
            f"{formatted(mean(rendering[variant], 'average_fps'), 3)} | "
            f"{formatted(mean(rendering[variant], 'mean_ms'), 1)} | "
            f"{formatted(mean(rendering[variant], 'p95_ms'), 1)} | "
            f"{formatted(mean(rendering[variant], 'gaussian_count'), 3, 1e6, ' M')} | "
            f"{formatted(mean(rendering[variant], 'ply_size_bytes'), 3, 1024 ** 3, ' GiB')} |"
        )
    difference = ((paired.get("difference_summary") or {}).get("mean"))
    comparison_detail_rows = []
    for row in read_csv(output_dir / "comparison_metrics.csv"):
        comparison_detail_rows.append(
            f"| {row.get('base_scene') or row.get('scene')} | {row.get('quality')} | "
            f"{row.get('variant')} | {formatted(_number(row.get('psnr_db')), 2)} | "
            f"{formatted(_number(row.get('ssim')), 3)} | "
            f"{formatted(_number(row.get('lpips')), 3)} | "
            f"{formatted(_number(row.get('mae')), 4)} | "
            f"{formatted(_number(row.get('training_time_seconds')), 1, 60, ' min')} | "
            f"{formatted(_number(row.get('peak_training_rss_bytes')), 2, 1024 ** 3, ' GiB')} | "
            f"{formatted(_number(row.get('average_fps')), 3)} |"
        )
    quality_scaling_detail_rows = []
    for row in read_csv(output_dir / "quality_scaling.csv"):
        quality_scaling_detail_rows.append(
            f"| {row.get('base_scene')} | {row.get('quality')} | "
            f"{formatted(_number(row.get('downsample_ratio')), 2)}× | "
            f"{formatted(_number(row.get('layered_training_time_seconds')), 1, 60, ' min')} | "
            f"{formatted(_number(row.get('monolithic_training_time_seconds')), 1, 60, ' min')} | "
            f"{formatted(_number(row.get('layered_speedup_percent')), 1, suffix='%')} | "
            f"{formatted(_number(row.get('layered_peak_training_rss_bytes')), 2, 1024 ** 3, ' GiB')} | "
            f"{formatted(_number(row.get('monolithic_peak_training_rss_bytes')), 2, 1024 ** 3, ' GiB')} | "
            f"{formatted(_number(row.get('layered_peak_memory_reduction_percent')), 1, suffix='%')} |"
        )
    quality_scaling_lines = []
    if quality_scaling_detail_rows:
        quality_scaling_lines = [
            "## Quality scaling by scene", "",
            "Positive speed-up or memory-reduction values favour the layered representation.", "",
            "| Scene | Quality | 3D support | Layered training | Monolithic training | Layered speed-up | Layered RSS | Monolithic RSS | RSS reduction |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            *quality_scaling_detail_rows,
            "",
        ]
    optional_lines = []
    if mean(editing, "edit_leakage_ratio") is not None:
        optional_lines.extend([
            "## Object-edit locality", "",
            f"- Leakage: mean {formatted(mean(editing, 'edit_leakage_ratio'), 5)}, "
            f"median {formatted(median(editing, 'edit_leakage_ratio'), 5)}",
            f"- Locality: mean {formatted(mean(editing, 'edit_locality_score'), 5)}, "
            f"median {formatted(median(editing, 'edit_locality_score'), 5)}",
            "",
        ])
    if mean(mood, "nonappearance_changed_percent") is not None:
        optional_lines.extend([
            "## Day/night topology", "",
            f"- Non-appearance properties changed: "
            f"{formatted(mean(mood, 'nonappearance_changed_percent'), 5, suffix='%')}",
            f"- Maximum position difference: "
            f"{formatted(mean(mood, 'position_max_abs'), 6)}",
            "",
        ])
    lines = [
        "# ObjSplat Benchmark Report", "",
        "## Run status", "",
        f"- Scene runs: {robustness.get('scene_runs', 0)}",
        f"- Successful: {robustness.get('successful', 0)}",
        f"- Partial: {robustness.get('partial', 0)}",
        f"- Failed: {robustness.get('failed', 0)}",
        "",
        "## Reconstruction fidelity", "",
        "| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MAE ↓ |",
        "|---|---:|---:|---:|---:|",
        *quality_rows,
        "",
        f"Mean paired monolithic − layered PSNR: "
        f"{formatted(difference, 3, suffix=' dB')}.",
        "",
        "## Computational efficiency", "",
        f"- Layer training: "
        f"{formatted(mean(comparison.get('layered', {}), 'training_time_seconds'), 1, 60, ' min')}",
        f"- Monolithic training: "
        f"{formatted(mean(comparison.get('monolithic', {}), 'training_time_seconds'), 1, 60, ' min')}",
        f"- Layered peak training memory: "
        f"{formatted(mean(comparison.get('layered', {}), 'peak_training_rss_bytes'), 2, 1024 ** 3, ' GiB')}",
        f"- Monolithic peak training memory: "
        f"{formatted(mean(comparison.get('monolithic', {}), 'peak_training_rss_bytes'), 2, 1024 ** 3, ' GiB')}",
        "",
        "| Variant | FPS ↑ | Mean ms ↓ | p95 ms ↓ | Gaussian count | PLY size |",
        "|---|---:|---:|---:|---:|---:|",
        *rendering_rows,
        "",
        "## Per-scene comparison", "",
        "| Scene | Quality | Variant | PSNR | SSIM | LPIPS | MAE | Training | Peak RSS | FPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *comparison_detail_rows,
        "",
        *quality_scaling_lines,
        *optional_lines,
        "## Validity and limitations", "",
        "Held-out perspective views derived from the input ERP measure panorama reconstruction fidelity. "
        "They do not constitute independent viewpoints and must not be interpreted as true multi-view "
        "geometric or novel-view accuracy.",
        "",
        "Intrinsic mask coverage describes partition completeness, not segmentation accuracy. Accuracy "
        "metrics are reported only when manual ground-truth masks are supplied.",
        "",
        "Process RSS and system memory on Apple Silicon are unified-memory estimates, not isolated CUDA VRAM.",
        "",
        "Instance-level editing is reported only when the Gaussian PLY retains an integer `label` property. "
        "Otherwise the result is unavailable; shared semantic layers support layer-level removal only.",
        "",
        "Missing optional metrics are reported as `n/a`, never imputed. Raw CSV and JSON files remain "
        "available for audit but are intentionally omitted from this concise report.",
        "", "## Scientific plots", "",
    ]
    important_plots = (
        "monolithic_vs_layered",
        "training_time_by_variant",
        "training_memory_by_variant",
        "quality_vs_training_time",
        "quality_vs_peak_memory",
        "rendering_fps_vs_gaussians",
    )
    for stem in important_plots:
        png = output_dir / "plots" / f"{stem}.png"
        if png.exists():
            lines.extend((f"![{stem}](plots/{png.name})", ""))
    report = output_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
