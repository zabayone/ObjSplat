from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from benchmark.io_utils import atomic_json, read_csv, write_csv
from benchmark.schemas import (
    EDITING_COLUMNS, LAYER_COLUMNS, MOOD_COLUMNS, RECONSTRUCTION_COLUMNS,
    RENDERING_COLUMNS, RESOURCE_COLUMNS, SEGMENTATION_COLUMNS, STAGE_COLUMNS,
)

TABLES = {
    "stage_timings.csv": STAGE_COLUMNS, "resource_samples.csv": RESOURCE_COLUMNS,
    "layer_metrics.csv": LAYER_COLUMNS, "segmentation_metrics.csv": SEGMENTATION_COLUMNS,
    "reconstruction_metrics.csv": RECONSTRUCTION_COLUMNS,
    "rendering_metrics.csv": RENDERING_COLUMNS, "editing_metrics.csv": EDITING_COLUMNS,
    "mood_metrics.csv": MOOD_COLUMNS,
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
        return {key: None for key in ("count", "mean", "median", "std", "min", "max", "p25", "p75", "ci95_low", "ci95_high")}
    mean, std = float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(data)) if len(data) > 1 else None
    return {
        "count": int(len(data)), "mean": mean, "median": float(np.median(data)),
        "std": std, "min": float(data.min()), "max": float(data.max()),
        "p25": float(np.percentile(data, 25)), "p75": float(np.percentile(data, 75)),
        "ci95_low": mean - half if half is not None else None,
        "ci95_high": mean + half if half is not None else None,
    }


def aggregate_results(input_root: str | Path, output_dir: str | Path) -> dict:
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated = {}
    for filename, columns in TABLES.items():
        rows = []
        for path in input_root.rglob(filename):
            if output_dir in path.parents:
                continue
            rows.extend(read_csv(path))
        write_csv(output_dir / filename, rows, columns)
        aggregated[filename] = len(rows)
    summaries = {
        "stage_wall_seconds": statistical_summary([
            value for row in read_csv(output_dir / "stage_timings.csv")
            if (value := _number(row.get("wall_seconds"))) is not None
        ]),
        "layer_final_gaussians": statistical_summary([
            value for row in read_csv(output_dir / "layer_metrics.csv")
            if (value := _number(row.get("final_gaussians"))) is not None
        ]),
        "reconstruction_psnr_db": statistical_summary([
            value for row in read_csv(output_dir / "reconstruction_metrics.csv")
            if (value := _number(row.get("psnr_db"))) is not None
        ]),
        "rendering_fps": statistical_summary([
            value for row in read_csv(output_dir / "rendering_metrics.csv")
            if (value := _number(row.get("average_fps"))) is not None
        ]),
    }
    summaries["paired_monolithic_vs_layered"] = paired_variant_summary(
        read_csv(output_dir / "reconstruction_metrics.csv"), "psnr_db"
    )
    run_summaries = []
    for path in input_root.rglob("run_summary.json"):
        if output_dir in path.parents:
            continue
        try:
            run_summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    failures = [row for row in run_summaries if row.get("status") not in ("success", "partial_success")]
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


def paired_variant_summary(rows: list[dict], metric: str) -> dict:
    by_scene: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = _number(row.get(metric))
        variant = row.get("variant")
        if value is not None and variant in {"layered", "monolithic"}:
            by_scene.setdefault(row.get("scene", ""), {}).setdefault(variant, []).append(value)
    pairs = []
    for variants in by_scene.values():
        if variants.get("layered") and variants.get("monolithic"):
            layered = float(np.mean(variants["layered"]))
            monolithic = float(np.mean(variants["monolithic"]))
            pairs.append({
                "layered": layered, "monolithic": monolithic,
                "difference_monolithic_minus_layered": monolithic - layered,
                "percent_difference": (monolithic - layered) / abs(layered) * 100 if layered else None,
            })
    differences = [row["difference_monolithic_minus_layered"] for row in pairs]
    return {
        "pair_count": len(pairs), "pairs": pairs,
        "difference_summary": statistical_summary(differences),
        "significance_test": None,
        "note": "No automatic significance test; sample-size assumptions are not guaranteed.",
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
    layer_scene_rows = []
    for scene in sorted({row.get("scene", "") for row in layers}):
        scene_layers = [row for row in layers if row.get("scene", "") == scene]
        times = [_number(row.get("training_time_seconds")) for row in scene_layers]
        layer_scene_rows.append({
            "layer_count": len(scene_layers),
            "training_time_seconds": (
                sum(value for value in times if value is not None)
                if any(value is not None for value in times) else None
            ),
        })
    try:
        aggregate_payload = json.loads((output_dir / "aggregated_summary.json").read_text())
        robustness = aggregate_payload.get("robustness", {})
    except (OSError, json.JSONDecodeError):
        aggregate_payload = {}
        robustness = {}
    memory_gaussian_rows = [
        {
            "final_gaussians": row.get("final_gaussian_count"),
            "peak_memory_gib": (
                float(row["peak_process_rss_bytes"]) / 1024 ** 3
                if row.get("peak_process_rss_bytes") else None
            ),
        }
        for row in aggregate_payload.get("runs", [])
    ]
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
    definitions = [
        ("total_runtime_by_scene", "Total runtime by scene", "Scene", "Seconds", total_runtime_rows, "scene", "wall_seconds", "bar"),
        ("runtime_breakdown_by_stage", "Runtime breakdown by stage", "Stage", "Seconds", breakdown_stages, "stage", "wall_seconds", "bar"),
        ("peak_memory_by_scene", "Peak process memory by scene", "Scene", "GiB", resources, "scene", "process_rss_bytes", "max_gib"),
        ("training_time_vs_final_gaussians", "Training time vs final Gaussian count", "Final Gaussians", "Seconds", layers, "final_gaussians", "training_time_seconds", "scatter"),
        ("memory_vs_final_gaussians", "Memory vs final Gaussian count", "Final Gaussians", "Peak GiB", memory_gaussian_rows, "final_gaussians", "peak_memory_gib", "scatter"),
        ("training_time_vs_layers", "Training time vs number of layers", "Number of layers", "Seconds", layer_scene_rows, "layer_count", "training_time_seconds", "scatter"),
        ("mask_coverage_vs_projected_points", "Mask coverage vs projected 3D points", "Coverage (%)", "Points", layers, "mask_coverage_percent", "projected_3d_points", "scatter"),
        ("projected_points_vs_final_gaussians", "Projected points vs final Gaussians", "Projected points", "Final Gaussians", layers, "projected_3d_points", "final_gaussians", "scatter"),
        ("per_layer_training_time_distribution", "Per-layer training-time distribution", "Layers", "Seconds", layers, "layer_index", "training_time_seconds", "hist"),
        ("reconstruction_quality_comparison", "Reconstruction quality comparison", "Variant", "Mean PSNR (dB)", reconstruction, "variant", "psnr_db", "mean_bar"),
        ("monolithic_vs_layered", "Monolithic vs layered comparison", "Variant", "Mean PSNR (dB)", reconstruction, "variant", "psnr_db", "mean_bar"),
        ("rendering_fps_vs_gaussians", "Rendering FPS vs Gaussian count", "Gaussians", "FPS", renders, "gaussian_count", "average_fps", "scatter"),
        ("edit_leakage_by_target", "Edit leakage by target", "Target", "Mean leakage ratio", edits, "target_id", "edit_leakage_ratio", "mean_bar"),
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
    lines = [
        "# ObjSplat Benchmark Report", "",
        "## Run summary", "",
        f"- Scene runs: {robustness.get('scene_runs', 0)}",
        f"- Successful: {robustness.get('successful', 0)}",
        f"- Partial: {robustness.get('partial', 0)}",
        f"- Failed: {robustness.get('failed', 0)}",
        "", "## Statistical summaries", "",
        "```json", json.dumps(summary.get("statistics", {}), indent=2), "```", "",
        "## Scientific interpretation", "",
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
        "Missing optional dependencies and technically unavailable metrics remain blank/null and are not "
        "imputed. Partial and failed runs retain completed stages and identify the failed stage.",
        "", "## Plots", "",
    ]
    for png in sorted((output_dir / "plots").glob("*.png")):
        lines.append(f"![{png.stem}](plots/{png.name})")
        lines.append("")
    report = output_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
