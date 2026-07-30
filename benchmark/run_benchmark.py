#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmark.config import load_config
from benchmark.evaluation import benchmark_rendering, evaluate_reconstruction
from benchmark.editing_evaluation import evaluate_editing
from benchmark.instrumentation import BenchmarkRecorder
from benchmark.input_preparation import file_sha256, prepare_panorama
from benchmark.io_utils import atomic_json
from benchmark.io_utils import write_csv
from benchmark.reporting import aggregate_results
from benchmark.scene_analysis import analyse_existing_scene
from benchmark.system_info import collect_system_info
from benchmark.schemas import EDITING_COLUMNS, RECONSTRUCTION_COLUMNS, RENDERING_COLUMNS


def _run_fingerprint(config: dict, scene: dict) -> str:
    """Hash only inputs that can change the scientific result."""
    ignored = {
        "_config_path",
        "fail_fast",
        "reuse_existing_outputs",
        "reuse_requires_scene_artifacts",
    }
    stable_config = {key: value for key, value in config.items() if key not in ignored}
    panorama_identity = None
    panorama = scene.get("input_panorama")
    if panorama:
        panorama_path = Path(panorama).expanduser().resolve()
        if panorama_path.exists():
            source_stat = panorama_path.stat()
            panorama_identity = {
                "path": str(panorama_path),
                "size_bytes": source_stat.st_size,
                "sha256": file_sha256(panorama_path),
            }
    payload = json.dumps(
        {
            "config": stable_config,
            "scene": scene,
            "input_panorama": panorama_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _find_reusable_run(
    experiment_root: Path, scene_name: str, fingerprint: str
) -> Path | None:
    scene_output = experiment_root / scene_name
    if not scene_output.exists():
        return None
    candidates = sorted(
        scene_output.glob("*/run_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved = json.loads(
                (summary_path.parent / "experiment_config.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError):
            continue
        if (
            summary.get("status") == "success"
            and saved.get("run_fingerprint") == fingerprint
        ):
            return summary_path.parent
    return None


def _has_reusable_scene_artifacts(scene: dict) -> bool:
    root = Path(scene["scene_root"]).expanduser().resolve()
    metadata = root / "traindata" / "layer_instances.json"
    return (
        (root / "rgb.png").exists()
        and metadata.exists()
        and any((root / "traindata").glob("layer*/frames/rgb_*.png"))
        and any((root / "traindata").glob("layer*/pcd_rgb_layer*.ply"))
    )


def _link_shared_preprocessing(scene_root: Path, shared_root: Path) -> None:
    """Reuse immutable segmentation outputs across training-only ablations."""
    shared_root = shared_root.expanduser().resolve()
    required = (
        shared_root / "rgb.png",
        shared_root / "input_preparation.json",
        shared_root / "traindata" / "layer_instances.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Shared preprocessing is incomplete. Run the main six-scene benchmark "
            f"first. Missing: {', '.join(missing)}"
        )
    scene_root.mkdir(parents=True, exist_ok=True)
    for name in ("rgb.png", "input_preparation.json", "traindata"):
        source = shared_root / name
        target = scene_root / name
        if target.exists() or target.is_symlink():
            try:
                if target.samefile(source):
                    continue
            except OSError:
                pass
            raise RuntimeError(
                f"{target} already exists but does not reference the configured "
                f"shared preprocessing source {source}. Use a fresh ablation "
                "scene_root to avoid stale inputs."
            )
        target.symlink_to(source, target_is_directory=source.is_dir())


def _pipeline_command(config: dict, scene: dict, repo: Path) -> list[str]:
    python = str(config.get("python_executable") or sys.executable)
    root = str(Path(scene["scene_root"]).resolve())
    command = [python, str(repo / "run_objsplat_pipeline.py"), "--input_dir", root, "--save_dir", root]
    command.extend(["--seed", str(config.get("random_seed", 42))])
    if config.get("rerun_segmentation"):
        command.append("--force_resegment")
    if not config.get("retrain"):
        command.append("--segment_only")
    split = config.get("splits", {})
    if float(split.get("evaluation_fraction", 0)) > 0:
        command.extend([
            "--benchmark_eval_fraction", str(split["evaluation_fraction"]),
            "--benchmark_split_seed", str(config.get("random_seed", 42)),
        ])
    command.extend(str(value) for value in config.get("pipeline_args", []))
    command.extend(str(value) for value in scene.get("pipeline_args", []))
    return command


def _variants(scene_root: Path) -> dict[str, Path]:
    candidates = {
        "layered": scene_root / "scene" / "gsplat_scene_merged.ply",
        "refined": scene_root / "scene" / "gsplat_scene_merged_refined.ply",
        "night": scene_root / "scene" / "gsplat_scene_night.ply",
        "monolithic": scene_root / "scene" / "gsplat_scene_monolithic.ply",
    }
    return {key: value for key, value in candidates.items() if value.exists()}


def expand_scenes(config: dict) -> list[dict]:
    ablations = config.get("ablations") or []
    if not ablations:
        return list(config["scenes"])
    expanded = []
    for scene in config["scenes"]:
        for ablation in ablations:
            item = dict(scene)
            item.update({
                key: value for key, value in ablation.items()
                if key not in {"pipeline_args", "name", "scene_root_template"}
            })
            base_name = str(scene.get("name") or Path(scene["scene_root"]).name)
            item["name"] = f"{base_name}__{ablation['name']}"
            item["base_scene_name"] = base_name
            if ablation.get("scene_root_template"):
                item["scene_root"] = str(ablation["scene_root_template"]).format(
                    scene=base_name, ablation=ablation["name"]
                )
            shared_template = config.get("shared_preprocessing_root_template")
            if shared_template:
                item["shared_preprocessing_root"] = str(shared_template).format(
                    scene=base_name, ablation=ablation["name"]
                )
            item["pipeline_args"] = (
                list(scene.get("pipeline_args", []))
                + list(ablation.get("pipeline_args", []))
            )
            item["ablation"] = ablation
            expanded.append(item)
    roots = [str(Path(item["scene_root"])) for item in expanded]
    if len(roots) != len(set(roots)):
        raise ValueError(
            "Expanded ablations share scene_root paths. Give every retraining "
            "variant a unique scene_root_template."
        )
    return expanded


def run_scene(
    config: dict,
    scene: dict,
    experiment_root: Path,
    repo: Path,
    *,
    run_fingerprint: str,
) -> dict:
    scene_root = Path(scene["scene_root"]).resolve()
    name = str(scene.get("name") or scene_root.name)
    run_id = uuid.uuid4().hex[:12]
    output = experiment_root / name / run_id
    panorama = scene.get("input_panorama")
    shared_preprocessing = scene.get("shared_preprocessing_root")
    if shared_preprocessing:
        _link_shared_preprocessing(scene_root, Path(shared_preprocessing))
    elif panorama:
        preprocessing = dict(config.get("input_preprocessing") or {})
        preprocessing.update(scene.get("input_preprocessing") or {})
        prepare_panorama(
            panorama,
            scene_root,
            target_width=preprocessing.get("target_width"),
            require_equirectangular_2_to_1=bool(
                preprocessing.get("require_equirectangular_2_to_1", True)
            ),
        )
    recorder = BenchmarkRecorder(
        output, config["experiment_name"], name,
        run_id=run_id,
        sample_interval=float(config["resource_sampling_interval"]),
    )
    context = {"experiment": config["experiment_name"], "scene": name, "run_id": recorder.run_id}
    for filename, columns in (
        ("reconstruction_metrics.csv", RECONSTRUCTION_COLUMNS),
        ("rendering_metrics.csv", RENDERING_COLUMNS),
        ("editing_metrics.csv", EDITING_COLUMNS),
    ):
        write_csv(output / filename, [], columns)
    atomic_json(output / "failures.json", [])
    system = collect_system_info(
        repo, argv=sys.argv, scene_name=name, config_name=config["experiment_name"],
        seed=int(config["random_seed"]), panorama_path=scene.get("input_panorama") or scene_root / "rgb.png",
    )
    atomic_json(output / "system_info.json", system)
    atomic_json(
        output / "experiment_config.json",
        {
            **config,
            "active_scene": scene,
            "run_fingerprint": run_fingerprint,
        },
    )
    failures = []
    status = "success"
    recorder.start()
    try:
        if config.get("rerun_segmentation") or config.get("retrain"):
            command = _pipeline_command(config, scene, repo)
            env = os.environ.copy()
            env.update({
                "OBJSPLAT_BENCHMARK_DIR": str(output),
                "OBJSPLAT_BENCHMARK_EXPERIMENT": config["experiment_name"],
                "OBJSPLAT_BENCHMARK_SCENE": name,
                "OBJSPLAT_BENCHMARK_RUN_ID": recorder.run_id,
                "PYTHONHASHSEED": str(config["random_seed"]),
            })
            with recorder.stage("complete_end_to_end"):
                subprocess.run(command, cwd=repo, env=env, check=True)
        with recorder.stage("existing_scene_analysis"):
            summary = analyse_existing_scene(
                scene_root, output, context,
                scene.get("ground_truth_root") or config.get("ground_truth_root"),
                run_mood_evaluation=bool(config.get("run_mood_evaluation")),
            )
            recorder.summary.update(summary)
        variants = _variants(scene_root)
        if config.get("run_monolithic_baseline"):
            baseline = scene_root / "scene" / "gsplat_scene_monolithic.ply"
            baseline_cfg = config.get("monolithic", {})
            command = [
                str(config.get("python_executable") or sys.executable),
                str(repo / "benchmark" / "monolithic_baseline.py"),
                "--scene_root", str(scene_root), "--output", str(baseline),
                "--iterations", str(baseline_cfg.get("iterations", 1000)),
                "--image_size", str(baseline_cfg.get("image_size", 640)),
                "--rasterizer", str(baseline_cfg.get("rasterizer", "cpp")),
            ]
            if baseline_cfg.get("adaptive_topology"):
                command.append("--adaptive")
            with recorder.stage("monolithic_baseline", iterations=baseline_cfg.get("iterations", 1000)):
                subprocess.run(command, cwd=repo, check=True)
            variants = _variants(scene_root)
        if config.get("run_quality_evaluation"):
            with recorder.stage("reconstruction_evaluation"):
                for variant, ply in variants.items():
                    if variant == "night":
                        continue
                    evaluate_reconstruction(
                        scene_root, ply, output, context, variant,
                        max_side=int(config.get("quality_evaluation", {}).get("max_side", 512)),
                        rasterizer=str(config.get("rendering", {}).get("rasterizer", "cpp")),
                    )
        if config.get("run_rendering_benchmark"):
            render_cfg = config["rendering"]
            render_variants = dict(variants)
            for layer_index in config.get("selected_layers", []):
                layer_path = scene_root / "scene" / f"gsplat_layer{int(layer_index)}.ply"
                if layer_path.exists():
                    render_variants[f"layer_{int(layer_index)}"] = layer_path
            with recorder.stage("rendering_benchmark", frames=render_cfg["measured_frames"]):
                benchmark_rendering(
                    scene_root, render_variants, output, context,
                    width=int(render_cfg["width"]), height=int(render_cfg["height"]),
                    warmup=int(render_cfg["warmup_frames"]), measured=int(render_cfg["measured_frames"]),
                    rasterizer=str(render_cfg.get("rasterizer", "cpp")),
                )
        if config.get("run_edit_locality"):
            render_cfg = config["rendering"]
            with recorder.stage("edit_locality_evaluation"):
                evaluate_editing(
                    scene_root, output, context,
                    [int(x) for x in config.get("selected_layers", [])],
                    [int(x) for x in config.get("selected_instances", [])],
                    width=int(render_cfg["width"]), height=int(render_cfg["height"]),
                    rasterizer=str(render_cfg.get("rasterizer", "cpp")),
                )
    except Exception as exc:
        status = "partial_success" if (output / "layer_metrics.csv").exists() else "failed"
        failures.append({
            "scene": name, "failed_stage": recorder.current_stage,
            "exception_type": type(exc).__name__, "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        })
        atomic_json(output / "failures.json", failures)
        if config.get("fail_fast"):
            raise
    finally:
        recorder.summary["outputs_successfully_generated"] = [
            str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
        ]
        recorder.finalize(status, failures=failures)
    return {"scene": name, "status": status, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible ObjSplat benchmarks")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore matching successful runs and execute every scene again",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    repo = Path(__file__).resolve().parents[1]
    experiment_root = Path(config["output_root"]).resolve() / config["experiment_name"]
    experiment_root.mkdir(parents=True, exist_ok=True)
    atomic_json(experiment_root / "experiment_config.json", config)
    results = []
    expanded_scenes = expand_scenes(config)
    for scene in expanded_scenes:
        name = str(scene.get("name") or Path(scene["scene_root"]).name)
        fingerprint = _run_fingerprint(config, scene)
        reusable = None
        if config.get("reuse_existing_outputs") and not args.force:
            reusable = _find_reusable_run(experiment_root, name, fingerprint)
            if (
                reusable is not None
                and config.get("reuse_requires_scene_artifacts")
                and not _has_reusable_scene_artifacts(scene)
            ):
                print(
                    f"[benchmark] Cached metrics exist for {name}, but reusable "
                    "scene artifacts are missing; rebuilding this scene."
                )
                reusable = None
        if reusable is not None:
            print(f"[benchmark] Reusing successful run for {name}: {reusable}")
            results.append(
                {
                    "scene": name,
                    "status": "success",
                    "output": str(reusable),
                    "reused": True,
                }
            )
            continue
        results.append(
            run_scene(
                config,
                scene,
                experiment_root,
                repo,
                run_fingerprint=fingerprint,
            )
        )
    aggregate_results(experiment_root, experiment_root / "report")
    print(json.dumps(results, indent=2))
    if any(row["status"] == "failed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
