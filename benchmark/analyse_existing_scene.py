#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.instrumentation import BenchmarkRecorder
from benchmark.io_utils import atomic_json, write_csv
from benchmark.reporting import aggregate_results
from benchmark.scene_analysis import analyse_existing_scene
from benchmark.system_info import collect_system_info
from benchmark.schemas import EDITING_COLUMNS, RECONSTRUCTION_COLUMNS, RENDERING_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse completed ObjSplat outputs without retraining")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--experiment", default="existing_scene_smoke")
    parser.add_argument("--ground_truth_root", default=None)
    parser.add_argument("--sample_interval", type=float, default=1.0)
    args = parser.parse_args()
    scene = Path(args.scene_root).resolve()
    output = Path(args.output or f"benchmark_results/{args.experiment}/{scene.name}").resolve()
    context = {"experiment": args.experiment, "scene": scene.name, "run_id": None}
    recorder = BenchmarkRecorder(output, args.experiment, scene.name, sample_interval=args.sample_interval)
    context["run_id"] = recorder.run_id
    for filename, columns in (
        ("reconstruction_metrics.csv", RECONSTRUCTION_COLUMNS),
        ("rendering_metrics.csv", RENDERING_COLUMNS),
        ("editing_metrics.csv", EDITING_COLUMNS),
    ):
        write_csv(output / filename, [], columns)
    atomic_json(output / "failures.json", [])
    atomic_json(output / "system_info.json", collect_system_info(
        Path(__file__).resolve().parents[1], argv=sys.argv, scene_name=scene.name,
        config_name=args.experiment, panorama_path=scene / "rgb.png",
    ))
    with recorder:
        with recorder.stage("existing_scene_analysis"):
            summary = analyse_existing_scene(scene, output, context, args.ground_truth_root)
        recorder.summary.update(summary)
    aggregate_results(output.parent, output.parent / "report")
    print(json.dumps({"output": str(output), "status": "success"}, indent=2))


if __name__ == "__main__":
    main()
