from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS = {
    "output_root": "benchmark_results",
    "reuse_existing_outputs": True,
    "reuse_requires_scene_artifacts": False,
    "rerun_segmentation": False,
    "retrain": False,
    "run_monolithic_baseline": False,
    "run_quality_evaluation": False,
    "run_rendering_benchmark": False,
    "run_edit_locality": False,
    "run_mood_evaluation": True,
    "fail_fast": False,
    "random_seed": 42,
    "resource_sampling_interval": 1.0,
    "rendering": {"width": 512, "height": 512, "warmup_frames": 5, "measured_frames": 30},
    "splits": {"training_fraction": 0.8, "evaluation_fraction": 0.2},
    "pipeline_args": [],
    "input_preprocessing": {
        "target_width": None,
        "require_equirectangular_2_to_1": True,
    },
    "shared_preprocessing_root_template": None,
    "selected_layers": [],
    "selected_instances": [],
    "ablation": {},
    "ablations": [],
}


def _merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configuration requires PyYAML; use JSON or install pyyaml") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Benchmark configuration must be a mapping")
    config = _merge(DEFAULTS, raw)
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict) -> None:
    if not str(config.get("experiment_name", "")).strip():
        raise ValueError("experiment_name is required")
    scenes = config.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a non-empty list")
    names = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"scenes[{index}] must be a mapping")
        root = scene.get("scene_root")
        if not root:
            raise ValueError(f"scenes[{index}].scene_root is required")
        name = scene.get("name") or Path(root).name
        if name in names:
            raise ValueError(f"Duplicate scene name: {name}")
        names.add(name)
    interval = float(config.get("resource_sampling_interval", 1.0))
    if interval <= 0:
        raise ValueError("resource_sampling_interval must be > 0")
    target_width = (config.get("input_preprocessing") or {}).get("target_width")
    if target_width is not None and (
        int(target_width) < 512 or int(target_width) % 2
    ):
        raise ValueError("input_preprocessing.target_width must be an even integer >= 512")
    splits = config.get("splits", {})
    train = float(splits.get("training_fraction", 0.8))
    evaluation = float(splits.get("evaluation_fraction", 0.2))
    if train <= 0 or evaluation < 0 or abs(train + evaluation - 1.0) > 1e-6:
        raise ValueError("training/evaluation fractions must be non-negative and sum to 1")
    valid_layer_selectors = {"smallest", "median", "largest", "all"}
    for selector in config.get("selected_layers", []):
        if isinstance(selector, int) or str(selector).lstrip("-").isdigit():
            continue
        if str(selector).strip().lower() not in valid_layer_selectors:
            raise ValueError(
                "selected_layers entries must be integers or one of "
                "smallest, median, largest, all"
            )
    for index, ablation in enumerate(config.get("ablations") or []):
        if not isinstance(ablation, dict) or not ablation.get("name"):
            raise ValueError(f"ablations[{index}] requires a name")
        if not isinstance(ablation.get("pipeline_args", []), list):
            raise ValueError(f"ablations[{index}].pipeline_args must be a list")
        template = ablation.get("scene_root_template")
        if template and "{scene}" not in str(template):
            raise ValueError(
                f"ablations[{index}].scene_root_template must contain {{scene}} "
                "to keep scene artifacts separate"
            )
    shared_template = config.get("shared_preprocessing_root_template")
    if shared_template:
        if "{scene}" not in str(shared_template):
            raise ValueError(
                "shared_preprocessing_root_template must contain {scene}"
            )
        if config.get("rerun_segmentation"):
            raise ValueError(
                "rerun_segmentation must be false when shared preprocessing is enabled"
            )
