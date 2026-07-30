from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.io_utils import write_csv
from benchmark.metrics.editing import edit_locality_metrics
from benchmark.metrics.image_metrics import absolute_error_visualization
from benchmark.ply_utils import filter_ply_by_label, inspect_ply
from benchmark.rendering import MLXSceneRenderer
from benchmark.schemas import EDITING_COLUMNS


def _first_target_frame(scene_root: Path, layer_index: int):
    frames = scene_root / "traindata" / f"layer{layer_index}" / "frames"
    for mask_path in sorted(frames.glob("mask_*.png")):
        index = int(mask_path.stem.rsplit("_", 1)[1])
        pose_path = frames / f"transform_matrix_{index}.npy"
        if pose_path.exists():
            return mask_path, pose_path
    return None, None


def evaluate_editing(
    scene_root: str | Path, output_dir: str | Path, context: dict,
    selected_layers: list[int], selected_instances: list[int],
    width: int = 512, height: int = 512, rasterizer: str = "cpp",
) -> list[dict]:
    from mps_splat_backend import merge_ply_layers

    scene_root, output_dir = Path(scene_root), Path(output_dir)
    metadata = json.loads((scene_root / "traindata" / "layer_instances.json").read_text())
    groups = {int(g["layer_idx"]): g for g in metadata.get("layer_groups", [])}
    instance_to_layer = {int(i["instance_id"]): int(i["layer_idx"]) for i in metadata.get("instances", [])}
    object_layers = sorted(groups)
    if not selected_layers and not selected_instances and object_layers:
        selected_layers = [object_layers[0]]
    source_candidates = [
        scene_root / "scene" / "gsplat_scene_merged_refined.ply",
        scene_root / "scene" / "gsplat_scene_merged.ply",
    ]
    source = next((p for p in source_candidates if p.exists()), None)
    if source is None:
        return [{**context, "status": "unavailable", "reason": "No merged scene PLY"}]
    original_renderer = MLXSceneRenderer(source, rasterizer)
    image_dir = output_dir / "images" / "editing"
    representation_dir = output_dir / "edited_representations"
    image_dir.mkdir(parents=True, exist_ok=True)
    representation_dir.mkdir(parents=True, exist_ok=True)
    targets = [("layer", int(value), int(value)) for value in selected_layers]
    targets += [("instance", int(value), instance_to_layer.get(int(value), -1)) for value in selected_instances]
    rows = []
    for target_type, target_id, layer_index in targets:
        mask_path, pose_path = _first_target_frame(scene_root, layer_index)
        if mask_path is None:
            rows.append({**context, "variant": "layered", "target_type": target_type,
                         "target_id": target_id, "status": "unavailable",
                         "reason": "No projected target mask/pose"})
            continue
        edited = representation_dir / f"without_{target_type}_{target_id}.ply"
        started = time.perf_counter()
        try:
            if target_type == "instance":
                edit_report = filter_ply_by_label(source, edited, {target_id})
            else:
                layer_plys = sorted((scene_root / "scene").glob("gsplat_layer*.ply"))
                retained = [p for p in layer_plys if p.name != f"gsplat_layer{target_id}.ply"]
                if not retained:
                    raise RuntimeError("Layer removal would leave no layers")
                merge_ply_layers(retained, str(edited), voxel_size=0, min_opacity=-20, max_points=None)
                source_count = inspect_ply(source)["vertex_count"]
                edited_count = inspect_ply(edited)["vertex_count"]
                edit_report = {
                    "source_gaussians": source_count, "edited_gaussians": edited_count,
                    "removed_gaussians": source_count - edited_count,
                }
            creation = time.perf_counter() - started
            pose = np.load(pose_path)
            before = original_renderer.render(pose, width, height)
            after = MLXSceneRenderer(edited, rasterizer).render(pose, width, height)
            mask = np.asarray(Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)) >= 128
            metrics = edit_locality_metrics(before, after, mask)
            stem = f"{target_type}_{target_id}"
            Image.fromarray(before).save(image_dir / f"{stem}_before.png")
            Image.fromarray(after).save(image_dir / f"{stem}_after.png")
            Image.fromarray(absolute_error_visualization(before, after)).save(image_dir / f"{stem}_difference.png")
            removed = int(edit_report["removed_gaussians"])
            source_count = int(edit_report["source_gaussians"])
            rows.append({
                **context, "variant": "layered", "target_type": target_type,
                "target_id": target_id, **metrics, "outside_lpips": None,
                "removed_gaussians": removed,
                "removed_gaussians_percent": removed / max(1, source_count) * 100,
                "creation_seconds": creation, "retraining_required": False,
                "edited_size_bytes": edited.stat().st_size, "status": "success",
            })
        except Exception as exc:
            rows.append({
                **context, "variant": "layered", "target_type": target_type,
                "target_id": target_id, "creation_seconds": time.perf_counter() - started,
                "retraining_required": False, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    write_csv(output_dir / "editing_metrics.csv", rows, EDITING_COLUMNS)
    return rows
