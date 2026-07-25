"""Preflight validation for generated ObjSplat layer data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_sky(metadata: dict) -> dict | None:
    sky = metadata.get("sky")
    if isinstance(sky, dict) and sky.get("layer_idx") is not None:
        return sky
    for group in metadata.get("layer_groups", []):
        if str(group.get("group_label", "")).strip().lower() == "sky":
            idx = int(group["layer_idx"])
            return {
                "layer_idx": idx,
                "mask_path": f"traindata/layer{idx}/layer{idx}_erp_mask.png",
            }
    return None


def validate_layer_data(
    scene_root: str | Path,
    metadata_path: str | Path | None = None,
    require_sky: bool = False,
    min_sky_coverage: float = 0.005,
) -> dict:
    scene_root = Path(scene_root).expanduser().resolve()
    metadata_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path
        else scene_root / "traindata" / "layer_instances.json"
    )
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict = {}

    if not metadata_path.exists():
        errors.append(f"Missing layer metadata: {metadata_path}")
        return {"ok": False, "errors": errors, "warnings": warnings, "metrics": metrics}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rgb_path = scene_root / "rgb.png"
    if not rgb_path.exists() and metadata.get("input_dir"):
        rgb_path = Path(str(metadata["input_dir"])).expanduser() / "rgb.png"
    if not rgb_path.exists():
        errors.append(f"Missing source ERP: {rgb_path}")
        image_size = None
    else:
        with Image.open(rgb_path) as image:
            image_size = image.size
        metrics["erp_size"] = list(image_size)
        ratio = float(image_size[0]) / max(1, image_size[1])
        metrics["erp_aspect_ratio"] = ratio
        if abs(ratio - 2.0) > 0.03:
            errors.append(f"Source image is not a 2:1 ERP (aspect ratio {ratio:.4f})")

    layer_indices = {
        int(group["layer_idx"])
        for group in metadata.get("layer_groups", [])
        if group.get("layer_idx") is not None
    }
    for key in ("background_layer_idx", "residual_layer_idx"):
        if metadata.get(key) is not None:
            layer_indices.add(int(metadata[key]))

    frame_counts = {}
    frame_mask_counts = {}
    for layer_idx in sorted(layer_indices):
        layer_dir = scene_root / "traindata" / f"layer{layer_idx}"
        pcd_path = layer_dir / f"pcd_rgb_layer{layer_idx}.ply"
        frames = sorted((layer_dir / "frames").glob("rgb_*.png"))
        masks = sorted((layer_dir / "frames").glob("mask_*.png"))
        frame_counts[str(layer_idx)] = len(frames)
        frame_mask_counts[str(layer_idx)] = len(masks)
        if not pcd_path.exists() or pcd_path.stat().st_size <= 0:
            errors.append(f"Layer {layer_idx} has no valid RGB point cloud")
        if not frames:
            errors.append(f"Layer {layer_idx} has no training frames")
        elif len(masks) != len(frames):
            errors.append(
                f"Layer {layer_idx} has {len(frames)} RGB frames but {len(masks)} "
                "supervision masks; black-filled targets would create layer seams"
            )
    metrics["layer_frame_counts"] = frame_counts
    metrics["layer_frame_mask_counts"] = frame_mask_counts

    sky = _resolve_sky(metadata)
    if sky is None:
        if require_sky:
            errors.append("No dedicated sky layer in metadata")
    else:
        layer_idx = int(sky["layer_idx"])
        mask_path = scene_root / str(
            sky.get("mask_path", f"traindata/layer{layer_idx}/layer{layer_idx}_erp_mask.png")
        )
        if not mask_path.exists():
            errors.append(f"Missing sky mask: {mask_path}")
        else:
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128
            if image_size and mask.shape != (image_size[1], image_size[0]):
                errors.append(
                    f"Sky mask shape {mask.shape} does not match ERP "
                    f"{(image_size[1], image_size[0])}"
                )
            coverage = float(mask.mean())
            metrics["sky_coverage"] = coverage
            metrics["sky_zenith_coverage"] = float(mask[: max(1, mask.shape[0] // 50)].mean())
            if require_sky and coverage < float(min_sky_coverage):
                errors.append(
                    f"Sky coverage {coverage:.4f} is below {min_sky_coverage:.4f}"
                )
            if coverage > 0.75:
                warnings.append(f"Sky coverage is unusually high ({coverage:.4f})")

    sky_diagnostics = (
        (metadata.get("sky_segmentation") or {}).get("segformer") or {}
    )
    raw_sky_coverage = sky_diagnostics.get("coverage")
    protected_sky_coverage = sky_diagnostics.get("coverage_after_sam_protection")
    if raw_sky_coverage is not None and protected_sky_coverage is not None:
        retained = float(protected_sky_coverage) / max(
            float(raw_sky_coverage), 1e-8
        )
        metrics["sky_fraction_retained_after_sam_protection"] = retained
        if retained < 0.70:
            errors.append(
                "SAM foreground protection removed more than 30% of the semantic "
                f"sky ({retained:.1%} retained); likely a broad tree/leaves mask"
            )

    coverage_3d = metadata.get("coverage_3d")
    if coverage_3d is not None:
        metrics["detected_coverage_3d"] = float(coverage_3d)
        if (
            float(coverage_3d) >= 0.999
            and bool(metadata.get("fill_unassigned_layers", True))
        ):
            warnings.append(
                "Detected layers cover virtually 100% of the scene using legacy "
                "unassigned-pixel filling; semantic labels may be inflated"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def assert_valid_layer_data(*args, **kwargs) -> dict:
    report = validate_layer_data(*args, **kwargs)
    for warning in report["warnings"]:
        print(f"[Preflight] warning: {warning}")
    if not report["ok"]:
        details = "\n".join(f"- {item}" for item in report["errors"])
        raise RuntimeError(f"Layer-data preflight failed:\n{details}")
    print("[Preflight] Layer data validation passed")
    return report
