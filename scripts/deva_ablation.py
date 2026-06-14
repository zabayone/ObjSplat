#!/usr/bin/env python3
"""Run DEVA mask ablations and generate a simple HTML comparison report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gen_layerdata_from_deva import generate_layers_from_deva


def _default_variants() -> List[Dict]:
    return [
        {
            "name": "baseline",
            "n_views": 8,
            "phi_bands": [45, 0, -45],
            "min_frame_area": 2000,
            "min_frames": 3,
            "min_total_pixels": 10000,
            "min_points_3d": 5000,
            "sam_pred_iou_threshold": 0.92,
            "sam_stability_score_threshold": 0.95,
            "mask_min_area": 1000,
            "detection_every": 5,
        },
        {
            "name": "more_views",
            "n_views": 12,
            "phi_bands": [45, 0, -45],
            "min_frame_area": 2000,
            "min_frames": 3,
            "min_total_pixels": 10000,
            "min_points_3d": 5000,
            "sam_pred_iou_threshold": 0.92,
            "sam_stability_score_threshold": 0.95,
            "mask_min_area": 1000,
            "detection_every": 5,
        },
        {
            "name": "more_phi",
            "n_views": 8,
            "phi_bands": [60, 30, 0, -30, -60],
            "min_frame_area": 2000,
            "min_frames": 3,
            "min_total_pixels": 10000,
            "min_points_3d": 5000,
            "sam_pred_iou_threshold": 0.92,
            "sam_stability_score_threshold": 0.95,
            "mask_min_area": 1000,
            "detection_every": 5,
        },
        {
            "name": "looser_sam",
            "n_views": 8,
            "phi_bands": [45, 0, -45],
            "min_frame_area": 1500,
            "min_frames": 3,
            "min_total_pixels": 8000,
            "min_points_3d": 4000,
            "sam_pred_iou_threshold": 0.88,
            "sam_stability_score_threshold": 0.92,
            "mask_min_area": 800,
            "detection_every": 4,
        },
        {
            "name": "stricter_filter",
            "n_views": 8,
            "phi_bands": [45, 0, -45],
            "min_frame_area": 2500,
            "min_frames": 4,
            "min_total_pixels": 12000,
            "min_points_3d": 7000,
            "sam_pred_iou_threshold": 0.92,
            "sam_stability_score_threshold": 0.96,
            "mask_min_area": 1200,
            "detection_every": 5,
        },
        {
            "name": "stricter_filter_plus",
            "n_views": 12,
            "phi_bands": [60, 30, 0, -30, -60],
            "min_frame_area": 2500,
            "min_frames": 4,
            "min_total_pixels": 12000,
            "min_points_3d": 7000,
            "sam_pred_iou_threshold": 0.90,
            "sam_stability_score_threshold": 0.93,
            "mask_min_area": 900,
            "detection_every": 5,
        },
    ]


def _build_overlay(input_rgb: Path, layer_masks: List[Path], out_path: Path) -> None:
    base = np.array(Image.open(input_rgb).convert("RGB"), dtype=np.uint8)
    overlay = base.astype(np.float32)
    if not layer_masks:
        Image.fromarray(base).save(out_path)
        return

    h, w = base.shape[:2]
    colors = np.array(
        [
            [255, 99, 71],
            [30, 144, 255],
            [60, 179, 113],
            [238, 130, 238],
            [255, 215, 0],
            [70, 130, 180],
            [255, 105, 180],
            [46, 139, 87],
            [255, 140, 0],
            [123, 104, 238],
        ],
        dtype=np.float32,
    )
    alpha = 0.45

    for idx, mask_path in enumerate(layer_masks):
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if mask.shape != (h, w):
            mask = np.array(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
        mask = mask > 0
        if not mask.any():
            continue
        color = colors[idx % len(colors)]
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def _write_report(output_dir: Path, results: List[Dict]) -> None:
    rows = []
    for res in results:
        name = res["name"]
        save_dir = Path(res["save_dir"])
        overlay_path = save_dir / "erp_overlay.png"
        if overlay_path.exists():
            link = f"<a href='{overlay_path.relative_to(output_dir)}'>overlay</a>"
        else:
            link = "(missing overlay)"
        rows.append(f"<tr><td>{name}</td><td>{link}</td></tr>")

        rows_html = "\n".join(rows)
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DEVA Ablation Report</title>
  <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
        th {{ background: #f5f5f5; }}
        td div {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>DEVA Ablation Report</h1>
    <p>Generated at: {datetime.now().isoformat(timespec="seconds")}</p>
  <table>
    <thead>
    <tr><th>Variant</th><th>ERP Overlay</th></tr>
    </thead>
    <tbody>
            {rows_html}
    </tbody>
  </table>
</body>
</html>
"""

    report_path = output_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEVA ablations and write a comparison report")
    parser.add_argument("--input_dir", required=True, help="Input directory (e.g. outputs_lgs)")
    parser.add_argument("--output_dir", required=True, help="Output directory (e.g. outputs_lgs/ablation)")
    parser.add_argument("--variants_json", default=None, help="Optional JSON list of variants")
    parser.add_argument("--variant", default=None, help="Run only a single variant by name")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variants_json:
        variants = json.loads(Path(args.variants_json).read_text(encoding="utf-8"))
    else:
        variants = _default_variants()

    if args.variant:
        variants = [v for v in variants if v.get("name") == args.variant]
        if not variants:
            raise SystemExit(f"Unknown variant: {args.variant}")

    results = []
    for variant in variants:
        name = variant.get("name", "variant")
        save_dir = output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = generate_layers_from_deva(
            input_dir=str(input_dir),
            save_dir=str(save_dir),
            min_frame_area=int(variant.get("min_frame_area", 2000)),
            min_frames=int(variant.get("min_frames", 3)),
            min_total_pixels=int(variant.get("min_total_pixels", 10000)),
            min_points_3d=int(variant.get("min_points_3d", 5000)),
            n_views=int(variant.get("n_views", 8)),
            phi_bands=variant.get("phi_bands", [45, 0, -45]),
            sam_pred_iou_threshold=variant.get("sam_pred_iou_threshold"),
            sam_stability_score_threshold=variant.get("sam_stability_score_threshold"),
            mask_min_area=variant.get("mask_min_area"),
            detection_every=variant.get("detection_every"),
            max_num_objects=variant.get("max_num_objects"),
        )

        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        instances = meta.get("instances", [])
        background_idx = meta.get("background_layer_idx")
        layer_ids = [item.get("layer_idx") for item in instances]
        if background_idx is not None:
            layer_ids.append(background_idx)

        layer_masks = []
        for layer_idx in layer_ids:
            if layer_idx is None:
                continue
            mask_path = save_dir / "traindata" / f"layer{layer_idx}" / f"layer{layer_idx}_erp_mask.png"
            if mask_path.exists():
                layer_masks.append(mask_path)

        input_rgb = input_dir / "rgb.png"
        overlay_path = save_dir / "erp_overlay.png"
        if input_rgb.exists():
            _build_overlay(input_rgb, layer_masks, overlay_path)
        results.append({"name": name, "save_dir": str(save_dir), "metadata": meta})

    _write_report(output_dir, results)
    print(f"Report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
