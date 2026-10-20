#!/usr/bin/env python3
"""Populate missing reconstruction LPIPS values from saved benchmark renders."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from benchmark.io_utils import read_csv, write_csv
from benchmark.metrics.image_metrics import lpips_score
from benchmark.schemas import RECONSTRUCTION_COLUMNS


def _image_pair(run_dir: Path, row: dict[str, str]) -> tuple[Path, Path]:
    variant = str(row.get("variant") or "")
    view_id = int(row["view_id"])
    image_dir = run_dir / "images" / "reconstruction" / variant
    stem = f"view_{view_id:04d}"
    return (
        image_dir / f"{stem}_ground_truth.png",
        image_dir / f"{stem}_rendered.png",
    )


def backfill_run(csv_path: Path, *, overwrite: bool = False) -> tuple[int, int]:
    rows = read_csv(csv_path)
    updated = unavailable = 0
    for row in rows:
        if row.get("status") != "success" or not row.get("view_id"):
            continue
        if row.get("lpips") and not overwrite:
            continue
        reference_path, rendered_path = _image_pair(csv_path.parent, row)
        if not reference_path.exists() or not rendered_path.exists():
            unavailable += 1
            continue
        with Image.open(reference_path) as handle:
            reference = np.asarray(handle.convert("RGB"))
        with Image.open(rendered_path) as handle:
            rendered = np.asarray(handle.convert("RGB"))
        value = lpips_score(reference, rendered)
        if value is None:
            raise RuntimeError(
                "LPIPS is unavailable. Install it in the benchmark Python environment."
            )
        row["lpips"] = value
        updated += 1
    if updated:
        write_csv(csv_path, rows, RECONSTRUCTION_COLUMNS)
        config_path = csv_path.parent / "experiment_config.json"
        if config_path.exists():
            from benchmark.comparison import write_comparison_metrics
            config = json.loads(config_path.read_text())
            scene = config.get("active_scene", {})
            if scene.get("scene_root"):
                context = {k: rows[0].get(k) for k in ("experiment", "scene", "run_id")}
                write_comparison_metrics(scene["scene_root"], csv_path.parent, context, config, scene)
    return updated, unavailable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill LPIPS from existing ground-truth/rendered PNG pairs"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory containing <scene>/<run>/reconstruction_metrics.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute LPIPS values that are already populated",
    )
    args = parser.parse_args()

    experiment = Path(args.experiment).expanduser().resolve()
    csv_paths = sorted(experiment.glob("*/*/reconstruction_metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No reconstruction_metrics.csv files found below {experiment}"
        )

    total_updated = total_unavailable = 0
    for csv_path in csv_paths:
        updated, unavailable = backfill_run(
            csv_path, overwrite=bool(args.overwrite)
        )
        total_updated += updated
        total_unavailable += unavailable
        print(
            f"{csv_path.parent.relative_to(experiment)}: "
            f"{updated} updated, {unavailable} missing image pairs",
            flush=True,
        )
    print(
        f"LPIPS backfill complete: {total_updated} updated, "
        f"{total_unavailable} unavailable.",
        flush=True,
    )


if __name__ == "__main__":
    main()
