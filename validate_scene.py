#!/usr/bin/env python3
"""Validate ObjSplat layer data before diffusion or 3DGS training."""

from __future__ import annotations

import argparse
import json

from utils.pipeline_validation import validate_layer_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--require_sky", action="store_true")
    parser.add_argument("--min_sky_coverage", type=float, default=0.005)
    args = parser.parse_args()
    report = validate_layer_data(
        args.scene_root,
        metadata_path=args.metadata_path,
        require_sky=args.require_sky,
        min_sky_coverage=args.min_sky_coverage,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
