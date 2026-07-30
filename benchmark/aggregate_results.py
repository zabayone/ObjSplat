#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.reporting import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ObjSplat benchmark runs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate_results(args.input, args.output)
    status = result.get("robustness", {})
    print(
        "Aggregated "
        f"{status.get('scene_runs', 0)} scene runs: "
        f"{status.get('successful', 0)} successful, "
        f"{status.get('partial', 0)} partial, "
        f"{status.get('failed', 0)} failed."
    )
    print(f"Scientific report: {Path(args.output).resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
