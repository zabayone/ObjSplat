#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    print(json.dumps(aggregate_results(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
