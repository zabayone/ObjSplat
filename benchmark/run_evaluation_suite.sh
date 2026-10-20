#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/anaconda3/envs/layerpano3d/bin/python}"

cd "$ROOT"

"$PY" benchmark/run_benchmark.py \
  --config benchmark/configs/layered_vs_monolithic_six_scenes.yaml

"$PY" benchmark/run_benchmark.py \
  --config benchmark/configs/quality_scaling.yaml
