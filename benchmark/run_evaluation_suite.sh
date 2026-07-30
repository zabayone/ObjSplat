#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/anaconda3/envs/layerpano3d/bin/python}"

cd "$ROOT"

"$PY" benchmark/run_benchmark.py \
  --config benchmark/configs/evaluation_six_scenes.yaml

"$PY" benchmark/run_benchmark.py \
  --config benchmark/configs/evaluation_focused_ablations.yaml
