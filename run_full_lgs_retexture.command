#!/bin/zsh
set -euo pipefail

cd "/Users/nicola/Documents/GitHub/ObjSplat"

export PYTHONUNBUFFERED=1
export OUT_DIR="outputs_lgs"
export CLEAN_OUTPUT=1
export FORCE_RESEGMENT=1
export SEGMENT_ONLY=0
export MOOD_ONLY=0
export RETEXTURE_NIGHT_SKY=1
export BUILD_NIGHT_MOOD=1
export MAX_POINTS=0
export ADAPTIVE_TOPOLOGY=0

echo "Starting full ObjSplat training with day/night retexturing"
echo "Scene: $OUT_DIR"
echo "The process is attached to this terminal and protected by caffeinate."
echo

exec caffeinate -dimsu ./run_from_pano.sh
