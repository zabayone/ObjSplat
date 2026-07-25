#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PY:-/opt/anaconda3/envs/layerpano3d/bin/python}"
OUT_DIR="${OUT_DIR:-outputs_park}"
RGB_PATH="$OUT_DIR/rgb.png"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -f "$RGB_PATH" ]]; then
  echo "[ERROR] Missing $RGB_PATH"
  echo "Place the input panorama at $RGB_PATH, then rerun this script."
  exit 1
fi

CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
if [[ "$CLEAN_OUTPUT" != "0" ]]; then
  echo "[1/3] Cleaning $OUT_DIR while preserving rgb.png"
  shopt -s dotglob nullglob
  for path in "$OUT_DIR"/*; do
    if [[ "$(basename "$path")" == "rgb.png" ]]; then
      continue
    fi
    rm -rf -- "$path"
  done
  shopt -u dotglob nullglob
else
  echo "[1/3] Reusing existing $OUT_DIR artifacts"
fi

echo "[2/3] Generating Grounding-SAM object-aware traindata"

DEPTH_SCALE="${DEPTH_SCALE:-200}"
DEVICE="${DEVICE:-mps}"

N_VIEWS="${N_VIEWS:-12}"
PHI_BANDS="${PHI_BANDS:-80,67.5,45,0,-45,-67.5,-80}"
PERSPECTIVE_SIZE="${PERSPECTIVE_SIZE:-1024}"

# GROUNDING_PROMPTS="${GROUNDING_PROMPTS:-sky . road . pavement . grass . leaves . tree . bush . person . plant}"
# GROUNDING_PROMPTS="${GROUNDING_PROMPTS:-sky . mountain . person . table . bench . building . railing . house . snow . rock}"
GROUNDING_PROMPTS="${GROUNDING_PROMPTS:-sky . road . pavement . grass . leaves . tree . bush . plant . stairway . fence . monument . lightpole}"
GROUNDING_BOX_THRESHOLD="${GROUNDING_BOX_THRESHOLD:-0.18}"
GROUNDING_TEXT_THRESHOLD="${GROUNDING_TEXT_THRESHOLD:-0.15}"
GROUNDING_MASK_MIN_AREA="${GROUNDING_MASK_MIN_AREA:-500}"
GROUNDING_INFER_MAX_SIDE="${GROUNDING_INFER_MAX_SIDE:-1536}"
GROUNDING_BOX_PADDING="${GROUNDING_BOX_PADDING:-0.00}"
GROUNDING_MORPH_OPEN_KERNEL="${GROUNDING_MORPH_OPEN_KERNEL:-9}"
GROUNDING_MIN_COMPONENT_AREA_RATIO="${GROUNDING_MIN_COMPONENT_AREA_RATIO:-0.08}"
SKY_SEGMENTATION_BACKEND="${SKY_SEGMENTATION_BACKEND:-hybrid}"
SKY_SEGFORMER_MODEL="${SKY_SEGFORMER_MODEL:-nvidia/segformer-b2-finetuned-ade-512-512}"
SKY_SEGFORMER_MAX_SIDE="${SKY_SEGFORMER_MAX_SIDE:-2048}"
SKY_SEGFORMER_THRESHOLD="${SKY_SEGFORMER_THRESHOLD:-0.45}"

MIN_FRAME_AREA="${MIN_FRAME_AREA:-1500}"
MIN_FRAMES="${MIN_FRAMES:-3}"
MIN_TOTAL_PIXELS="${MIN_TOTAL_PIXELS:-10000}"
MIN_POINTS_3D="${MIN_POINTS_3D:-1000}"

EQUIRECT_MIN_VOTES="${EQUIRECT_MIN_VOTES:-1}"
EQUIRECT_KERNEL_SIZE="${EQUIRECT_KERNEL_SIZE:-15}"

SAM_VARIANT="${SAM_VARIANT:-sam2}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-checkpoints/SAM 2.1 Hiera Large.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
GROUNDING_DINO_CHECKPOINT="${GROUNDING_DINO_CHECKPOINT:-IDEA-Research/grounding-dino-base}"
FORCE_RESEGMENT="${FORCE_RESEGMENT:-0}"
SEGMENT_ONLY="${SEGMENT_ONLY:-0}"
MOOD_ONLY="${MOOD_ONLY:-0}"
SEGMENT_ARGS=()
if [[ "$FORCE_RESEGMENT" != "0" && "$MOOD_ONLY" == "0" ]]; then
  SEGMENT_ARGS+=(--force_resegment)
fi
if [[ "$SEGMENT_ONLY" != "0" ]]; then
  SEGMENT_ARGS+=(--segment_only)
fi
if [[ "$MOOD_ONLY" != "0" ]]; then
  SEGMENT_ARGS+=(--mood_only)
fi

RETEXTURE_NIGHT_SKY="${RETEXTURE_NIGHT_SKY:-0}"
BUILD_NIGHT_MOOD="${BUILD_NIGHT_MOOD:-$RETEXTURE_NIGHT_SKY}"
SKY_MODEL_PATH="${SKY_MODEL_PATH:-checkpoints/FLUX.1-Fill-dev}"
SKY_STEPS="${SKY_STEPS:-50}"
SKY_SEED="${SKY_SEED:-42}"
SKY_MAX_PIXELS="${SKY_MAX_PIXELS:-1048576}"
NIGHT_MOOD_REFINE_ITERS="${NIGHT_MOOD_REFINE_ITERS:-0}"
NIGHT_TRAINING_IMAGE_SIZE="${NIGHT_TRAINING_IMAGE_SIZE:-512}"
MOOD_ARGS=()
if [[ "$RETEXTURE_NIGHT_SKY" != "0" ]]; then
  MOOD_ARGS+=(
    --retexture_night_sky
    --sky_model_path "$SKY_MODEL_PATH"
    --sky_steps "$SKY_STEPS"
    --sky_seed "$SKY_SEED"
    --sky_max_pixels "$SKY_MAX_PIXELS"
  )
fi
if [[ "$BUILD_NIGHT_MOOD" != "0" ]]; then
  MOOD_ARGS+=(
    --build_night_mood
    --night_mood_refine_iters "$NIGHT_MOOD_REFINE_ITERS"
    --night_training_image_size "$NIGHT_TRAINING_IMAGE_SIZE"
  )
fi

echo "[3/3] Training and merging standard-quality 3DGS scene"

QUALITY="${QUALITY:-standard}"
MPS_RASTERIZER="${MPS_RASTERIZER:-cpp}"

MAX_POINTS="${MAX_POINTS:-0}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-1.0}"
TRAINING_IMAGE_SIZE="${TRAINING_IMAGE_SIZE:-640}"
ADAPTIVE_TOPOLOGY="${ADAPTIVE_TOPOLOGY:-0}"
TOPOLOGY_ARGS=()
if [[ "$ADAPTIVE_TOPOLOGY" != "0" ]]; then
  TOPOLOGY_ARGS+=(--adaptive_topology)
fi
MEAN_LR_SCALE="${MEAN_LR_SCALE:-0.35}"
REPULSION_WEIGHT="${REPULSION_WEIGHT:-5e-5}"

MERGE_VOXEL_SIZE="${MERGE_VOXEL_SIZE:-0}"
MERGE_MIN_OPACITY="${MERGE_MIN_OPACITY:--20}"
MERGE_MAX_POINTS="${MERGE_MAX_POINTS:-0}"

GLOBAL_REFINE_ITERS="${GLOBAL_REFINE_ITERS:-0}"
LAYER_ITERATIONS="${LAYER_ITERATIONS:-800}"
BACKGROUND_ITERATIONS="${BACKGROUND_ITERATIONS:-1000}"
SKY_ITERATIONS="${SKY_ITERATIONS:-500}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
LR_PLATEAU_PATIENCE="${LR_PLATEAU_PATIENCE:-0}"
LR_PLATEAU_FACTOR="${LR_PLATEAU_FACTOR:-0.5}"
LR_PLATEAU_MIN_LR="${LR_PLATEAU_MIN_LR:-1e-6}"

"$PY" run_objsplat_pipeline.py \
  --input_dir "$OUT_DIR" \
  --save_dir "$OUT_DIR" \
  ${SEGMENT_ARGS[@]+"${SEGMENT_ARGS[@]}"} \
  ${MOOD_ARGS[@]+"${MOOD_ARGS[@]}"} \
  --device "$DEVICE" \
  --depth_scale "$DEPTH_SCALE" \
  --min_frame_area "$MIN_FRAME_AREA" \
  --min_frames "$MIN_FRAMES" \
  --min_total_pixels "$MIN_TOTAL_PIXELS" \
  --min_points_3d "$MIN_POINTS_3D" \
  --n_views "$N_VIEWS" \
  --phi_bands "$PHI_BANDS" \
  --perspective_size "$PERSPECTIVE_SIZE" \
  --equirect_min_votes "$EQUIRECT_MIN_VOTES" \
  --equirect_kernel_size "$EQUIRECT_KERNEL_SIZE" \
  --use_grounding_dino \
  --grounding_dino_checkpoint "$GROUNDING_DINO_CHECKPOINT" \
  --grounding_prompts "$GROUNDING_PROMPTS" \
  --grounding_box_threshold "$GROUNDING_BOX_THRESHOLD" \
  --grounding_text_threshold "$GROUNDING_TEXT_THRESHOLD" \
  --grounding_mask_min_area "$GROUNDING_MASK_MIN_AREA" \
  --grounding_infer_max_side "$GROUNDING_INFER_MAX_SIDE" \
  --grounding_box_padding "$GROUNDING_BOX_PADDING" \
  --grounding_morph_open_kernel "$GROUNDING_MORPH_OPEN_KERNEL" \
  --grounding_min_component_area_ratio "$GROUNDING_MIN_COMPONENT_AREA_RATIO" \
  --sky_segmentation_backend "$SKY_SEGMENTATION_BACKEND" \
  --sky_segformer_model "$SKY_SEGFORMER_MODEL" \
  --sky_segformer_max_side "$SKY_SEGFORMER_MAX_SIDE" \
  --sky_segformer_threshold "$SKY_SEGFORMER_THRESHOLD" \
  --aggregate_by_label \
  --require_sky_layer \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_variant "$SAM_VARIANT" \
  --sam2_checkpoint "$SAM2_CHECKPOINT" \
  --quality "$QUALITY" \
  --mps_rasterizer "$MPS_RASTERIZER" \
  --max_points "$MAX_POINTS" \
  --downsample_ratio "$DOWNSAMPLE_RATIO" \
  --training_image_size "$TRAINING_IMAGE_SIZE" \
  --layer_iterations "$LAYER_ITERATIONS" \
  --background_iterations "$BACKGROUND_ITERATIONS" \
  --sky_iterations "$SKY_ITERATIONS" \
  ${TOPOLOGY_ARGS[@]+"${TOPOLOGY_ARGS[@]}"} \
  --mean_lr_scale "$MEAN_LR_SCALE" \
  --repulsion_weight "$REPULSION_WEIGHT" \
  --merge_voxel_size "$MERGE_VOXEL_SIZE" \
  --merge_min_opacity "$MERGE_MIN_OPACITY" \
  --merge_max_points "$MERGE_MAX_POINTS" \
  --global_refine_iters "$GLOBAL_REFINE_ITERS" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
  --lr_plateau_patience "$LR_PLATEAU_PATIENCE" \
  --lr_plateau_factor "$LR_PLATEAU_FACTOR" \
  --lr_plateau_min_lr "$LR_PLATEAU_MIN_LR"

echo "Done: Grounding-SAM multilayer pipeline finished. Outputs in $OUT_DIR/scene"
