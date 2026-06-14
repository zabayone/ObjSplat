# DEVA Instance Layered Training

This document describes the DEVA-instance layered training pipeline.

## Overview
- DEVA+SAM extracts per-frame instance masks from perspective views.
- Each instance becomes a dedicated training layer.
- A background layer is optionally trained separately.
- All layer PLYs are merged into a single `gsplat_scene_merged.ply`.
- An optional global refinement step can be run after the merge.

## Key Scripts
- `gen_layerdata_from_deva.py` creates per-instance `traindata/layerK` folders.
- `run_layered_deva_pipeline.py` runs the full pipeline end-to-end.

## Example Run
```bash
/opt/anaconda3/envs/layerpano3d/bin/python run_layered_deva_pipeline.py \
  --input_dir outputs_lgs \
  --save_dir outputs_lgs \
  --device mps \
  --min_frames 3 \
  --min_total_pixels 10000 \
  --min_points_3d 5000 \
  --merge_voxel_size 0.005 \
  --merge_min_opacity 0.02
```

## Notes
- The pipeline writes `traindata/deva_instances.json` with instance-to-layer mapping.
- `LayerPano.create_deva_instances(...)` disables inter-layer transfer by design.
- Merge uses simple voxel dedup + opacity pruning.
