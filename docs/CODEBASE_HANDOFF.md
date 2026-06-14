# ObjSplat Codebase Handoff

Last updated: 2026-06-13

This note is meant as a compact handoff for future development sessions. ObjSplat is a thesis project for an object-aware 3D Gaussian Splatting pipeline from a single equirectangular panorama.

## Current Goal

The project takes `outputs_lgs/rgb.png`, detects semantic objects/stuff regions with GroundingDINO, segments them with SAM/SAM2, projects masks into the panorama point cloud, writes LayerPano-compatible per-layer training data, trains 3DGS layers with Splat-Apple, and merges/refines the final object-aware scene.

The main research direction is object-aware 3DGS: the final PLY should remain inspectable/editable by object or semantic region. Same-label instances can be aggregated into one training layer while preserving per-instance point labels.

## Main Entry Point

Use:

```bash
./run_from_pano.sh
```

Input:

```text
outputs_lgs/rgb.png
```

Important behavior:

- The script preserves only `outputs_lgs/rgb.png` and deletes the rest of `outputs_lgs` before rebuilding.
- It runs GroundingDINO + SAM/SAM2 first.
- It uses MLX Splat-Apple training by default.
- It enables `--aggregate_by_label`, so same-label detections are grouped into shared training layers.
- `run_from_pano_deva.sh` is no longer the active script in the workspace; use `run_from_pano.sh`.

Default scene classes for the park scene:

```text
sky . road . pavement . grass . leaves . tree . bush . person . plant
```

## Key Files

- `run_from_pano.sh`
  One-command workflow. Sets defaults for Grounding-SAM, 3DGS quality, MLX backend, merge/refinement, and cleanup.

- `run_layered_deva_pipeline.py`
  Orchestrates the full pipeline: layer generation, per-layer 3DGS training, merge, optional global refinement.

- `gen_layerdata_from_deva.py`
  Generates LayerPano-compatible training data. Despite the historical filename, it now supports the Grounding-SAM-first path and optional DEVA frame segmentation.

- `utils/semantic_instance_detection.py`
  GroundingDINO + SAM/SAM2 detection and mask cleanup. This is where most mask artifact fixes live.

- `utils/deva_instance_segmentation.py`
  Optional legacy/alternative DEVA + SAM perspective-frame segmentation.

- `LayerPano.py`
  Loads `traindata/layerK`, trains each layer, and preserves labels from PLY input. It deduplicates layer indices so aggregated layers are trained once.

- `mps_splat_backend.py`
  Splat-Apple bridge for torch and MLX training, adaptive topology, PLY export, merge, and global refinement.

- `submodules/splat-apple/mlx_gs/...`
  Local MLX backend code. Some compatibility/stability patches have been applied here.

## Output Layout

Typical generated files:

```text
outputs_lgs/traindata/deva_instances.json
outputs_lgs/traindata/layer_mask_visualization.png
outputs_lgs/traindata/layerK/pcd_rgb_layerK.ply
outputs_lgs/traindata/layerK/pcd_mask_layerK.ply
outputs_lgs/traindata/layerK/frames/rgb_*.png
outputs_lgs/scene/gsplat_layerK.ply
outputs_lgs/scene/gsplat_scene_merged.ply
outputs_lgs/scene/gsplat_scene_merged_refined.ply
```

The metadata file still uses the historical name `deva_instances.json` for compatibility, even when the backend is `grounding_sam`.

## Current Grounding-SAM Behavior

The Grounding-SAM path is the default. Current important mask cleanup choices:

- SAM is prompted with GroundingDINO boxes.
- Masks are clipped to the original Dino box by default.
- `GROUNDING_BOX_PADDING=0.00` in `run_from_pano.sh`.
- Only one connected component is kept per detection.
- The kept component must overlap the original Dino box.
- Morphological opening is enabled with `GROUNDING_MORPH_OPEN_KERNEL=9`.
- Small detached components are controlled by `GROUNDING_MIN_COMPONENT_AREA_RATIO=0.08`.
- Same-label instances are aggregated with `--aggregate_by_label`.

Why aggregation matters:

- Earlier runs produced many tiny layers, often around 1000-1800 points.
- Full-frame masked training targets were mostly black for tiny objects.
- MLX training then collapsed opacity and exported invisible layer PLYs.
- Aggregating same-label instances makes each training layer larger and gives the optimizer a stronger signal.
- Instance identity is still preserved because PLY point labels remain the original `instance_id`.

## MLX 3DGS Fixes Already Applied

Several fixes were made to avoid smeared or invisible MLX output:

- MLX PLY export writes log-scale values directly instead of exponentiating them.
- `scale_log_max` is forced down for MLX, currently capped around `-2.8`.
- Projected covariance/radius is clamped in `submodules/splat-apple/mlx_gs/renderer/projection.py`.
- `mx.full_like` was replaced with `mx.ones_like(...) * value` for compatibility with the installed MLX version.
- MLX opacity clamp floor was raised from `-10.0` to `-4.0`.
- Opacity mean regularization is disabled for MLX object-aware layers because it pushed opacity toward invisibility.
- `MERGE_MIN_OPACITY` default is `-20`, because opacity is stored as raw logits, not sigmoid opacity.

## Important Defaults In `run_from_pano.sh`

Segmentation:

```bash
GROUNDING_BOX_THRESHOLD=0.18
GROUNDING_TEXT_THRESHOLD=0.15
GROUNDING_MASK_MIN_AREA=500
GROUNDING_INFER_MAX_SIDE=1536
GROUNDING_BOX_PADDING=0.00
GROUNDING_MORPH_OPEN_KERNEL=9
GROUNDING_MIN_COMPONENT_AREA_RATIO=0.08
MIN_POINTS_3D=1000
```

Views:

```bash
N_VIEWS=12
PHI_BANDS=80,67.5,45,0,-45,-67.5,-80
```

Training:

```bash
QUALITY=standard
MPS_TRAINING_BACKEND=mlx
MPS_RASTERIZER=cpp
MAX_POINTS=3000000
DOWNSAMPLE_RATIO=1.0
MEAN_LR_SCALE=0.35
REPULSION_WEIGHT=5e-5
MERGE_MIN_OPACITY=-20
GLOBAL_REFINE_ITERS=300
```

## Known Issues / Watch List

- Mask blobs:
  The current fix is strict Dino-box clipping plus single-component retention. If blobs remain, inspect `outputs_lgs/traindata/layerK/layerK_erp_mask.png` and compare with GroundingDINO boxes in `deva_instances.json`.

- Sky/road overreach:
  Large stuff classes can still dominate if GroundingDINO boxes are huge or overlapping. Because same-label aggregation is now enabled, inspect `layer_groups` in metadata to see which instances are grouped.

- Invisible trained layers:
  Previous invalid runs had layer PLYs with `opacity=-10.0` and only 128 gaussians. Those outputs should be discarded and regenerated. New runs should not reuse old `outputs_lgs/scene/gsplat_layer*.ply`.

- Metadata compatibility:
  `deva_instances.json` may be from an older run. New runs with aggregation should include `training_layer_count` and `layer_groups`. If those keys are missing, the data predates aggregation.

- Git:
  In one session, `git status` reported that the workspace was not a Git repository, even though `.git` may be readable in the sandbox profile. Do not rely on Git status unless it works locally.

## Quick Diagnostics

Check shell syntax:

```bash
bash -n run_from_pano.sh
```

Check Python syntax:

```bash
python3 -m py_compile \
  gen_layerdata_from_deva.py \
  run_layered_deva_pipeline.py \
  LayerPano.py \
  mps_splat_backend.py \
  utils/semantic_instance_detection.py \
  utils/deva_instance_segmentation.py
```

Inspect whether current metadata is from the aggregation-aware pipeline:

```bash
python3 - <<'PY'
import json
m=json.load(open('outputs_lgs/traindata/deva_instances.json'))
print('instances:', m.get('instance_count'))
print('training layers:', m.get('training_layer_count'))
print('groups:', m.get('layer_groups', [])[:5])
PY
```

Inspect trained PLY opacity with the project environment:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python - <<'PY'
from pathlib import Path
from plyfile import PlyData
import numpy as np
for p in sorted(Path('outputs_lgs/scene').glob('gsplat_layer*.ply'))[:10]:
    v = PlyData.read(p)['vertex'].data
    op = np.asarray(v['opacity']) if 'opacity' in v.dtype.names else np.array([])
    print(p.name, len(v), float(op.min()) if op.size else None, float(op.max()) if op.size else None)
PY
```

## Suggested Next Steps

1. Rerun the full pipeline from a clean `outputs_lgs/rgb.png` using `./run_from_pano.sh`.
2. Confirm `deva_instances.json` contains `training_layer_count` and `layer_groups`.
3. Inspect `layer_mask_visualization.png` and the per-layer ERP masks before spending time on full training.
4. If masks are good but training is still weak, compare MLX vs torch for one aggregated layer.
5. If MLX still produces excessive blur, continue tuning `scale_log_max`, covariance clamp, and opacity regularization.
