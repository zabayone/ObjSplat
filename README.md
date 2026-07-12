# ObjSplat

ObjSplat is a thesis project for object-aware 3D Gaussian Splatting from a single panoramic image. The pipeline starts from one equirectangular panorama, detects semantic regions with GroundingDINO, segments them with SAM/SAM2, projects the masks into 3D, and trains a layered 3DGS scene where each object or stuff region has its own layer.

The active workflow targets macOS / Apple Silicon and uses the MLX Splat-Apple backend.

## Pipeline

1. Put the input panorama at `outputs_park/rgb.png` or set `OUT_DIR` to another output folder containing `rgb.png`.
2. Estimate or load panoramic depth.
3. Generate perspective training views from the panorama.
4. Run GroundingDINO on prompted classes and use SAM/SAM2 box prompts to create masks.
5. Project panorama masks into the 3D point cloud.
6. Write object-aware `traindata/layerK` folders.
7. Train each layer as a 3D Gaussian Splatting scene with MLX.
8. Merge layer PLY files and optionally run a conservative opacity-only refinement pass.

## Quick Start

```bash
./run_from_pano.sh
```

Useful overrides:

```bash
OUT_DIR=outputs_mountain ./run_from_pano.sh
GROUNDING_PROMPTS="sky . mountain . snow . house . railing . person" ./run_from_pano.sh
CLEAN_OUTPUT=0 FORCE_RESEGMENT=0 ./run_from_pano.sh
```

## Main Outputs

- `traindata/layer_instances.json`: instance metadata, labels, scores, layer mapping, and coverage.
- `traindata/layerK/`: masked frames and per-layer point clouds.
- `traindata/layer_mask_visualization.png`: panorama-level layer visualization.
- `scene/gsplat_layerK.ply`: trained layer splats.
- `scene/gsplat_scene_merged.ply`: merged scene.
- `scene/gsplat_scene_merged_refined.ply`: optional conservative refinement.

## Important Files

- `run_from_pano.sh`: one-command workflow from `OUT_DIR/rgb.png`.
- `run_objsplat_pipeline.py`: orchestration for layer generation, training, merge, and refinement.
- `generate_layer_data.py`: Grounding-SAM layer data generation.
- `LayerPano.py`: per-layer training driver.
- `mps_splat_backend.py`: MLX 3DGS training, adaptive topology, merge, and refinement logic.
- `utils/semantic_instance_detection.py`: GroundingDINO + SAM/SAM2 detection and mask cleanup.
- `utils/open_ply_in_supersplat.py`: helper to inspect PLY outputs in SuperSplat.

## Useful Settings

Most defaults can be overridden through environment variables before running `run_from_pano.sh`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OUT_DIR` | `outputs_park` | Output folder containing `rgb.png`. |
| `GROUNDING_PROMPTS` | park classes | Semantic classes detected by GroundingDINO. |
| `GROUNDING_BOX_THRESHOLD` | `0.18` | Lower values keep more detections. |
| `GROUNDING_TEXT_THRESHOLD` | `0.15` | Lower values accept weaker text matches. |
| `GROUNDING_MASK_MIN_AREA` | `500` | Minimum SAM mask area in pixels. |
| `GROUNDING_INFER_MAX_SIDE` | `1536` | Max panorama side used for GroundingDINO inference. |
| `GROUNDING_BOX_PADDING` | `0.00` | Box padding used to clip SAM masks. |
| `GROUNDING_MORPH_OPEN_KERNEL` | `9` | Morphological cleanup kernel for thin mask artifacts. |
| `GROUNDING_MIN_COMPONENT_AREA_RATIO` | `0.08` | Minimum detached component size kept after SAM cleanup. |
| `MIN_POINTS_3D` | `1000` | Minimum projected 3D points required for a layer. |
| `QUALITY` | `standard` | Training preset for 3DGS layers. |
| `MAX_POINTS` | `0` | Per-layer gaussian cap; `0` disables the explicit cap. |
| `GLOBAL_REFINE_ITERS` | `120` | Iterations for conservative opacity-only refinement. |
| `CLEAN_OUTPUT` | `1` | Set `0` to reuse existing output artifacts. |
| `FORCE_RESEGMENT` | `1` | Set `0` to reuse existing `traindata`. |

## Checkpoints

Expected checkpoint paths are:

- `checkpoints/SAM 2.1 Hiera Large.pt`
- `checkpoints/sam_vit_h_4b8939.pth`
- `checkpoints/groundingdino_swinb_cogvlm.pth`
- `checkpoints/depth_anything_v2_vitl.pth`

The exact model files are not committed to the repository.

## Notes For Thesis Work

ObjSplat is designed to make the reconstructed 3DGS scene inspectable and editable at object level. Instead of optimizing one monolithic scene, it separates detected semantic regions into layers, trains them independently, and merges them back into a unified object-aware representation.

By default, same-label Grounding-SAM instances are aggregated into shared training layers to improve training signal, while each point keeps its original instance label.

## Credits

This project builds on concepts and code from LayerPano3D, LabelGS, GroundingDINO, SAM/SAM2, and Splat-Apple. Please cite the original projects when using their components or pretrained models.
