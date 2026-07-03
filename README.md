# ObjSplat

ObjSplat is a thesis project for object-aware 3D Gaussian Splatting from a single panoramic image.
The pipeline starts from one equirectangular panorama, detects semantic objects with GroundingDINO, segments them with SAM/SAM2, projects the masks into 3D, and trains a layered 3DGS scene where each object or stuff region has its own layer.

The current workflow is optimized for local development on macOS / Apple Silicon. It supports a fast MLX backend and a slower torch backend for 3DGS training.

## Pipeline

1. Put the input panorama at `outputs_lgs/rgb.png`.
2. Estimate or load panoramic depth.
3. Generate perspective training views from the panorama.
4. Run GroundingDINO on prompted classes and use SAM/SAM2 box prompts to create masks.
5. Project panorama masks and perspective masks into a shared 3D point cloud.
6. Write object-aware `traindata/layerK` folders.
7. Train each layer as a 3D Gaussian Splatting scene.
8. Merge layer PLY files and optionally run a final global refinement pass.

## Quick Start

Prepare the environment and checkpoints, then place your panorama at:

```bash
outputs_lgs/rgb.png
```

Run the full pipeline:

```bash
./run_from_pano.sh
```

The script preserves `outputs_lgs/rgb.png`, clears the rest of `outputs_lgs`, regenerates training data, and trains a standard-quality object-aware 3DGS scene.

Default prompted classes:

```text
sky . road . pavement . grass . leaves . tree . bush . person . plant
```

You can override them without editing the script:

```bash
GROUNDING_PROMPTS="sky . road . tree . person . bench . lamp" ./run_from_pano.sh
```

## Main Outputs

- `outputs_lgs/traindata/deva_instances.json`: instance metadata, labels, scores, layer mapping, and coverage.
- `outputs_lgs/traindata/layerK/`: masked frames and per-layer point clouds.
- `outputs_lgs/traindata/layer_mask_visualization.png`: panorama-level layer visualization.
- `outputs_lgs/scene/`: trained and merged 3DGS PLY files.

The metadata file keeps the historical `deva_instances.json` name for compatibility with the existing training code, even when the segmentation backend is Grounding-SAM.

## Important Files

- `run_from_pano.sh`: one-command thesis workflow from `outputs_lgs/rgb.png`.
- `run_layered_deva_pipeline.py`: full object-aware orchestration: layer generation, training, merge, refinement.
- `gen_layerdata_from_deva.py`: builds object-aware LayerPano training data from Grounding-SAM or DEVA masks.
- `utils/semantic_instance_detection.py`: GroundingDINO + SAM/SAM2 detection and mask cleanup.
- `utils/deva_instance_segmentation.py`: optional DEVA + SAM perspective-frame segmentation.
- `LayerPano.py`: per-layer training driver.
- `mps_splat_backend.py`: torch and MLX 3DGS training, adaptive topology, merge, and refinement logic.

## Useful Settings

Most defaults can be overridden through environment variables before running `run_from_pano.sh`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROUNDING_PROMPTS` | default classes (listed above) | Semantic classes detected by GroundingDINO. |
| `GROUNDING_BOX_THRESHOLD` | `0.18` | Lower values keep more detections. |
| `GROUNDING_TEXT_THRESHOLD` | `0.15` | Lower values accept weaker text matches. |
| `GROUNDING_MASK_MIN_AREA` | `500` | Minimum SAM mask area in pixels. |
| `GROUNDING_INFER_MAX_SIDE` | `1536` | Max panorama side used for GroundingDINO inference. |
| `GROUNDING_BOX_PADDING` | `0.00` | Box padding used to clip SAM masks; `0.00` keeps masks inside the Dino box. |
| `GROUNDING_MORPH_OPEN_KERNEL` | `9` | Morphological cleanup kernel for thin mask artifacts. |
| `GROUNDING_MIN_COMPONENT_AREA_RATIO` | `0.08` | Minimum detached component size kept after SAM cleanup. |
| `MIN_POINTS_3D` | `1000` | Minimum projected 3D points required for a layer. |
| `QUALITY` | `standard` | Training preset for 3DGS layers. |
| `MPS_TRAINING_BACKEND` | `mlx` | `mlx`, `torch`, or `auto`. |
| `GLOBAL_REFINE_ITERS` | `300` | Iterations for the final merged-scene refinement. |

## Checkpoints

Expected checkpoint paths are:

- `checkpoints/SAM 2.1 Hiera Large.pt`
- `checkpoints/sam_vit_h_4b8939.pth`
- `checkpoints/groundingdino_swinb_cogvlm.pth`

The exact model files are not committed to the repository.

## Notes For Thesis Work

ObjSplat is designed to make the reconstructed 3DGS scene inspectable and editable at object level.
Instead of optimizing one monolithic scene, it separates detected semantic regions into layers, trains them independently, and merges them back into a unified object-aware representation.
By default, same-label Grounding-SAM instances are aggregated into shared training layers to improve training signal, while each point keeps its original instance label.

This makes the pipeline useful for studying object-level control, residual/background behavior, segmentation quality, and Apple Silicon training trade-offs in 3D Gaussian Splatting.

## Credits

This project builds on concepts and code from LayerPano3D, LabelGS, GroundingDINO, SAM/SAM2, DEVA, and Splat-Apple.
Please cite the original projects when using their components or pretrained models.
