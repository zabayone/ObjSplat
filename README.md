# ObjSplat

ObjSplat is a thesis project for object-aware 3D Gaussian Splatting from a single panoramic image. The pipeline starts from one equirectangular panorama, detects semantic regions with GroundingDINO, segments them with SAM/SAM2, projects the masks into 3D, and trains a layered 3DGS scene where each object or stuff region has its own layer.

The active workflow targets macOS / Apple Silicon and uses the MLX Splat-Apple backend.

> **Research status.** ObjSplat is a master's thesis prototype. It prioritizes
> reproducible experimentation and inspectable object-aware outputs; it is not a
> production reconstruction service.

## Features

- Single-ERP reconstruction with an Apple Silicon–native MLX training backend.
- GroundingDINO + SAM/SAM2 semantic-instance decomposition.
- Independent object, background, residual, and far-sphere sky layers.
- Integer instance labels retained in Gaussian PLY files when supported.
- Layer removal, instance filtering, conservative global refinement, and mood variants.
- Reproducible multi-scene benchmarking with resource traces, quantitative
  metrics, plots, failure recovery, and Markdown reports.

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

Clone the repository with its submodules, prepare the Apple Silicon environment,
and consult [checkpoints/README.md](checkpoints/README.md) for model placement:

```bash
git clone --recurse-submodules https://github.com/zabayone/ObjSplat.git
cd ObjSplat
./setup_arm64.sh
```

Prepare the environment and checkpoints, then place your panorama at:

```bash
outputs_lgs/rgb.png
```

Run the full pipeline:

```bash
./run_from_pano.sh
```

Useful overrides:

```bash
OUT_DIR=outputs_mountain ./run_from_pano.sh
GROUNDING_PROMPTS="sky . mountain . snow . house . railing . person" ./run_from_pano.sh
CLEAN_OUTPUT=0 FORCE_RESEGMENT=0 ./run_from_pano.sh
RETEXTURE_NIGHT_SKY=1 ./run_from_pano.sh
OUT_DIR=outputs_park CLEAN_OUTPUT=0 FORCE_RESEGMENT=1 SEGMENT_ONLY=1 RETEXTURE_NIGHT_SKY=1 ./run_from_pano.sh
OUT_DIR=outputs_park CLEAN_OUTPUT=0 MOOD_ONLY=1 BUILD_NIGHT_MOOD=1 ./run_from_pano.sh
OUT_DIR=outputs_park CLEAN_OUTPUT=0 MOOD_ONLY=1 MOOD_PRESETS="serene,joyful,tense,melancholic" ./run_from_pano.sh
```

Generate the night-sky ERP after segmentation:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python retexture_sky.py \
  --scene_root outputs_mountain \
  --model_path checkpoints/FLUX.1-Fill-dev
```

The command reads the original `rgb.png` and the canonical sky mask, applies
FLUX Fill with circular horizontal context, and writes:

- `traindata/sky/night_rgb.png`: masked night sky layer;
- `traindata/sky/night_composite.png`: full ERP with non-sky pixels preserved;
- `traindata/sky/night_generation.json`: reproducibility parameters and seam metrics.

Use `--dry_run` to validate paths, mask coverage, and working resolution without
loading the diffusion model. The same operation can be included in the main
pipeline with `--retexture_night_sky`.

With `RETEXTURE_NIGHT_SKY=1`, the full pipeline also relights non-sky pixels,
fits the resulting ERP to the already trained Gaussian topology, and creates
`scene/gsplat_scene_night.ply`. Geometry, scale, rotation, opacity, labels, and
the number of points are preserved exactly; only SH appearance coefficients
change. Optional `NIGHT_MOOD_REFINE_ITERS` performs a short SH-only refinement.
It defaults to `0` because the analytic ERP fit is substantially faster for
multi-gigabyte scenes.

Switch variants without copying a PLY:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python switch_mood.py \
  --scene_root outputs_park --mood night
```

The active symlink is `scene/gsplat_scene_active.ply`. Use `--mood day` to
switch back. If a day scene and night ERP already exist, `MOOD_ONLY=1` builds
the night PLY without rerunning segmentation or day training.

### Circumplex Moods

Mood variants use the circumplex coordinates `valence` and `arousal`, both in
`[-1, 1]`. Time of day remains independent, so the same emotional coordinates
can produce a day or night variant. Built-in anchors are:

| Preset | Valence | Arousal | Appearance |
| --- | ---: | ---: | --- |
| `serene` | `+0.72` | `-0.68` | Soft contrast, warm-neutral light, lifted shadows. |
| `joyful` | `+0.82` | `+0.74` | Brighter, warmer, more saturated appearance. |
| `tense` | `-0.76` | `+0.82` | Cold tint, stronger contrast, directional lighting. |
| `melancholic` | `-0.72` | `-0.62` | Dim, blue, desaturated and softly directional. |
| `night` | `-0.15` | `-0.35` | Generated night sky and full-scene night relighting. |

Generate several preset PLY variants from an existing trained day scene:

```bash
OUT_DIR=outputs_park CLEAN_OUTPUT=0 MOOD_ONLY=1 \
  MOOD_PRESETS="serene,joyful,tense,melancholic" ./run_from_pano.sh
```

Generate an arbitrary point in the continuous circumplex:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python run_objsplat_pipeline.py \
  --input_dir outputs_park --save_dir outputs_park --mood_only \
  --mood_name focused --mood_valence 0.25 --mood_arousal 0.70
```

Add `--mood_time_of_day night` to apply the emotional offset to the night base;
this requires an existing generated `traindata/sky/night_composite.png`.
Every mood reuses the day Gaussian topology. Only the ERP appearance and SH
coefficients change, and `scene/moods.json` stores its circumplex coordinates.

Run the inexpensive layer-data preflight independently:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python validate_scene.py \
  --scene_root outputs_park \
  --require_sky
```

The default shell workflow uses a hybrid sky segmenter: lightweight
`nvidia/segformer-b2-finetuned-ade-512-512` provides the semantic sky region,
while high-resolution SAM masks protect trees, poles, roofs, and other thin
foreground silhouettes. The sky pass also fills conservative zenith holes so
the top-center panorama does not collapse to black, and the night retexturing
path adds a sparse procedural star field if FLUX smooths stars away.
GroundingDINO + SAM2 remain responsible for the object-aware layers.

## Main Outputs

- `traindata/layer_instances.json`: instance metadata, labels, scores, layer mapping, and coverage.
- `traindata/layerK/`: full-color frames, explicit supervision masks, and per-layer point clouds.
- `traindata/layer_mask_visualization.png`: panorama-level layer visualization.
- `traindata/sky/mask.png` and `traindata/sky/day_rgb.png`: stable ERP inputs for sky retexturing.
- `scene/gsplat_layerK.ply`: trained layer splats.
- `scene/gsplat_scene_merged.ply`: merged scene.
- `scene/gsplat_scene_merged_refined.ply`: optional conservative refinement.
- `traindata/moods/<mood>/scene_rgb.png`: complete relit ERP for each mood.
- `scene/gsplat_scene_<mood>.ply`: mood appearance with day geometry preserved.
- `scene/moods.json` and `scene/gsplat_scene_active.ply`: mood manifest and active variant.

## Reproducible Benchmarking

The benchmark can analyse completed outputs or retrain from original ERP images
with deterministic held-out perspective views. It records system/software
metadata, stage timings, resource samples, layer and segmentation statistics,
reconstruction fidelity, rendering speed, edit locality, mood topology, robust
failure information, aggregate statistics, and thesis-ready plots.

Analyse an existing scene without retraining:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/analyse_existing_scene.py \
  --scene_root outputs_lgs
```

Run the prepared six-scene indoor/outdoor evaluation suite:

```bash
bash benchmark/run_evaluation_suite.sh
```

The suite expects local 2:1 ERP files under `benchmark/evaluation/`; these large
inputs are intentionally ignored by Git. See
[benchmark/evaluation/README.md](benchmark/evaluation/README.md) for naming and
[benchmark/README.md](benchmark/README.md) for protocol details.

Metric definitions and interpretation are available in
[docs/ObjSplat_Benchmark_Metrics.docx](docs/ObjSplat_Benchmark_Metrics.docx)
and [benchmark/METHODOLOGY.md](benchmark/METHODOLOGY.md).

## Important Files

- `run_from_pano.sh`: one-command workflow from `OUT_DIR/rgb.png`.
- `run_objsplat_pipeline.py`: orchestration for layer generation, training, merge, and refinement.
- `generate_layer_data.py`: Grounding-SAM layer data generation.
- `LayerPano.py`: per-layer training driver.
- `mps_splat_backend.py`: MLX 3DGS training, adaptive topology, merge, and refinement logic.
- `utils/semantic_instance_detection.py`: GroundingDINO + SAM/SAM2 detection and mask cleanup.
- `utils/open_ply_in_supersplat.py`: helper to inspect PLY outputs in SuperSplat.
- `benchmark/run_benchmark.py`: multi-scene benchmark orchestrator.
- `benchmark/configs/scientific_core.yaml`: default thesis protocol with a
  controlled Gaussian budget, paired monolithic baseline, held-out fidelity,
  multi-size edit targets, rendering, memory, and day/night topology metrics.
- `benchmark/schemas.py`: stable machine-readable CSV schemas.
- `benchmark/configs/`: smoke, complete, thesis, and ablation configurations.
- `benchmark/run_evaluation_suite.sh`: six-scene evaluation entry point.

## Repository Structure

```text
ObjSplat/
├── benchmark/              Reproducible evaluation framework and configurations
├── gaussian_renderer/      Legacy Gaussian rendering integration
├── rendering/              Video and trajectory render utilities
├── scene/                  Gaussian scene/model abstractions
├── src/                    Upstream research components
├── submodules/             External backends, including Splat-Apple
├── tests/                  Lightweight pipeline regression tests
├── utils/                  Detection, depth, mood, sky, and viewer utilities
├── generate_layer_data.py  ERP-to-layer training-data generation
├── run_objsplat_pipeline.py
└── run_from_pano.sh        Default end-to-end shell entry point
```

For large local PLY files, open the scene through:

```bash
python utils/open_ply_in_supersplat.py outputs_lgs
```

The helper serves real HTTP byte ranges (`206 Partial Content`) and streams
them in bounded chunks. This avoids SuperSplat's whole-file memory fallback,
which can exhaust the browser buffer on scenes containing millions of
Gaussians. When the scene contains `scene/moods.json`, the local viewer also
shows an ObjSplat mood panel:

- drag the valence-arousal pad for immediate whole-scene GPU modulation;
- use the keyboard-accessible sliders for precise coordinates;
- filter exact variants by day/night;
- click a preset to load its precomputed PLY while preserving the camera;
- use **Load nearest exact mood** to snap a continuous point to the closest
  generated variant.

Passing a specific PLY remains supported. If that PLY is inside a mood-enabled
`scene/` directory, the controls are discovered automatically.

## Useful Settings

Most defaults can be overridden through environment variables before running `run_from_pano.sh`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROUNDING_PROMPTS` | default classes (listed above) | Semantic classes detected by GroundingDINO. |
| `GROUNDING_BOX_THRESHOLD` | `0.18` | Lower values keep more detections. |
| `GROUNDING_TEXT_THRESHOLD` | `0.15` | Lower values accept weaker text matches. |
| `GROUNDING_MASK_MIN_AREA` | `500` | Minimum SAM mask area in pixels. |
| `GROUNDING_INFER_MAX_SIDE` | `1536` | Max panorama side used for GroundingDINO inference. |
| `GROUNDING_BOX_PADDING` | `0.12` | Box padding used to clip SAM masks. |
| `GROUNDING_MORPH_OPEN_KERNEL` | `3` | Morphological cleanup kernel for thin mask artifacts. |
| `GROUNDING_MIN_COMPONENT_AREA_RATIO` | `0.01` | Minimum detached component size kept after SAM cleanup. |
| `MIN_POINTS_3D` | `1000` | Minimum projected 3D points required for a layer. |
| `PERSPECTIVE_SIZE` | `1024` | Maximum side of generated perspective training views. |
| `SKY_SEGMENTATION_BACKEND` | `hybrid` | Use SegFormer for sky and Grounding-SAM for object boundaries. |
| `SKY_SEGFORMER_MAX_SIDE` | `2048` | Semantic sky inference resolution before restoring the ERP mask. |
| `QUALITY` | `standard` | Training preset for 3DGS layers. |
| `MAX_POINTS` | `0` | Per-layer gaussian cap; `0` disables pruning and densification caps to preserve scene coverage. |
| `TRAINING_IMAGE_SIZE` | `640` | Rasterized training side; source perspective frames remain at full resolution. |
| `LAYER_ITERATIONS` | `800` | Standard object-layer iterations. |
| `BACKGROUND_ITERATIONS` | `1000` | Background-layer iterations. |
| `SKY_ITERATIONS` | `500` | Sky-layer iterations. |
| `ADAPTIVE_TOPOLOGY` | `0` | Opt-in prune/clone/split; disabled by default to preserve ERP coverage. |
| `GLOBAL_REFINE_ITERS` | `0` | Optional conservative opacity-only refinement. |
| `CLEAN_OUTPUT` | `0` | Set `1` only for a deliberately clean rebuild; the default reuses expensive artifacts. |
| `FORCE_RESEGMENT` | `0` | Set `1` when the panorama or segmentation configuration changed. |
| `SEGMENT_ONLY` | `0` | Stop after segmentation and optional sky retexturing. |
| `MOOD_ONLY` | `0` | Reuse segmentation and day PLY, generating only mood outputs. |
| `RETEXTURE_NIGHT_SKY` | `0` | Generate canonical night-sky ERP assets before 3DGS training. |
| `BUILD_NIGHT_MOOD` | same as `RETEXTURE_NIGHT_SKY` | Relight the rest of the scene and build the night PLY. |
| `MOOD_PRESETS` | empty | Comma-separated circumplex anchors to build. |
| `SKY_VAE_TILING` | `0` | Enable memory-saving tiled VAE decoding; leave disabled to avoid bands in the generated sky texture. |
| `SKY_STAR_DENSITY` | `0.00065` | Density of spherical, PSF-rendered procedural stars. |
| `SKY_LUMA_CAP` | `0.42` | Maximum low-frequency night-sky luminance before hotspot compression. |
| `SKY_HOTSPOT_RATIO` | `1.55` | Maximum broad luminance relative to the median sky. |
| `NIGHT_MOOD_REFINE_ITERS` | `0` | Optional SH-only night refinement iterations. |
| `NIGHT_TRAINING_IMAGE_SIZE` | `512` | Raster size used only by optional night refinement. |
| `SKY_MODEL_PATH` | `checkpoints/FLUX.1-Fill-dev` | Local Diffusers checkpoint used for sky filling. |
| `SKY_STEPS` | `50` | FLUX Fill denoising steps. |
| `SKY_MAX_PIXELS` | `1048576` | Maximum ERP working pixels before circular padding. |

## Checkpoints

Expected checkpoint paths are:

- `checkpoints/SAM 2.1 Hiera Large.pt`
- `checkpoints/sam_vit_h_4b8939.pth`
- Hugging Face cache entry for `IDEA-Research/grounding-dino-base`
- `checkpoints/depth_anything_v2_vitl.pth`


The exact model files are not committed to the repository.

## Notes For Thesis Work

ObjSplat is designed to make the reconstructed 3DGS scene inspectable and editable at object level. Instead of optimizing one monolithic scene, it separates detected semantic regions into layers, trains them independently, and merges them back into a unified object-aware representation.

By default, same-label Grounding-SAM instances are aggregated into shared training layers to improve training signal, while each point keeps its original instance label.

The default panoramic workflow requires a dedicated sky layer. Sky detections across the ERP seam are grouped together, preserved independently of generic object-size thresholds, and excluded from the full-scene background so a future day/night sky can be switched without leaving duplicate daytime gaussians. `layer_instances.json` exposes `sky_layer_idx` and canonical day/night ERP paths.

Unassigned pixels are intentionally kept in the background instead of being
forced into the nearest detected semantic layer. The legacy dense behavior is
available only through `--fill_unassigned_layers`. The sky point cloud is
placed on a derived far sphere rather than using unreliable monocular depth
estimates for an effectively infinite surface.

## Credits

This project builds on concepts and code from LayerPano3D, LabelGS, GroundingDINO, SAM/SAM2, and Splat-Apple. Please cite the original projects when using their components or pretrained models.

## License

ObjSplat is distributed under the terms in [LICENSE](LICENSE). Third-party
components and model checkpoints retain their respective licenses.
