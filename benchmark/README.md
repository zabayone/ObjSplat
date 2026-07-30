# ObjSplat thesis benchmark

This directory implements reproducible measurement for completed ObjSplat
scenes and end-to-end experiments. It produces atomic JSON metadata, stable CSV
tables, resource traces, PNG/PDF thesis plots, and a Markdown report.

## Quick commands

Run the inexpensive existing-output smoke test:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/run_benchmark.py \
  --config benchmark/configs/smoke_test.yaml
```

Or analyse one completed scene directly:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/analyse_existing_scene.py \
  --scene_root outputs_lgs \
  --output benchmark_results/manual/outputs_lgs
```

Edit `input_panorama` and `scene_root` in the complete configuration, then run:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/run_benchmark.py \
  --config benchmark/configs/complete_single_scene.yaml
```

For multiple scenes, edit and run:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/run_benchmark.py \
  --config benchmark/configs/thesis_benchmark.yaml
```

Run the prepared six-scene suite (three indoor and three outdoor ERP images):

```bash
bash benchmark/run_evaluation_suite.sh
```

This runs the matched complete protocol on all six scenes, then focused
one-factor-at-a-time ablations on `indoor_1` and `outdoor_1`. The ablations
reuse immutable depth, masks, point clouds, and perspective frames from the
main run; only training-dependent artifacts remain isolated. To run only the
main six-scene experiment:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/run_benchmark.py \
  --config benchmark/configs/evaluation_six_scenes.yaml
```

Successful runs with the same resolved scientific configuration are reused.
Pass `--force` when a genuinely independent rerun is required:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/run_benchmark.py \
  --config benchmark/configs/evaluation_six_scenes.yaml --force
```

Regenerate aggregate tables, statistics, plots, and the report without rerunning
the pipeline:

```bash
/opt/anaconda3/envs/layerpano3d/bin/python benchmark/aggregate_results.py \
  --input benchmark_results/thesis_multi_scene \
  --output benchmark_results/thesis_multi_scene/report
```

## Output layout

Each scene run contains:

```text
experiment_config.json    resolved experiment and active scene
system_info.json          system, software, Git, command, seed, ERP metadata
run_summary.json          status, outputs, peak RSS, failed stage
stage_timings.csv         structured wall/CPU/memory stage records
resource_samples.csv      periodic process-tree and system memory estimates
layer_metrics.csv         one row per semantic/background/residual layer
segmentation_metrics.csv  intrinsic rows and optional ground-truth rows
reconstruction_metrics.csv
rendering_metrics.csv
editing_metrics.csv
mood_metrics.csv
failures.json             created when a stage fails
images/                   evaluation evidence
edited_representations/   optional no-retraining edit PLYs
```

The experiment `report/` directory contains aggregated versions, a JSON
statistical summary, 15 plot types in PNG and PDF, and `report.md`. A plot with
missing optional inputs is deliberately rendered as “Metric unavailable”.

## Configuration

Required fields are `experiment_name` and a non-empty `scenes` list. Each scene
needs `scene_root`; `name`, `input_panorama`, per-scene `pipeline_args`, and
`ground_truth_root` are optional.

Main switches:

- `rerun_segmentation`, `retrain`, and `reuse_existing_outputs`;
- `run_monolithic_baseline`;
- `run_quality_evaluation`, `run_rendering_benchmark`,
  `run_edit_locality`, and `run_mood_evaluation`;
- `random_seed`, `resource_sampling_interval`, and `fail_fast`;
- deterministic `splits`;
- rendering resolution, warm-up frames, and measured frames;
- explicit baseline training budget;
- selected layer and instance edit targets;
- arbitrary `pipeline_args`.

An `ablations` list creates named experiment variants using configuration, not
hard-coded Python. Each entry contains `name`, optional overrides, and
`pipeline_args`. Retraining ablations must use distinct `scene_root` values;
otherwise sequential variants would intentionally reuse/replace the same
pipeline artifacts. When every ablation changes only training parameters,
`shared_preprocessing_root_template` can point to a completed main scene. The
runner links its immutable preprocessing data into each isolated ablation root
and rejects `rerun_segmentation: true`. The following parameters map directly
to existing CLI flags:

- sky backend: `--sky_segmentation_backend`;
- far-sphere alternative: `--sky_sphere_radius` and associated radius settings;
- aggregated versus separate instances: include/omit `--aggregate_by_label`;
- filled versus unassigned background: `--fill_unassigned_layers`;
- adaptive topology: `--adaptive_topology`;
- image size and object/background/sky iteration counts;
- `--min_points_3d`;
- `--global_refine_iters`;
- analytic mood fitting versus `--mood_refine_iters`.

The exact resolved configuration is saved per run.

### Panorama format and resolution

`input_preprocessing` accepts JPEG and PNG, applies EXIF orientation, converts
to RGB, verifies the 2:1 ERP aspect ratio, and writes a real `rgb.png` working
file. It records the source SHA-256, original format and dimensions, output
dimensions, and resize filter in `input_preparation.json`.

The six-scene protocol normalizes its 10K–14K sources to `4096×2048`, preventing
source pixel count from becoming an uncontrolled memory/runtime variable.
Original images are not modified. Reusing a working `scene_root` with a
different source or preprocessing size fails explicitly, avoiding stale depth
or training artifacts.

## Adding scenes and manual masks

Add another mapping under `scenes`. For a completed scene it must contain
`rgb.png`, `traindata/layer_instances.json`, layer folders, and preferably a
merged PLY. Missing outputs create unavailable/failed rows rather than invented
measurements.

Manual masks use either:

```text
benchmark_ground_truth/<scene_name>/<semantic_label>.png
benchmark_ground_truth/<scene_name>/layer<index>.png
```

Set `ground_truth_root: benchmark_ground_truth`. Label names take precedence.
Intrinsic coverage is always separated from ground-truth accuracy.

## Train/evaluation split

Complete runs select evaluation indices with NumPy's seeded generator. Those
views may inform panorama segmentation but are excluded from each layer’s 3DGS
training frames, global refinement, and monolithic baseline. Existing scenes
without `benchmark_view_split` do not receive misleading post-hoc “held-out”
scores; reconstruction rows say the metric is unavailable.

All perspective references derive from the input ERP. Therefore PSNR, SSIM,
LPIPS, and MAE measure panorama reconstruction fidelity, not true independent
multi-view geometry.

## Baseline, edits, and moods

The monolithic baseline uses the same point data, unmasked perspective images,
held-out split, resolution, and MLX renderer with an explicit comparable budget.
Joint versus independent optimization remains an unavoidable structural
difference.

Layer removal rebuilds a PLY from all non-target layer PLYs. Instance filtering
requires the integer `label` property. If identity was lost, the row is
unavailable. Exact leakage/locality equations are in
[METHODOLOGY.md](METHODOLOGY.md).

Mood comparison verifies count and schema before comparing corresponding
Gaussian properties. Night seam metrics are reused from
`night_generation.json`. Target-night render fidelity is blank unless a valid
render/reference evaluation is run.

## Resource and runtime implications

The benchmark does not publish unsupported runtime estimates. Full training,
the monolithic baseline, LPIPS, rendering sweeps, edited PLY copies, and mood
variants can each add substantial time or disk use depending on ERP size and
Gaussian count. The smoke configuration performs no neural-network training.
The prepared suite uses five warm-up frames and 30 measured renderer frames:
enough for median and percentile statistics without repeating 100 equivalent
renders. Renderer timing includes MLX execution and synchronization, but not
the NumPy/uint8 conversion needed only when saving evaluation images.

Training views are sampled in shuffled epochs, so every view is used equally
often before any view is repeated. Python, NumPy, Torch, and MLX share the
configured seed. LPIPS keeps one AlexNet instance per process rather than
reloading it for every held-out image.

On Apple Silicon, process RSS includes the benchmark process and subprocess
tree, while system available/used/swap values reflect unified memory. They are
estimates and must not be described as dedicated GPU VRAM.

## Stable CSV schemas

Schemas live in `benchmark/schemas.py`; new columns must be appended. Empty
optional fields are valid. Units are encoded in names (`_seconds`, `_bytes`,
`_percent`, `_db`, `_ms`). Nested structures such as instance ID lists remain
in JSON or JSON-encoded CSV cells.

## Failures and recovery

Every completed stage and resource sample is appended, flushed, and `fsync`ed.
The recorder finalizes failed contexts and `run_benchmark.py` continues to the
next scene unless `fail_fast` is true. `partial_success` means analysis outputs
exist despite a later failure. Aggregation accepts missing optional files.
With `reuse_existing_outputs: true`, the runner fingerprints the resolved
configuration and reuses only a matching successful run; failed, partial, or
scientifically different runs are never treated as complete. Set
`reuse_requires_scene_artifacts: true` when a later experiment reuses the
scene’s masks, point clouds, or frames; a missing working scene is then rebuilt
even if its old metric tables still exist.

## Tests

```bash
/opt/anaconda3/envs/layerpano3d/bin/python -m unittest discover \
  -s benchmark/tests -p 'test_*.py' -v
```

Tests use synthetic arrays and tiny PLYs; they do not load model checkpoints or
run expensive training.
