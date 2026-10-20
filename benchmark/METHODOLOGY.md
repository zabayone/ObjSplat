# ObjSplat Benchmark Methodology

## Experimental unit and reproducibility

An experimental unit is one scene, configuration, variant, and random seed. Each
run records the repository commit and dirty state, command, relevant environment
variables, Python and package versions, operating system, Apple hardware, input
ERP dimensions, and the complete resolved configuration. Missing system fields
are `null`; they are never guessed.

JPEG and PNG inputs are EXIF-normalized, converted to RGB PNG, validated as 2:1
ERPs, and optionally resized to a common working resolution. Source hash,
format, dimensions, and the transformation are recorded. The six-scene
indoor/outdoor protocol uses `4096×2048` working ERPs while preserving the
original files.

The benchmark uses UTC ISO-8601 timestamps and monotonic clocks for durations.
Stage records are appended and flushed when each context exits, including during
an exception. Resource samples are appended approximately once per configured
interval. On Apple Silicon, RSS and system memory are unified-memory estimates,
not CUDA VRAM.

The layered/monolithic comparison reports stage-local process-tree peaks:
`all_layer_training` for the layered representation and
`monolithic_training_process` for the joint representation. The maximum of the complete
benchmark process is retained for diagnostics but is not used as either
variant's training-memory result.

## Timings and resources

Wall time and CPU process time are measured independently. Structured stages
cover depth, perspective generation, combined GroundingDINO/SAM inference,
3D projection, layer-data generation, each object/background/sky training job,
merge, optional global refinement, sky generation, ERP relighting, analytic SH
fitting, optional mood refinement, baseline training, evaluation, and end-to-end
execution. A combined detection/segmentation duration is used because the current
detector API performs both in one call; it is not falsely decomposed.

For a stage with duration \(t\), \(I\) iterations, \(F\) frames, and \(P\)
input points, derived rates are \(t/I\), \(t/F\), and
\(t/(P/10^6)\). Rates are emitted only when their denominators were measured.

## Layer and segmentation metrics

Layer statistics combine `layer_instances.json`, ERP masks, supervised frame
masks, initial point PLYs, and trained Gaussian PLYs. Confidence statistics are
GroundingDINO detection scores for instances assigned to the layer.

Intrinsic coverage measures partition completeness and is not accuracy. Manual
ground truth follows:

```text
benchmark_ground_truth/<scene_name>/<semantic_label>.png
benchmark_ground_truth/<scene_name>/layer<index>.png
```

Label filenames take precedence. Masks are binary (`>=128` foreground).

For prediction \(P\) and ground truth \(G\):

\[
\mathrm{IoU}=\frac{|P\cap G|}{|P\cup G|},\quad
\mathrm{Dice}=\frac{2|P\cap G|}{|P|+|G|}
\]

Precision and recall use pixel counts. Boundary F-score matches morphological
boundaries within a two-pixel tolerance. Thin-structure IoU remains unavailable
unless a dedicated thin-structure annotation is supplied; no heuristic score is
presented as ground truth.

## Reconstruction protocol

The perspective grid is deterministic. A seeded subset is written to
`benchmark_view_split.evaluation_indices` and omitted from layer training,
global refinement, and monolithic training. Evaluation renders use the stored
poses and compare with the corresponding ERP-derived perspective images using
PSNR, SSIM, mean absolute error, and optional LPIPS.

These reference views originate from the same input panorama. The results
measure panorama reconstruction fidelity, not independent-view novel-view
synthesis or true geometric accuracy.

## Monolithic baseline

The optional baseline concatenates the same generated layer point clouds, keeps
their labels, uses the same unmasked training perspectives, resolution, held-out
split, MLX renderer, and explicit iteration budget. It differs structurally:
all Gaussians are optimized jointly rather than as independent semantic layers.
This difference and the chosen budget are retained in the configuration, so the
comparison is controlled but not claimed to be perfectly equivalent.

Non-adaptive comparison runs use no per-layer Gaussian cap. The exporter repairs
non-finite learned appearance/shape attributes with finite neutral values and,
when topology is unchanged, restores invalid positions from the input cloud.
The benchmark rejects any layer retaining less than 95% of its input Gaussian
support, preventing a numerically damaged, visibly incomplete scene from being
accepted as a valid result.

## Quality scaling

The quality sweep keeps layer, background, and sky iteration counts fixed and
changes only the sampled fraction of a common full-resolution 3D support:
`low` = 0.25, `standard` = 0.50, `high` = 0.75, and `ultra` = 1.00. The source
ERP, segmentation, held-out split, training-image size, renderer, topology
policy, and random seed remain fixed. This isolates the training cost of scene
support density without making segmentation a resolution-dependent confound.
The sweep uses one indoor and one outdoor scene and reports training wall time
and stage-local peak RSS for both layered and monolithic training; it does not
run retexturing. Every quality level is paired within the same scene and is also
compared with that variant's `standard` run before cross-scene summaries are
produced.

## Rendering performance

The first render is reported as cold start. Warm-up renders are discarded.
Steady-state latency reports mean, median, p90, p95, average FPS, minimum FPS,
Gaussian count, and megapixels per second for a deterministic sequence of stored
camera poses.

## Edit locality

Layer removal merges all trained layers except the target. Instance removal is
permitted only when the final PLY contains retained integer instance labels.
Neither operation retrains the scene.

Let \(D(x)\) be per-pixel mean absolute RGB change, \(T\) the projected target
mask, \(\mu_T\) its mean change, and \(\mu_{\bar T}\) the outside mean change:

\[
\mathrm{edit\_leakage\_ratio}
=\frac{\mu_{\bar T}}{\max(\mu_T,10^{-12})}
\]

\[
\mathrm{edit\_locality\_score}
=\frac{\mu_T}{\max(\mu_T+\mu_{\bar T},10^{-12})}
\]

Lower leakage and higher locality are better. Changed-pixel percentages use a
one-8-bit-level threshold. Occlusion and disocclusion can legitimately change
pixels outside a 2D target mask; these metrics quantify locality rather than
semantic correctness of the inpainted result.

## Mood topology preservation

Count and property schemas are verified before correspondence is assumed.
Matched arrays report mean and maximum absolute differences for position,
log-scale, quaternion, opacity, SH coefficients, and label changes. A Gaussian
is a non-appearance change if any position, scale, rotation, opacity, or label
changes beyond \(10^{-7}\). Analytic mood fitting is expected to change only SH
fields. Quaternion component differences are reported rather than an angular
distance because exact stored-topology preservation is the tested invariant.

## Statistics, failures, and limitations

Across scenes the report includes count, mean, median, sample standard deviation,
minimum, maximum, quartiles, and a Student-t 95% confidence interval
when at least two observations exist and SciPy is available. Reconstruction
statistics first average views within each scene/run/variant, so a scene with
more held-out frames does not receive extra weight. These intervals assume
independent scene-level observations and are descriptive for this small,
purposefully chosen scene set; they do not establish population-wide performance. No significance test is run automatically.

Runs are `success`, `partial_success`, or `failed`. Completed CSV rows survive
failure. The failed stage, exception type/message, last resource sample, and
successfully detected outputs are retained. Optional or technically impossible
metrics remain null with an explanation; values are never fabricated.


## Protocol corrections and result provenance (2026-09-05)

Downsampling is applied once, before both representations enter the trainer.
The monolithic subprocess seeds Python, NumPy and MLX and carries the original
ERP height into scale initialization. The existing fixed budgets remain 800
iterations per object layer, 1000 for background, 500 for sky and 1000 for the
whole monolithic scene. They are explicit method budgets, not equal total
iterations or equal compute. Layer masks, layer-dependent scale statistics and
small-layer regularization also differ from joint training.

A benchmark validates the actual stored view split and layer frame filenames,
including its match to the configured seed/fraction. Holding out optimization
views does not hide their panorama pixels from depth, segmentation or initial
point colors. This is a same-panorama reconstruction experiment.

Reuse fingerprints include source code contents (including dirty files), input
panorama hash and preprocessing manifests. This does not hash all multi-GB PLY
payloads or model checkpoints: shared preprocessing must remain immutable.
Selected run directories are written to `selected_runs.json`; aggregation uses
that manifest, or the latest attempt per scene directory for historical runs.
Attempts are not pooled as independent replicates. Paired fidelity requires the
same experiment, scene, run and view IDs. Failed/unavailable requested metrics
make a run partial and prevent automatic successful-run reuse.

`peak_training_rss_bytes` is a sampled sum of RSS over the training process tree.
It may count shared pages more than once and is neither unique physical memory
nor a separate GPU-VRAM counter. Sampling can miss short peaks. The parent and
child resource CSV writers use file locking. CPU percentages without a valid
sampling interval are null; stage CPU seconds cover the recording process, not
all subprocess CPU time.

Rendering cold-start time is the first render for that renderer instance, not
a fresh-process or cold-driver measurement (reconstruction evaluation can have
already warmed Metal). It excludes PLY loading. Exactly the configured number
of additional warm-up frames is discarded before steady-state timings.

Instance edit locality is unavailable when only a multi-instance group mask is
available. Layer deletion is compared against the unrefined merged scene,
avoiding a confound from combining refined before-images with unrefined layers.

See [the audit](BENCHMARK_AUDIT.md) for validation and the treatment of historical
results. Do not combine pre-correction and post-correction training-time/RAM
results: the trainer's SH allocation and frozen-parameter behavior also changed.
