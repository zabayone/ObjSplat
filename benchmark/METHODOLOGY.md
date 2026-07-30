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
minimum, maximum, quartiles, and a normal-approximation 95% confidence interval
when at least two observations exist. No significance test is run automatically.

Runs are `success`, `partial_success`, or `failed`. Completed CSV rows survive
failure. The failed stage, exception type/message, last resource sample, and
successfully detected outputs are retained. Optional or technically impossible
metrics remain null with an explanation; values are never fabricated.
