# Scene benchmark audit — 2026-09-05

The active six-scene comparison and quality-scaling configurations, orchestration,
input splitting, monolithic baseline, evaluation renderer, image metrics,
resource instrumentation, comparison tables, aggregation, edit/mood diagnostics
and tests were reviewed. Existing user changes and historical result files were
preserved. No full six-scene experiment was launched.

## Corrected issues affecting thesis results

| Issue | Correction | Consequence for existing results |
| --- | --- | --- |
| Layered downsampling applied in both loader and trainer | Apply the configured fraction once; monolithic and layered now share that fraction | Retrain any fraction below 1.0; previous sweep support budgets were unequal |
| Baseline `--seed` did not seed training RNGs | Seed Python, NumPy and MLX in the subprocess | Rerun baseline for reproducible paired experiments |
| Baseline lacked original ERP height for large-cloud scale initialization | Carry ERP height from the common panorama | Retrain large baseline scenes for the corrected comparison |
| Run reuse ignored code changes | Fingerprint actual source content, input and preprocessing manifests | Old successes cannot validate corrected code |
| Stored split was trusted without checking layer frame files | Validate split partition, camera files, seed/fraction and leakage | Split validation is now mandatory for requested fidelity |
| Nearly black uint8 images could be interpreted as normalized floats | Normalize according to uint8 dtype, avoiding input mutation | Recompute affected image/edit metrics |
| Missing held-out results were returned without being saved | Persist explicit unavailable rows | Missing results can no longer disappear silently |
| Failed requested metrics could leave run status as success | Mark partial, retain reasons, return nonzero suite status | Historical success flags alone are insufficient |
| Aggregation pooled previous attempts and individual views | Select explicit runs; average fidelity per scene/run; pair identical view sets | Regenerate reports; preserve raw historical attempts |
| Cold frame counted against configured additional warmup | Run all additional warmups | Remeasure steady-state performance under the corrected protocol |
| Parent/child CSV writers could race | Lock CSV appends including header creation | Use new resource traces for thesis memory comparisons |
| Disabled fidelity/rendering marked cost-only sweep rows incomplete | Require only configured outputs | Cost-only sweep status now reflects its actual scope |
| Instance locality used a group mask; layer removal could compare refined vs unrefined geometry | Mark unavailable without an exact instance mask; use unrefined source for layer deletion | Recompute affected editing diagnostics |
| Monolithic PLY could be classified as a mood | Exclude it from mood topology analysis | Regenerate affected mood summaries |

Other changes remove stale in-process evaluation caches, report missing
segmentation coverage as null, retain valid child RSS samples if another child
exits, avoid bogus first-sample CPU percentages, record SPLAT environment
variables and LPIPS version, refresh comparison tables after LPIPS backfill,
and exclude stale optional PLY variants from a fresh training comparison.

## Checks on existing six-scene outputs

Read-only inspection of `benchmark_results/evaluation_six_scenes_complete`:

| Scene | Held-out views | Stored split/leakage check | Reconstruction rows marked success | Rendering rows marked success |
| --- | ---: | --- | ---: | ---: |
| indoor_1 | 17 | Passed | 34 | 3 |
| indoor_2 | 17 | Passed | 34 | 3 |
| indoor_3 | 17 | Passed | 34 | 3 |
| outdoor_1 | 17 | Passed | 34 | 4 |
| outdoor_2 | 17 | Passed | 34 | 4 |
| outdoor_3 | 17 | Passed | 34 | 4 |

These counts verify availability and split structure, not the correctness of
historical numerical results. In particular the earlier PLY SH-order correction
changes rendering, and the training memory changes make timing/RAM results from
old and new implementations incomparable without rerunning both methods.
The old experiment used fraction 1.0, so the double-downsampling issue does not
apply to that experiment; it affects sub-unit quality fractions.

## Validation

- Existing benchmark regressions plus new protocol tests for dark uint8 images,
  input immutability, split leakage, stale split seeds, strict run/view pairing,
  scene-level aggregation, explicit run selection, unavailable metric rows,
  cost-only sweeps and warmup counting.
- Real Metal integration regression on a synthetic 40-point scene: the layered
  and monolithic paths retain exactly 20 points at fraction 0.5. Two independent
  monolithic subprocesses with seed 42 and two training iterations produce
  identical parameter arrays after PLY reload.
- Python compilation and `git diff --check`.

## Interpretation for the thesis

Report panorama reconstruction fidelity, sampled process-tree RSS, training wall
time and steady-state renderer latency as separate outcomes. The 17 views are
not 17 independent scenes: the experimental sample comprises six selected
scenes, with one seed in the default configuration. Geometry and initial colors
still originate from the entire panorama, including held-out directions.

Layered and monolithic use explicit but different optimization budgets and losses
(masked per-layer supervision versus unmasked joint supervision). The comparison
measures these configured systems, not a strictly equal-compute ablation. The
two-scene quality sweep measures support-density cost and does not enable image
quality evaluation; it cannot by itself establish a quality/cost frontier.

The main comparison and quality sweep depend on existing shared preprocessing
under `benchmark_work/evaluation_main/<scene>`. Fresh environments need that
preprocessing prepared first. The corrected default suite is:

```bash
bash benchmark/run_evaluation_suite.sh
```

This runs expensive training. It was not executed during this audit. For thesis
numbers, rerun both representations with the corrected code and keep the run
manifests, complete configurations, system metadata and raw per-view CSVs.
