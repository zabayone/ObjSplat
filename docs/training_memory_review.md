# Training memory review — 2026-09-05

The review targets the active Apple Silicon training path (`LayerPano.py`,
`mps_splat_backend.py`, and the bundled `mlx_gs` trainer). Existing worktree
changes were preserved. This is not an exhaustive audit of all vendored research
projects or the legacy CUDA backend.

## Are frozen parameters allocated?

Yes. Geometry, opacity, and appearance are inputs to rendering even when only
one group is optimized. Omitting these values changes the rendered scene.
The optimizer dictionary already excluded frozen groups from Adam; the trainer
now also passes only those selected groups to `mx.value_and_grad`. Frozen arrays
are captured as rendering inputs. Clamping, opacity reset, topology changes,
and PLY normalization no longer modify valid frozen attributes. Invalid values
can still be repaired by the existing export sanitation.

The custom Metal backward function still produces temporary derivatives for
screen-space means, covariance, opacity, and color together. The change avoids
requesting full model gradients for frozen groups; it does not eliminate every
intermediate derivative buffer inside that kernel.

The legacy per-point `frozen_mask` used by transfer/topology is a separate
mechanism from the `freeze_geometry`, `freeze_sh_coeffs`, and `freeze_opacity`
flags discussed here. It is not a per-point gradient mask.

## Unused SH storage

The current bundled renderer evaluates SH degrees 0 and 1 only. New training
previously allocated 16 RGB coefficients per Gaussian and scheduled degrees
2 and 3 even though the renderer ignored those orders. New scenes now allocate
4 RGB coefficients. Export pads higher orders with zeros for degree-3 PLY
compatibility. Existing PLY coefficients are retained when loading a refinement
input; they are not silently discarded.

For one million Gaussians in float32, the 36 removed scalar values per Gaussian
save 144 MB of parameter storage and 288 MB for Adam's two moments. A full dense
gradient would account for another 144 MB. These are tensor-size calculations,
not measured reductions in process peak RAM. Rendering buffers, images, caches,
input PLY arrays and temporary export padding remain additional costs.

The initial Gaussian container and scale initialization array are released once
parameters are assembled. The opacity floor reuses the immutable initial MLX
array instead of keeping a separate NumPy copy and another MLX allocation.

## Other fixes

- PLY SH loading now reverses channel-major serialization correctly, sorts fields
  numerically, and supports degrees 0, 1 and 3 instead of assuming 16 coefficients.
- Refinement with an implicit point cap no longer references an uninitialized
  `gaussians` variable; non-finite point filtering also filters initial parameters.
- Refinement starts with the renderer's full supported appearance instead of
  temporarily hiding direction-dependent colors.
- Learning-rate plateau reductions retain Adam moments and honor the minimum LR.
  Topology/opacity resets retain the current LR multiplier.
- Voxel uniqueness uses coordinate tuples instead of overflow-prone integer hashes;
  empty clouds honor `return_indices` and invalid voxel sizes are rejected.
- Removed unused CUDA mask helpers and the unused memory-report helper from
  `LayerPano.py`, and removed the obsolete SH-degree schedule and unused loader variable.

## Validation

- 13 tests in `tests/`, including five new training/serialization regressions.
- 19 benchmark tests.
- Real Metal smoke run: two ordinary training steps, two opacity-only refinement
  steps, and two appearance-only refinement steps; finite output and exact
  preservation of frozen arrays checked after PLY reload.
- `git diff --check`.

A full panorama training run and measured peak-RAM comparison were not performed.
