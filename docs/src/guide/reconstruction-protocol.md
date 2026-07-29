# Reconstruction protocol

!!! info "Protocol status"
    Phase 0 convention freeze · Version 1 · Last updated 2026-07-24

This document defines the conventions that must remain fixed when comparing
nominal, conventional, explicit high-order, and low-rank high-order
reconstructions. Any intentional change requires a protocol-version increment,
an explanation in the experiment manifest, and regeneration of affected
results.

## 1. Encoding model

For receive channel `c`, sample `j`, and masked voxel `v`, the explicit
forward model is

```text
y[j,c] = 1/sqrt(nVox) *
         sum_v x[v] * csm[v,c] *
         exp(+2πim * phase[j,v])
```

with

```text
phase[j,v] = times[j] * fieldmap[v] +
             sum_term kspha[term,j] * basis[v,term].
```

The phase is measured in cycles before multiplication by `2π`. The forward
phase sign is positive. The adjoint uses the conjugate phase.

`HighOrderLowRankOp` preserves this model. Its higher-order and off-resonance
phase matrix is approximated by rSVD and a global shared spatial basis; its
first-order terms are evaluated by AbstractNFFTs, and its spatially constant
zeroth-order coefficient is folded into the temporal factor `q`.

## 2. Spatial grid

For an axis of length `N` and voxel spacing `Δ`, `Grid` uses

```text
r[i] = (i - (N + 1)/2) * Δ,  i = 1,...,N.
```

Consequences:

- odd dimensions contain a voxel at zero;
- even dimensions are symmetric about zero with centres at half-integers;
- `grid.x`, `grid.y`, and `grid.z` use Julia column-major voxel ordering;
- `Δx`, `Δy`, and `Δz` use the same physical-length unit as the spatial
  spherical-harmonic basis.

Axis exchange and reversal must be applied before creating every compared
operator. Images, masks, field maps, and sensitivity maps must undergo the
same permutation/reversal.

## 3. Spherical-harmonic term order

The production term order is:

| Index | Spatial basis |
|---:|---|
| 1 | `1` |
| 2 | `x` |
| 3 | `y` |
| 4 | `z` |
| 5 | `x*y` |
| 6 | `z*y` |
| 7 | `3*z^2 - (x^2 + y^2 + z^2)` |
| 8 | `x*z` |
| 9 | `x^2 - y^2` |
| 10–16 | third-order terms defined in `basisfunc_spha` |

The coefficients in `kspha` must be expressed in reciprocal units consistent
with the corresponding spatial basis so that every product contributes phase
in cycles.

## 4. Input shapes and vector ordering

### Explicit single-dynamic operator

- `kspha`: `(nTerm, nSam)`
- `times`: `(nSam,)`
- `fieldmap`: `(nX, nY, nZ)` or `(nX, nY)` when `nZ == 1`
- `csm`: `(nX, nY, nZ, nCha)` or `(nX, nY, nCha)` when `nZ == 1`
- `mask`: `(nX, nY, nZ)` or `(nX, nY)` when `nZ == 1`
- output layout: `(nSam, nCha)`, vectorized with samples first

### Low-rank dynamic operator

- `kspha`: `(nTerm, nSam, nDyn)`
- `times`: `(nSam, nDyn)`
- spatial inputs: same convention as the explicit operator
- output layout: `(nSam, nDyn, nCha)`, vectorized with samples first,
  then dynamics, then channels

The input arrays are read-only model inputs. `prep_kspha` returns a prepared
copy and must not mutate the caller's coefficients.

## 5. Reconstruction-term selection

For `nTerm == 9`, `recon_terms` has three binary digits:

```text
zeroth order, first order, second order
```

For `nTerm == 16`, it has four binary digits:

```text
zeroth order, first order, second order, third order
```

Rules:

- disabled zeroth-, second-, or third-order coefficients are set to zero;
- disabling the first-order measured coefficients replaces them with
  `k_nominal`;
- omitting `recon_terms` enables all available orders;
- only `0` and `1` are valid digits.

The primary paper comparisons are:

| Condition | First order | Static B0 | Dynamic higher order |
|---|---|---|---|
| Nominal | nominal | off | off |
| Conventional corrected | measured/GIRF | on | off |
| Explicit HighOrder | measured/GIRF | on | on |
| LowRank HighOrder | measured/GIRF | on | on, low-rank approximation |

The exact strings used for every condition must be stored in the experiment
configuration.

## 6. NFFT convention

AbstractNFFTs evaluates its forward transform with a negative Fourier
exponent on integer grid indices. To reproduce the explicit positive-phase
model, the physical first-order coefficient for axis `i` is converted to the
NFFT node

```text
node_i = -k_i * Δ_i.
```

Each axis uses its own spacing. A common minimum spacing is not valid for an
anisotropic grid.

`nfft_center_correction=true` is the publication convention:

- even dimensions receive the required half-voxel phase correction;
- odd dimensions receive no centre correction.

`nfft_center_correction=false` is legacy/debug behavior and must not be used
for primary comparisons.

## 7. Operator normalization

Both explicit and LowRank forward/adjoint operators use

```text
1 / sqrt(nVox)
```

where `nVox = sum(mask)`. In `HighOrderLowRankOp`, this factor, the zeroth-order
temporal phase, and the NFFT centre phase are folded into `q` once. No
additional post-reconstruction normalization may be applied to make two
operators agree.

## 8. LowRank parameters

The following quantities must be reported separately:

- `L_rank`: local per-dynamic rSVD truncation rank;
- `shared_rank`: final global shared spatial rank;
- `shared_basis_tol`: adaptive shared-basis relative-error tolerance;
- `rsvd_seed`;
- `rsvd_oversample`;
- `rsvd_finalize`;
- `rsvd_backend`;
- `rsvd_distribution`.

The default seed schedule is deterministic: dynamic `d` uses
`rsvd_seed + d - 1`.

## 9. Solver freeze

For a comparison, all methods must use identical:

- data and weights;
- sensitivity maps, field map, mask, and field coefficients;
- reconstruction grid;
- initial image;
- regularizer and regularization strength;
- solver and numerical precision;
- iteration count and/or stopping criterion.

Report both fixed-iteration timing and time to a common stopping criterion
when evaluating performance.

## 10. Primary error metrics

The primary LowRank-vs-explicit metric is raw complex relative error:

```text
norm(x_lowrank - x_explicit) / norm(x_explicit).
```

The primary simulation metric is raw complex NRMSE against `x_truth`.

No global complex scale, phase alignment, intensity rescaling, registration,
or cropping may be applied to a primary metric. Aligned magnitude/phase
metrics may be reported only as explicitly labelled secondary analyses.

Also record:

- magnitude NRMSE;
- masked phase RMSE with a fixed magnitude threshold;
- SSIM;
- data-consistency residual;
- predefined ROI errors;
- adjoint identity error.

## 11. Timing and memory

- Separate Julia/CUDA compilation from steady-state execution.
- Synchronize participating GPUs before and after timed regions.
- Record setup, forward, adjoint, normal, CG, and end-to-end times.
- Use at least five steady-state repetitions for final benchmarks and report
  median and IQR.
- Measure peak GPU memory externally through NVML or `nvidia-smi`.
- Record GPU UUIDs, driver, CUDA version, Julia threads, and GPU workers.
- Exclude runs affected by Xid errors, device loss, server restart, or
  competing GPU processes.

## 12. Required provenance

Every result row must identify:

- run and dataset IDs;
- protocol/config version;
- Git SHA and dirty-worktree state;
- Julia, CUDA, package, and NFFT backend versions;
- matrix size, resolution, samples, dynamics, channels, and terms;
- field source;
- method and all method-specific parameters;
- solver settings;
- GPU IDs and UUIDs;
- timings, memory, convergence, and image metrics.

Use `experiments/configs/phase0_reference.toml` as the starting manifest.

## 13. Change control

Changes to any convention above require:

1. incrementing `protocol_version`;
2. adding or updating a regression test;
3. documenting why the change is scientifically necessary;
4. identifying which previous results are invalidated;
5. regenerating all invalidated primary results.
