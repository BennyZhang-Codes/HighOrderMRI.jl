# Reconstruction protocol

::: info Protocol status
Phase 0 convention freeze · Numerical convention version 1 · Documentation revised 2026-08-25
:::

This document defines the conventions that must remain fixed when comparing nominal, conventional, explicit high-order, and low-rank high-order reconstructions. Any intentional change to a numerical convention requires a protocol-version increment, an explanation in the experiment manifest, and regeneration of affected results. Editorial clarification alone does not change the convention version.

Symbols follow the [unified notation](/theory/symbols).

## 1. Encoding model

For receive channel $c$, sample $j$, and masked voxel $v$, the explicit single-dynamic implementation computes the phase in cycles as

$$
\phi_{jv}
=
t_j\,\Delta f_{0,v}
+
\sum_{p=1}^{N_{\mathrm{term}}}
k_{p,j}\,b_{p,v},
$$

where Julia row $p=1$ is the spatially constant zeroth-order basis $b_{1,v}=1$. The forward model is

$$
y_{jc}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
x_v C_{vc}
\exp\!\left(+i2\pi\phi_{jv}\right).
$$

This expression mirrors the production `HighOrderOp` and `HighOrderKernelOp`: static off-resonance contributes `times .* fieldmap`, and all selected dynamic field terms contribute `bf * kspha`. The phase is in cycles before multiplication by $2\pi$. The forward sign is positive and the adjoint uses the conjugate phase and $C_{vc}^*$.

`HighOrderLowRankOp` targets the same model. It sends the active first-order terms to AbstractNFFTs, approximates static off-resonance plus residual spatial phase with per-dynamic rSVD and the incremental shared spatial basis, and folds the spatially constant zeroth-order phase into the sample-domain factor `q`.

## 2. Spatial grid

For an axis of length $N$ and voxel spacing $\Delta$, `Grid` uses

$$
r[i]
=
\left(i-\frac{N+1}{2}\right)\Delta,
\qquad
i=1,\ldots,N.
$$

Consequences:

- odd dimensions contain a voxel centre at zero;
- even dimensions are symmetric about zero with centres at half-integer multiples of $\Delta$;
- `grid.x`, `grid.y`, and `grid.z` follow Julia column-major voxel ordering;
- $\Delta_x$, $\Delta_y$, and $\Delta_z$ use the same physical-length unit as the spatial solid-harmonic basis.

Axis exchange and reversal must be applied before creating every compared operator. Images, masks, field maps, and sensitivity maps must undergo the same permutation/reversal.

## 3. Solid-harmonic term order

The production term order is:

| Index | Spatial basis |
|---:|---|
| 1 | $1$ |
| 2 | $x$ |
| 3 | $y$ |
| 4 | $z$ |
| 5 | $xy$ |
| 6 | $zy$ |
| 7 | $3z^2-(x^2+y^2+z^2)$ |
| 8 | $xz$ |
| 9 | $x^2-y^2$ |
| 10–16 | third-order terms listed in the [expanded encoding model](/theory/encoding-model#solid-harmonic-basis-order) |

The coefficients in `kspha` must be expressed in reciprocal units consistent with the corresponding spatial basis so that every product contributes phase in cycles.

## 4. Input shapes and vector ordering

### Explicit single-dynamic operator

```julia
size(kspha) == (nTerm, nSam)
size(times) == (nSam,)
size(fieldmap) in ((nX, nY, nZ), (nX, nY))  # 2D form only when nZ == 1
size(csm) in ((nX, nY, nZ, nCha), (nX, nY, nCha))
size(mask) in ((nX, nY, nZ), (nX, nY))
```

Its logical output has shape `(nSam, nCha)` and vectorizes with samples first.

### Low-rank dynamic operator

```julia
size(kspha) == (nTerm, nSam, nDyn)
size(times) == (nSam, nDyn)
size(data) == (nSam, nDyn, nCha)
```

The low-rank output vector is compatible with `vec(data)`: samples vary fastest, then dynamics, then channels. `prep_kspha` returns a prepared copy and does not mutate the caller's coefficients.

## 5. Reconstruction-term selection

For `nTerm == 9`, `recon_terms` has three binary digits ordered as **zeroth, first, second**. For `nTerm == 16`, it has four binary digits ordered as **zeroth, first, second, third**.

Rules:

- disabled zeroth-, second-, or third-order coefficients are set to zero;
- disabling the first-order measured coefficients replaces rows 2–4 with `k_nominal`;
- omitting `recon_terms` enables all available orders;
- only `0` and `1` are valid digits.

The primary comparison conditions are:

| Condition | First order | Static $B_0$ | Dynamic higher order |
|---|---|---|---|
| Nominal | nominal | off | off |
| Conventional corrected | measured/GIRF | on | off |
| Explicit HighOrder | measured/GIRF | on | on |
| LowRank HighOrder | measured/GIRF | on | on, low-rank approximation |

The exact strings used for every condition must be stored in the experiment configuration.

## 6. NFFT convention

AbstractNFFTs evaluates its forward transform with a negative Fourier exponent on integer-centred grid indices. To reproduce the explicit positive-phase model, the physical first-order coefficient for axis $i$ is converted to

$$
\xi_{i,jd}
=
-k_{i,jd}\Delta_i.
$$

Each axis uses its own spacing. A common minimum spacing is not valid for an anisotropic grid.

In 3D, the $x$, $y$, and $z$ first-order terms are represented by the NFFT. In 2D, only $x$ and $y$ are passed to the 2D NFFT; the $z$ first-order term remains in the residual phase representation and is evaluated using `grid.z`. For the default centred single-slice grid, `grid.z = 0`, so that residual contribution vanishes.

`nfft_center_correction=true` is the fixed reference-comparison convention. For an even active dimension, the physical voxel centres lie half a voxel above the integer-centred NFFT coordinates. The implementation therefore folds

$$
\exp\!\left(i\pi k_{i,jd}\Delta_i\right)
$$

into the temporal coefficient for that axis. Odd dimensions receive no centre correction. `nfft_center_correction=false` is legacy/debug behavior and must not be used for the primary reference comparison.

## 7. Operator normalization

Both explicit and low-rank forward/adjoint operators use

$$
\frac{1}{\sqrt{N_v}},
\qquad
N_v
=
\sum_v \mathbf 1_{\mathrm{mask}}(v).
$$

In `HighOrderLowRankOp`, this normalization, the zeroth-order temporal phase, and the NFFT centre correction are folded into `q` once. No additional post-reconstruction normalization may be applied solely to force agreement between operators.

## 8. Low-rank parameters

The following quantities must be reported separately:

- `L_rank`: local per-dynamic rSVD truncation rank;
- `shared_rank`: final incremental shared spatial rank;
- `shared_basis_tol`: adaptive incremental shared-basis relative-error tolerance;
- `shared_rank_max`: hard cap on the accumulated shared rank;
- `rsvd_seed`;
- `rsvd_oversample`;
- `rsvd_finalize`;
- `rsvd_backend`;
- `rsvd_distribution`.

The default seed schedule is deterministic:

$$
s_d
=
s_0+d-1.
$$

The shared-basis implementation is incremental. Coefficients of an earlier dynamic are not recomputed when later dynamics append new basis columns; those new rows are zero-padded for the earlier dynamic. Consequently, the stored final coefficient block should not be described as the exact projection $S^H\widetilde V_d$ onto the completed basis. See [Low-rank shared subspace](/theory/low-rank) for the implementation-accurate derivation.

## 9. Solver freeze

For a comparison, all methods must use identical:

- data and weights;
- sensitivity maps, field map, mask, and field coefficients;
- reconstruction grid;
- initial image;
- regularizer and regularization strength;
- solver and numerical precision;
- iteration count and/or stopping criterion.

Report both fixed-iteration timing and time to a common stopping criterion when evaluating performance.

## 10. Primary error metrics

The primary LowRank-vs-explicit metric is raw complex relative error:

$$
\varepsilon_{\mathrm{rel}}
=
\frac{\lVert x_{\mathrm{lowrank}}-x_{\mathrm{explicit}}\rVert_2}
{\lVert x_{\mathrm{explicit}}\rVert_2}.
$$

The primary simulation metric is raw complex NRMSE against `x_truth`.

No post-hoc global complex scale, phase alignment, intensity rescaling, registration, or crop selected to improve agreement may be applied to a primary metric. A predefined common FOV, mask, crop, or ROI is allowed when it is specified before method comparison and applied identically to every condition. Aligned or magnitude-only metrics may be reported as explicitly labelled secondary analyses.

Also record, when relevant:

- magnitude NRMSE;
- masked phase RMSE with a fixed magnitude threshold;
- SSIM;
- data-consistency residual;
- predefined ROI errors;
- adjoint identity error.

## 11. Timing and memory

- Separate Julia/CUDA compilation from steady-state execution.
- Synchronize participating GPUs before and after timed regions.
- Record setup, forward, adjoint, normal, solver, and end-to-end times.
- Use at least five steady-state repetitions for final benchmarks and report median and IQR.
- Measure peak GPU memory with a documented method that has sufficient temporal resolution, such as allocator statistics or NVML sampling. Coarse `nvidia-smi` polling may miss short-lived peaks and should be identified as an approximation when used.
- Record GPU UUIDs, driver, CUDA version, Julia threads, and GPU workers.
- Exclude runs affected by Xid errors, device loss, server restart, or competing GPU processes.

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

Changes to any numerical convention above require:

1. incrementing `protocol_version`;
2. adding or updating a regression test;
3. documenting why the change is scientifically necessary;
4. identifying which previous results are invalidated;
5. regenerating all invalidated primary results.
