# `HighOrderLowRankOp`

Low-rank implementation of the residual high-order/off-resonance phase. The default uses per-dynamic randomized SVD and an incrementally constructed shared spatial basis; the experimental joint method constructs the shared basis directly from representative phase snapshots.

## Single-dynamic constructor

```julia
HighOrderLowRankOp(
    grid::Grid{T},
    kspha::AbstractArray{T,2},
    times::AbstractArray{T,1};
    kwargs...,
) where T<:AbstractFloat
```

The single-dynamic overload inserts a singleton dynamic dimension and delegates to the dynamic constructor.

## Dynamic constructor

```julia
HighOrderLowRankOp(
    grid::Grid{T},
    kspha::AbstractArray{T,3},
    times::AbstractArray{T,2};
    fieldmap=zeros(T, grid.matrixSize...),
    csm=ones(Complex{T}, grid.matrixSize..., 1),
    mask=trues(grid.matrixSize...),
    recon_terms=nothing,
    k_nominal=kspha[2:4, :, :],
    arrayType=Array,
    gpus=[0],
    L_rank=15,
    rsvd_seed=1234,
    rsvd_chunk=4096,
    rsvd_oversample=5,
    rsvd_finalize=:svd,
    rsvd_backend=:auto,
    rsvd_distribution=:auto,
    shared_rank_max=128,
    shared_basis_tol=T(1e-2),
    global_basis_tol=nothing,
    shared_basis_method=:rsvd,
    joint_snapshots=256,
    joint_samples=512,
    joint_roi_tol=nothing,
    joint_basis_report=nothing,
    normal_distribution=:single,
    nfft_center_correction=true,
    verbose=false,
) where T<:AbstractFloat
```

## Arguments

- `grid`: physical reconstruction grid.
- `kspha`: dynamic field coefficients `(nTerm, nSam, nDyn)`.
- `times`: ADC times `(nSam, nDyn)`.

## Spatial and encoding keywords

- `fieldmap`: static off-resonance map.
- `csm`: complex coil-sensitivity maps.
- `mask`: reconstruction mask; only masked voxels enter the low-rank setup.
- `recon_terms`: order-selection string.
- `k_nominal`: nominal first-order trajectory `(3, nSam, nDyn)`.
- `nfft_center_correction`: parity-aware correction aligning the NFFT grid with physical voxel centres.

## Low-rank keywords

For the default rSVD construction:

- `L_rank`: retained rank of each dynamic-specific rSVD.
- `rsvd_seed`: deterministic base seed; dynamic `d` uses `rsvd_seed + d - 1`.
- `rsvd_chunk`: voxel chunk size for the chunked backend.
- `rsvd_oversample`: additional randomized range directions.
- `rsvd_finalize`: `:svd` or small-Gram `:gram` finalization.
- `rsvd_backend`: `:chunked`, `:kernel`, or `:auto`.
- `rsvd_distribution`: `:single`, `:voxel`, or `:auto`.
- `shared_basis_tol`: incremental second-stage residual tolerance.
- `shared_rank_max`: hard upper bound on the accumulated shared rank.

The fused CUDA backend supports wide sketches in batches of at most 16
columns. The matrix-dimension bound `L_rank + rsvd_oversample <= min(nSam,nVox)`
still applies; this batch size does not truncate the final shared rank.

## Shared-basis parameters

::: info Experimental joint construction
Joint construction is explicitly enabled with `shared_basis_method=:joint`.
The default remains `:rsvd`. Both methods return the same operator type and
use the same factor layout, NFFT, and reconstruction interface.
:::

| Keyword | Default | Meaning |
|:--|:--|:--|
| `shared_basis_method` | `:rsvd` | Existing per-dynamic rSVD and incremental sharing; `:joint` explicitly selects joint snapshot construction. |
| `global_basis_tol` | `nothing` | Optional final recompression of the rSVD representation; measures additional relative Frobenius error. Must remain `nothing` for joint. |
| `joint_snapshots` | `256` | Representative phase-row budget $K$, used before rank selection and bounded by available phase geometries. |
| `joint_samples` | `512` | Initial fitting voxel count $J$; may double once if selection fails, within the fitting pool. |
| `joint_roi_tol` | `nothing` | Optional additional acceptance tolerance on sampled largest-$\lvert B_0\rvert$ voxels. Without it, excessive ROI error produces a warning. |
| `joint_basis_report` | `nothing` | Optional `Ref{Any}()` receiving selection and audit diagnostics. |

For `:joint`, `shared_rank_max=128` caps the final shared rank, and
`shared_basis_tol=T(1e-2)` bounds each dynamic's sampled random-voxel error
against the original phase model. Selection uses `0.9 * shared_basis_tol`;
the final audit uses the requested tolerance. The search tests rank 1 and
blocks of eight, including the last available rank. It does not guarantee the
smallest rank or a full-matrix/image error bound.

The existing `rsvd_seed=1234` controls joint sampling, while
`rsvd_chunk=4096` bounds temporary phase matrices. `L_rank` is retained as
metadata; neither it nor `rsvd_oversample` selects a local rank in this mode.
`rsvd_backend` and `rsvd_finalize` do not change the joint algorithm.
Use `rsvd_distribution=:auto` or `:single`; explicit `:voxel` is rejected.
Setup uses the primary GPU or the CPU. The existing
`normal_distribution=:channel` option can still distribute reconstruction.

The report contains `rank`, `rank_max`, `snapshots`, `sample_count`,
`tolerance`, `selection_tolerance`, `roi_tolerance`, `validation_errors`,
`validation_roi_errors`, `audit_errors`, `roi_errors`, `orthogonality`,
`passed`, and `stage`. It also records snapshot, fitting, validation and audit
indices, held-out profiles, `disjoint_times`, and `seed`. Errors are relative
Frobenius errors on sampled phase matrices, one value per dynamic.

Rank/sample exhaustion and failed final audits throw instead of returning
partial factors; those failures populate the report with `stage=:rank_limit`
or `:audit`. A report is not guaranteed for an earlier input or CUDA failure.
Float32 and Float64 are supported. The factors preserve the requested
precision; small CPU eigendecompositions/QR and phase-audit references use
Float64. See the [joint example](/guide/operators#automatic-joint-shared-basis)
and [mathematical scope](/theory/low-rank#direct-joint-shared-basis).

## Execution keywords

- `arrayType`: `Array` or `CuArray`.
- `gpus`: zero-based CUDA device IDs.
- `normal_distribution`: `:single` or `:channel` for the weighted normal operator.
- `verbose`: report setup and resource information.

## Returns

A `HighOrderLowRankOp` with dimensions `(nSam * nDyn * nCha, prod(grid.matrixSize))`, together with stored low-rank factors `q` and `basis` whose second dimension is the final shared rank `R`.

The default rSVD implementation uses streaming coefficient blocks: when later dynamics expand the shared basis, earlier coefficient blocks are zero-padded rather than recomputed. See [Low-rank shared subspace](/theory/low-rank) for the exact factorization and approximation-error interpretation.

## Example

```julia
using CUDA

op = HighOrderLowRankOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
    gpus=[0, 1, 2],
    L_rank=15,
    rsvd_finalize=:gram,
    shared_basis_tol=1f-2,
    normal_distribution=:channel,
)
```

[Source: `HighOrderLowRankOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/dev_jinyuan/src/EncodingOperator/HighOrderLowRankOp.jl)
