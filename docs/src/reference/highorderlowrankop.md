# `HighOrderLowRankOp`

Low-rank implementation of the residual high-order/off-resonance phase using per-dynamic randomized SVD and an incrementally constructed shared spatial basis.

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

- `L_rank`: retained rank of each dynamic-specific rSVD.
- `rsvd_seed`: deterministic base seed; dynamic `d` uses `rsvd_seed + d - 1`.
- `rsvd_chunk`: voxel chunk size for the chunked backend.
- `rsvd_oversample`: additional randomized range directions.
- `rsvd_finalize`: `:svd` or small-Gram `:gram` finalization.
- `rsvd_backend`: `:chunked`, `:kernel`, or `:auto`.
- `rsvd_distribution`: `:single`, `:voxel`, or `:auto`.
- `shared_basis_tol`: incremental second-stage residual tolerance.
- `shared_rank_max`: hard upper bound on the accumulated shared rank.

The fused CUDA rSVD kernel requires

$$
L_{\mathrm{rank}} + p \leq 32,
$$

where $p$ is `rsvd_oversample`.

## Execution keywords

- `arrayType`: `Array` or `CuArray`.
- `gpus`: zero-based CUDA device IDs.
- `normal_distribution`: `:single` or `:channel` for the weighted normal operator.
- `verbose`: report setup and resource information.

## Returns

A `HighOrderLowRankOp` with dimensions `(nSam * nDyn * nCha, prod(grid.matrixSize))`, together with stored low-rank factors `q` and `basis` whose second dimension is the final shared rank `R`.

The implementation uses streaming coefficient blocks: when later dynamics expand the shared basis, earlier coefficient blocks are zero-padded rather than recomputed. See [Low-rank shared subspace](/theory/low-rank) for the exact factorization and approximation-error interpretation.

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

[Source: `HighOrderLowRankOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/EncodingOperator/HighOrderLowRankOp.jl)
