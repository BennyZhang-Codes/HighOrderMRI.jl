# `HighOrderKernelOp`

Fused CUDA implementation of the same explicit field-aware signal model used by `HighOrderOp`.

```julia
HighOrderKernelOp(
    grid::Grid{T},
    kspha::AbstractArray{T,2},
    times::AbstractVector{T};
    fieldmap=zeros(T, grid.matrixSize...),
    csm=ones(Complex{T}, grid.matrixSize..., 1),
    mask=trues(grid.matrixSize...),
    recon_terms=nothing,
    k_nominal=kspha[2:4, :],
    kspha_dt=nothing,
    nBlock=50,
    arrayType=CuArray,
    gpus=[0],
    verbose=false,
) where T<:AbstractFloat
```

## Arguments

- `grid`: physical reconstruction grid.
- `kspha`: field coefficients `(nTerm, nSam)` with 9 or 16 terms.
- `times`: ADC sampling times.

## Keywords

- `fieldmap`, `csm`, `mask`, `recon_terms`, `k_nominal`: common encoding inputs.
- `kspha_dt`: optional derivative coefficients. The current derivative forward path falls back to the array-based derivative implementation.
- `nBlock`: block count used by the derivative path.
- `arrayType`: must be `CuArray`.
- `gpus`: zero-based CUDA device IDs.
- `verbose`: print progress information.

## Returns

A CUDA-backed explicit linear operator with `nSam * nCha` rows and `prod(grid.matrixSize)` columns.

With multiple GPUs, masked voxels are partitioned across devices. Forward partial signals are reduced on the host; adjoint voxel results are gathered into their original masked positions. The partition changes execution and reduction order, not the explicit phase model.

## Example

```julia
using CUDA

op = HighOrderKernelOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    gpus=[0, 1],
)
```

See [Multi-GPU execution](/guide/multi-gpu) for the voxel decomposition and reporting requirements.

[Source: `HighOrderKernelOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/EncodingOperator/HighOrderKernelOp.jl)
