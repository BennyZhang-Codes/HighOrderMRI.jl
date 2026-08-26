# `HighOrderOp`

Array-based explicit implementation of the field-aware encoding model on CPU or CUDA.

```julia
HighOrderOp(
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
    verbose=false,
) where T<:AbstractFloat
```

## Arguments

- `grid`: physical reconstruction grid.
- `kspha`: field coefficients with shape `(nTerm, nSam)`; `nTerm` is 9 or 16.
- `times`: ADC sampling times with length `nSam`.

## Keywords

- `fieldmap`: static off-resonance map.
- `csm`: complex coil-sensitivity maps.
- `mask`: spatial reconstruction mask.
- `recon_terms`: order-selection string.
- `k_nominal`: nominal first-order trajectory `(kx, ky, kz)`.
- `kspha_dt`: optional time derivative of the field coefficients; when supplied, the forward action evaluates the derivative product used by `FindDelay`.
- `nBlock`: number of sample blocks used to limit temporary memory.
- `arrayType`: `Array` for CPU or `CuArray` for CUDA array execution.
- `verbose`: print progress information.

## Returns

A linear operator with `nSam * nCha` rows and `prod(grid.matrixSize)` columns. Forward and adjoint use the same explicit signal model and symmetric normalization described in [Expanded encoding model](/theory/encoding-model).

::: info Derivative path
When `kspha_dt` is supplied, only the forward action represents the derivative product. The adjoint remains the ordinary encoding adjoint; this configuration should therefore not be interpreted as a general derivative operator for adjointness testing.
:::

## Example

```julia
op = HighOrderOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    arrayType=Array,
)
```

[Source: `HighOrderOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/EncodingOperator/HighOrderOp.jl)
