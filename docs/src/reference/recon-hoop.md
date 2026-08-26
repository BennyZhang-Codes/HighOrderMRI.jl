# `recon_HOOp`

Iterative reconstruction wrapper for a high-order encoding operator.

```julia
recon_HOOp(
    HOOp::HOOp{Complex{T}},
    Data::AbstractArray{Complex{T},2},
    weight::AbstractVector{Complex{T}},
    recParams::Dict;
    release_backend=true,
) where T<:AbstractFloat
```

## Arguments

- `HOOp`: a compatible high-order linear operator.
- `Data`: k-space matrix `(nSampleTotal, nCha)`.
- `weight`: square-root sample weights with length `nSampleTotal`.
- `recParams`: reconstruction settings passed to the MRIReco/RegularizedLeastSquares solver path.

For dynamic low-rank data,

```julia
nSampleTotal = nSam * nDyn
```

in the same ordering as `vec(data)` for data shaped `(nSam, nDyn, nCha)`.

## Keywords

- `release_backend`: when `true` (default), release a channel-distributed `HighOrderLowRankOp` normal backend after reconstruction, including when the solve exits with an exception.

## Returns

The reconstructed image reshaped to `recParams[:reconSize]` and copied to a CPU `Array`.

Internally, `weight` is applied to both data and encoding through the weighting operator, so the normal term is

$$
A^H W^H W A.
$$

## Example

```julia
image = recon_HOOp(
    op,
    ComplexF32.(data_matrix),
    ComplexF32.(weight_vector),
    rec_params,
)
```

See [Reconstruction workflow](/guide/reconstruction) for data layout, solver configuration, accuracy metrics, and reproducibility requirements.

[Source: `recon_HOOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/recon_HOOp.jl)
