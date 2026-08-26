# `samplingDensity`

Compute square-root sampling-density compensation weights for a 2D non-Cartesian trajectory.

```julia
samplingDensity(
    tr::AbstractArray{T,2},
    shape::Tuple,
) where T
```

## Arguments

- `tr`: two-dimensional k-space trajectory.
- `shape`: reconstruction matrix size.

## Returns

- `weights::Vector{Complex{T}}`: square-root density weights compatible with `recon_HOOp`.

The implementation normalizes the trajectory for the NFFT plan, evaluates iterative sampling-density compensation, and returns its square root. Consequently, `recon_HOOp` forms the weighted normal term $A^H W^H W A$.

## Example

```julia
weight = samplingDensity(
    reshape(kspha[2:3, :, :], 2, :),
    (grid.nX, grid.nY),
)
```

Use the same trajectory convention and the same returned weights when comparing encoding implementations.

[Source: `SampleDensity.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/SampleDensity/SampleDensity.jl)
