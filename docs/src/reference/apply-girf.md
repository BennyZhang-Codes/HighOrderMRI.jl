# `apply_girf`

Apply a frequency-domain GIRF model to nominal physical gradients.

```julia
apply_girf(
    G_DCS_nom::AbstractArray{T,N},
    Hw::AbstractArray;
    dim_spatial=1,
    dim_time=2,
    rbw=1.0,
) where {T,N}
```

A convenience overload also accepts `GIRFModel` directly.

## Arguments

- `G_DCS_nom`: nominal physical gradient array.
- `Hw`: GIRF transfer function with shape `(nFreq, 3, nOut)` and centered DC component.

## Keywords

- `dim_spatial`: dimension containing the three physical gradient channels.
- `dim_time`: time dimension.
- `rbw`: relative retained bandwidth. Values below 1 suppress the outer frequency band.

## Returns

An array with the same dimensional structure as the input, except that the physical-gradient dimension is replaced by `nOut` GIRF output channels.

If the required FFT length exceeds the GIRF frequency-grid length, the implementation converts the response to the time domain, inserts middle zero padding between its causal and non-causal portions, and transforms it back before applying the response.

## Example

```julia
predicted = apply_girf(
    nominal_gradients,
    model;
    dim_spatial=1,
    dim_time=2,
    rbw=1.0,
)
```

[Source: `apply_girf.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/GIRF/apply_girf.jl)
