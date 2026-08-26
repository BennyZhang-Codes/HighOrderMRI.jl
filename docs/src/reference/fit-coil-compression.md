# `fit_coil_compression`

Fit a global SVD-based receive-coil compression transform, optionally after noise prewhitening.

```julia
fit_coil_compression(
    calibration_data;
    coil_dim=ndims(calibration_data),
    n_virtual_coils=nothing,
    energy_threshold=nothing,
    noise_covariance=nothing,
    prewhitening_scale_factor=1,
)
```

## Arguments

- `calibration_data`: complex calibration samples. All dimensions other than `coil_dim` are flattened into calibration observations.

## Keywords

- `coil_dim`: receive-channel dimension.
- `n_virtual_coils`: explicit retained rank.
- `energy_threshold`: alternatively, retain the smallest rank reaching the requested cumulative squared singular-value energy.
- `noise_covariance`: optional receive-noise covariance for Cholesky prewhitening.
- `prewhitening_scale_factor`: positive correction for dwell-time / effective-bandwidth differences.

Specify exactly one of `n_virtual_coils` and `energy_threshold`.

## Returns

A [`CoilCompressionTransform`](/reference/coil-compression-transform). When `noise_covariance` is supplied, the returned matrix combines whitening and compression.

## Example

```julia
transform = fit_coil_compression(
    calibration_data;
    coil_dim=2,
    n_virtual_coils=12,
    noise_covariance,
    prewhitening_scale_factor=scale,
)
```

The truncated-SVD optimality statement applies to this single global linear transform in the whitened calibration space; it does not imply equivalence to spatially varying GCC. See [Coil compression](/guide/coil-compression) for the derivation and literature context.

[Source: `CoilCompression.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/CoilCompression.jl)
