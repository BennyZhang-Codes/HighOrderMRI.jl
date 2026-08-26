# `estimate_noise_covariance`

Estimate the complex receive-noise covariance from noise-only samples.

```julia
estimate_noise_covariance(
    noise_data;
    coil_dim=ndims(noise_data),
    center=true,
)
```

## Arguments

- `noise_data`: complex noise-only samples acquired with the same receiver gains and channel ordering as the MRI data.

## Keywords

- `coil_dim`: array dimension containing receive channels.
- `center`: remove the per-coil sample mean before covariance estimation. With `center=true`, unbiased normalization by `nObservation - 1` is used; otherwise the input is assumed zero mean and normalized by `nObservation`.

## Returns

A Hermitian complex covariance matrix of size `(nCoil, nCoil)`.

Do not apply density compensation or coil compression before covariance estimation. If the noise acquisition and MRI data use different dwell times or effective receiver bandwidths, compute a correction with [`noise_prewhitening_scale_factor`](/reference/noise-prewhitening-scale).

## Example

```julia
noise_covariance = estimate_noise_covariance(
    noise_data;
    coil_dim=2,
)
```

[Source: `CoilCompression.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/CoilCompression.jl)
