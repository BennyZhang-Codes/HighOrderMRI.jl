# `compress_coils`

Fit one coil-compression transform from acquired data and apply it consistently to both data and coil-sensitivity maps.

```julia
compress_coils(
    data,
    csm;
    data_coil_dim=ndims(data),
    csm_coil_dim=ndims(csm),
    n_virtual_coils=nothing,
    energy_threshold=nothing,
    noise_covariance=nothing,
    prewhitening_scale_factor=1,
)
```

## Arguments

- `data`: complex acquired data used to fit the transform.
- `csm`: complex coil-sensitivity maps transformed with the same fitted matrix.

## Keywords

- `data_coil_dim`, `csm_coil_dim`: receive-channel dimensions of the two inputs.
- `n_virtual_coils` or `energy_threshold`: rank-selection criterion; specify exactly one.
- `noise_covariance`: optional covariance for noise-whitened fitting.
- `prewhitening_scale_factor`: optional dwell-time / receiver-bandwidth correction.

## Returns

```julia
compressed_data, compressed_csm, transform
```

where `transform` is a [`CoilCompressionTransform`](/reference/coil-compression-transform).

## Example

```julia
data_cc, csm_cc, transform = compress_coils(
    data,
    csm;
    data_coil_dim=2,
    csm_coil_dim=ndims(csm),
    n_virtual_coils=12,
    noise_covariance,
)
```

For calibration-region fitting, call [`fit_coil_compression`](/reference/fit-coil-compression) on the calibration view and apply the result to the full arrays with [`apply_coil_compression`](/reference/apply-coil-compression).

[Source: `CoilCompression.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/CoilCompression.jl)
