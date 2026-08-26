# `CoilCompressionTransform`

Container for a fitted linear receive-coil transform.

```julia
CoilCompressionTransform
```

## Fields

- `compression_matrix`: matrix with size `(input_coils, virtual_coils)` applied from the right along the coil dimension.
- `singular_values`: complete singular-value spectrum of the fitted calibration matrix after optional prewhitening.
- `retained_energy`: fraction of squared singular-value energy retained by the selected virtual coils.
- `noise_covariance`: fitted receive-noise covariance, or `nothing`.
- `prewhitening_scale_factor`: dwell-time / receiver-bandwidth correction used during fitting.

If noise prewhitening is used, the stored transform combines whitening and compression in one matrix.

## Example

```julia
transform = fit_coil_compression(
    calibration_data;
    coil_dim=2,
    n_virtual_coils=12,
    noise_covariance,
)

size(transform)
transform.retained_energy
```

Use the same fitted transform for acquired data and coil-sensitivity maps.

[Source: `CoilCompression.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/CoilCompression.jl)
