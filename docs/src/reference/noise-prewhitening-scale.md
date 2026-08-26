# `noise_prewhitening_scale_factor`

Compute the dimensionless correction used when a noise acquisition and MRI acquisition have different dwell times or effective receiver bandwidths.

```julia
noise_prewhitening_scale_factor(
    acquisition_dwell_time,
    noise_dwell_time;
    receiver_bandwidth_ratio=1,
)
```

## Arguments

- `acquisition_dwell_time`: dwell time of the MRI acquisition.
- `noise_dwell_time`: dwell time of the noise acquisition.

Both must use the same time unit.

## Keywords

- `receiver_bandwidth_ratio`: scanner- or receiver-specific effective noise-bandwidth correction.

## Returns

The positive scale factor

$$
s = \frac{T_{\mathrm{acq}}}{T_{\mathrm{noise}}} R_{\mathrm{BW}}.
$$

When a covariance `Psi_noise` is supplied to coil-compression fitting, the corrected acquisition covariance is interpreted as `Psi_noise / s`.

## Example

```julia
scale = noise_prewhitening_scale_factor(
    acquisition_dwell,
    noise_dwell;
    receiver_bandwidth_ratio=1,
)
```

[Source: `CoilCompression.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/CoilCompression.jl)
