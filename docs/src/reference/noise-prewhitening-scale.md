# `noise_prewhitening_scale_factor`

Use this correction only when acquisition and noise-prescan receiver conditions
are known. The dwell-time ratio does not by itself identify a scanner-specific
effective bandwidth correction. See the [noise model](/guide/coil-compression).

```julia
using HighOrderMRI
scale = noise_prewhitening_scale_factor(4e-6, 8e-6;
    receiver_bandwidth_ratio=1)
@assert scale == 0.5
```

This is arithmetic with illustrative inputs, not a calibration recommendation.

## API

<!--@include: ../../.generated/api/noise-prewhitening-scale-factor.md-->
