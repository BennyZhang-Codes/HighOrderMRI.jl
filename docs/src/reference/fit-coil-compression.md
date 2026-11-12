# `fit_coil_compression`

Fit one global transform from a calibration subset, then apply it to all
samples and coil maps. Retained calibration energy is not an image-error bound
or a guarantee about parallel-imaging noise amplification. The
[derivation and references](/guide/coil-compression#global-svd-compression)
describe the scope of truncated-SVD optimality.

## Example

```julia
using HighOrderMRI, Random

rng = MersenneTwister(21)
calibration = randn(rng, ComplexF64, 128, 4)
noise = randn(rng, ComplexF64, 256, 4)
covariance = estimate_noise_covariance(noise; coil_dim=2)
transform = fit_coil_compression(calibration;
    coil_dim=2, n_virtual_coils=2, noise_covariance=covariance)
@assert size(transform, 2) == 2
```



## API

<!--@include: ../../.generated/api/fit-coil-compression.md-->
