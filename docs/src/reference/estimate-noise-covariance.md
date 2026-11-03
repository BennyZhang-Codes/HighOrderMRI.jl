# `estimate_noise_covariance`

Estimate receive-noise covariance before fitting compression. The noise sample
ordering, gains and receiver conditions must correspond to the acquired data.
The [coil guide](/guide/coil-compression#receive-noise-whitening) explains the
row-vector covariance convention and bandwidth correction.

## Example

```julia
using HighOrderMRI, Random, LinearAlgebra

noise = randn(MersenneTwister(21), ComplexF64, 128, 4)
covariance = estimate_noise_covariance(noise; coil_dim=2)
@assert size(covariance) == (4, 4)
@assert ishermitian(covariance)
```



## API

<!--@include: ../../.generated/api/estimate-noise-covariance.md-->
