# `apply_coil_compression`

Reuse a fitted transform for both measured data and sensitivity maps, even
when the coil dimensions occur at different positions. The result is a CPU
array; transfer it afterwards if the encoding backend needs CUDA storage.

## Example

```julia
using HighOrderMRI, Random

rng = MersenneTwister(21)
data = randn(rng, ComplexF32, 64, 4)
csm = ones(ComplexF32, 8, 8, 1, 4)
transform = fit_coil_compression(data; coil_dim=2, n_virtual_coils=2)
data_cc = apply_coil_compression(data, transform; coil_dim=2)
csm_cc = apply_coil_compression(csm, transform; coil_dim=4)
@assert size(data_cc) == (64, 2)
@assert size(csm_cc) == (8, 8, 1, 2)
```



## API

<!--@include: ../../.generated/api/apply-coil-compression.md-->
