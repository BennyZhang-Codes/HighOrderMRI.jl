# `compress_coils`

Use this convenience function when the complete data array is the calibration
input. For a separate calibration subset, use
[`fit_coil_compression`](/reference/fit-coil-compression) followed by
[`apply_coil_compression`](/reference/apply-coil-compression). The returned
arrays use the same fitted transform.

## Example

```julia
using HighOrderMRI, Random

data = randn(MersenneTwister(21), ComplexF32, 64, 4)
csm = ones(ComplexF32, 8, 8, 1, 4)
data_cc, csm_cc, transform = compress_coils(data, csm;
    data_coil_dim=2, csm_coil_dim=4, n_virtual_coils=2)
@assert size(data_cc, 2) == size(csm_cc, 4) == size(transform, 2)
```



## API

<!--@include: ../../.generated/api/compress-coils.md-->
