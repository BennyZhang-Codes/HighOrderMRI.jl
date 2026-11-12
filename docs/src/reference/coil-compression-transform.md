# `CoilCompressionTransform`

This object records one fitted global coil transform and its calibration
metadata. Prefer [`fit_coil_compression`](/reference/fit-coil-compression) to
manual construction, and reuse the same object for data and sensitivity maps.
See the [coil compression workflow](/guide/coil-compression).

## Example

```julia
using HighOrderMRI, Random

calibration = randn(MersenneTwister(21), ComplexF32, 64, 4)
transform = fit_coil_compression(calibration; n_virtual_coils=2)
@assert size(transform) == (4, 2)
transform.retained_energy
```



## API

<!--@include: ../../.generated/api/coil-compression-transform.md-->
