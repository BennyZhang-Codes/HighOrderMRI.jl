# `CoilCombineSOS`

Sum-of-squares produces a magnitude image from separate coil images. It
discards phase; use a sensitivity-aware encoding reconstruction when the complex
image is required.

## Example

```julia
using HighOrderMRI

coil_images = ones(ComplexF32, 8, 8, 1, 4)
magnitude_image = CoilCombineSOS(coil_images, 4)
@assert size(magnitude_image) == (8, 8, 1)
@assert all(magnitude_image .== 2f0)
```



## API

<!--@include: ../../.generated/api/coil-combine-sos.md-->
