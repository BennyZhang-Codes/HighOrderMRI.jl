# Reconstruction metrics

Compare images on the same grid, mask and intensity convention. Raw complex
NRMSE measures amplitude and phase differences; globally aligned metrics remove
one fitted complex scale, and magnitude metrics discard phase. Report which
choice you use. A small value against a reconstructed reference is not an
independent physical validation of that reference.

## A complete CPU example

This controlled example changes only a known global scale and phase. The aligned
error is near zero by construction; it demonstrates the metric definitions.

```julia
using HighOrderMRI

reference = reshape(ComplexF64.(1:256), 16, 16)
reconstruction = (1.2 * cis(0.2)) .* reference
metric_values = (
    raw = raw_complex_nrmse(reconstruction, reference),
    aligned = aligned_complex_nrmse(reconstruction, reference),
    magnitude = magnitude_nrmse(reconstruction, reference),
    ssim = magnitude_ssim(reconstruction, reference),
)
@assert metric_values.raw > 0
@assert metric_values.aligned < 1e-12
```

The newer functions take `(reconstruction, reference)`. The `HO_*` compatibility
functions below use **reference first**. They have different validation rules:
for example, normalized errors reject a zero reference norm, while unscaled
`HO_MSE` does not need a reference norm. No helper applies a mask internally.
SSIM copies magnitudes to CPU storage and uses the dependency's image/window
conventions; its default peak normalization uses the same scale for both images.

See [Scientific validation strategy](/guide/validation) for how to separate raw,
aligned and magnitude-only results.

## `complex_alignment_scale`

<!--@include: ../../.generated/api/complex-alignment-scale.md-->

## `raw_complex_nrmse`

<!--@include: ../../.generated/api/raw-complex-nrmse.md-->

## `aligned_complex_nrmse`

<!--@include: ../../.generated/api/aligned-complex-nrmse.md-->

## `magnitude_nrmse`

<!--@include: ../../.generated/api/magnitude-nrmse.md-->

## `magnitude_ssim`

<!--@include: ../../.generated/api/magnitude-ssim.md-->

## Compatibility API

These bindings remain available for existing scripts. `HO_img_scale` returns an
aligned image; `complex_alignment_scale` returns the scalar itself.

## `HO_img_scale`

<!--@include: ../../.generated/api/ho-img-scale.md-->

## `HO_MSE`

<!--@include: ../../.generated/api/ho-mse.md-->

## `HO_RMSE`

<!--@include: ../../.generated/api/ho-rmse.md-->

## `HO_NRMSE`

<!--@include: ../../.generated/api/ho-nrmse.md-->

## `HO_SSIM`

<!--@include: ../../.generated/api/ho-ssim.md-->
