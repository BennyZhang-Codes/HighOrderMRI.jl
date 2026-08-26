# Reconstruction metrics

HighOrderMRI provides explicit metric names that distinguish raw complex accuracy from globally aligned or magnitude-only comparisons.

## `complex_alignment_scale`

```julia
complex_alignment_scale(reconstruction, reference)
```

Returns the least-squares complex scalar that minimizes the global difference between the scaled reconstruction and the reference.

## `raw_complex_nrmse`

```julia
raw_complex_nrmse(reconstruction, reference)
```

Computes

$$
\frac{\lVert x-x_{\mathrm{ref}}\rVert_2}{\lVert x_{\mathrm{ref}}\rVert_2}
$$

without intensity or phase alignment. This is the primary complex accuracy metric used by the validation guide.

## `aligned_complex_nrmse`

```julia
aligned_complex_nrmse(
    reconstruction,
    reference;
    scale=complex_alignment_scale(reconstruction, reference),
)
```

Computes complex NRMSE after one global least-squares complex scale.

## `magnitude_nrmse`

```julia
magnitude_nrmse(reconstruction, reference; align=false)
```

Computes NRMSE between magnitude images. Set `align=true` only for an explicitly aligned secondary analysis.

## `magnitude_ssim`

```julia
magnitude_ssim(
    reconstruction,
    reference;
    align=false,
    normalize=true,
)
```

Computes SSIM between magnitude images. With `normalize=true`, both images are divided by the peak magnitude of the reference using the same scale.

## Compatibility API

The legacy helpers

```julia
HO_MSE(reference, reconstruction; scale=false)
HO_RMSE(reference, reconstruction; scale=false)
HO_NRMSE(reference, reconstruction; scale=false)
HO_SSIM(reference, reconstruction; scale=false)
HO_img_scale(reference, reconstruction)
```

use **reference-first** argument order. Prefer the explicitly named functions above in new code.

See [Scientific validation strategy](/guide/validation) for the interpretation of raw, aligned, and magnitude-only metrics.

[Source: `ImageMetrics.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/utils/ImageMetrics.jl)
