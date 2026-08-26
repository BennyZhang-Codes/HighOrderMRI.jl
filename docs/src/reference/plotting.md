# Plotting helpers

HighOrderMRI exports lightweight plotting helpers for reconstruction images, field coefficients, gradients, and derived field maps.

```julia
plt_plot(...)
plt_scatter(...)
plt_image(...)
plt_B0map(...)
plt_kspha(...)
plt_ksphas(...)
plt_bfield(...)
plt_bfield_com(...)
plt_grad(...)
mosaic(...)
```

These functions are convenience interfaces around the package plotting stack and are intended for interactive inspection and figure preparation. Quantitative validation should use the numerical metrics and frozen comparison procedures described in [Scientific validation strategy](/guide/validation) rather than visual assessment alone.

## Example

```julia
plt_image(abs.(image); title="Reconstruction")
plt_kspha(kspha)
```

[Source: `src/plot`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/tree/docs-modern-ui/src/plot)
