# Plotting helpers

The plotting helpers return Matplotlib figures through PyPlot. Figure widths and
heights are generally specified in approximately **centimetres**, converted to
inches internally; font sizes use points. Image-mosaic width means width per
tile column, and `plt_B0map` reserves additional colorbar space.

Use numerical metrics for quantitative comparisons. Plot appearance alone does
not establish reconstruction accuracy or correct physical coordinate mapping.

## A complete CPU example

This example saves two PNGs in a temporary directory and closes both figures.
Set `MPLBACKEND=Agg` before loading the package when running without a display.

```julia
ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")
using HighOrderMRI
import PyPlot

output_dir = mktempdir()
image = reshape(collect(range(0.0, 1.0; length=64)), 8, 8)
image_path = joinpath(output_dir, "image.png")
fig = plt_image(image; title="Numerical example", width=6, vmin=0, vmax=1)
try
    fig.savefig(image_path, dpi=120, bbox_inches="tight")
finally
    PyPlot.close(fig)
end

# Encoding coefficients are term-first and use cycles; plotting is sample-first
# and its fixed phase labels use radians. Select one readout before this conversion.
phase_cycles = zeros(Float64, 9, 32)
phase_cycles[2, :] .= range(-1.0, 1.0; length=32)
phase_radians = (2pi) .* permutedims(phase_cycles)
phase_path = joinpath(output_dir, "phase.png")
phase_fig = plt_kspha(phase_radians, 1e-5)
try
    phase_fig.savefig(phase_path, dpi=120, bbox_inches="tight")
finally
    PyPlot.close(phase_fig)
end
@assert isfile(image_path) && isfile(phase_path)
println(output_dir)
```

## Shapes, units and existing limitations

- `plt_kspha`/`plt_ksphas` take `(nSample,nTerm)` matrices. Their axes assume
  radians per spatial-order unit and plot the input unchanged. Their time axes
  start at one dwell, whereas interpolation helpers use a zero time origin.
- `plt_bfield`/`plt_bfield_com` plot the first nine columns with fixed field-basis
  labels and mT/m^p units. They do not convert phase into a magnetic field.
- The GIRF forms of `plt_grad` accept Hz/m and convert displayed values to mT/m;
  the geometry forms document their own coordinate and unit contracts below.
- In `plt_image`, specifying only one of `vmin` and `vmax` currently recomputes
  both from quantiles. Supply both to retain explicit limits.
- `plt_scatter` currently accepts but does not forward `marker`, `marker_size`
  or `marker_linewidth`. Their values do not control the plotted markers.
- `plt_images` and `plt_trajs` are declared exports without definitions (PG-11).
  Use the defined `plt_image` overloads. The loaded, unexported
  `HighOrderMRI.plt_traj` expects `(nPoint,3)` columns `[kx,ky,kz]` and draws only
  kx/ky; it is not the undefined `plt_trajs` export.

These are documented limits of the current implementation, not fixes introduced
by the documentation migration. The unreferenced `src/plot/plt_plotscatter.jl`
file is not loaded and is not presented as an available API.

## `plt_plot`

<!--@include: ../../.generated/api/plt-plot.md-->

## `plt_scatter`

<!--@include: ../../.generated/api/plt-scatter.md-->

## `plt_image`

<!--@include: ../../.generated/api/plt-image.md-->

## `plt_B0map`

<!--@include: ../../.generated/api/plt-b0map.md-->

## `plt_kspha`

<!--@include: ../../.generated/api/plt-kspha.md-->

## `plt_ksphas`

<!--@include: ../../.generated/api/plt-ksphas.md-->

## `plt_bfield`

<!--@include: ../../.generated/api/plt-bfield.md-->

## `plt_bfield_com`

<!--@include: ../../.generated/api/plt-bfield-com.md-->

## `plt_grad`

<!--@include: ../../.generated/api/plt-grad.md-->

## `mosaic`

<!--@include: ../../.generated/api/mosaic.md-->
