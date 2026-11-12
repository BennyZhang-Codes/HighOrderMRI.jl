# Array and signal utilities

These helpers act on arrays and numeric values. They do not establish acquisition
units, geometry conventions or a reconstruction model automatically. Use host
arrays for the CPU example below, and consult the source-generated contracts
before relying on boundary behavior.

## A complete CPU example

```julia
using HighOrderMRI
using LinearAlgebra

# Input already represents Hz/m; no gyromagnetic conversion is done here.
gradient = [0.0 1.0; 2.0 1.0; 4.0 1.0; 6.0 1.0]
dt = 0.5                         # seconds for this numerical example
trajectory = grad2traj(gradient, dt)   # cycles/m, first row zero
interval_average = traj2grad(trajectory, dt)
trajectory_f32 = f32(trajectory)

image = reshape(Float64.(1:16), 4, 4)
cropped = get_center_crop(image, Int64(2), Int64(2))
scaled = normalization(image)
standardized = standardization(image)
mask = get_mask(image; threshold=8.0)
metadata = struct_to_dict((spacing=1.0, description="numerical example"))

spectrum = convert_fft(complex.(image))
restored = convert_ifft(spectrum)
@assert size(interval_average) == (3, 2)
@assert cropped == image[2:3, 2:3]
@assert norm(restored - image) / norm(image) < 1e-12
```

Differencing a trapezoidal cumulative integral returns interval-average gradients,
not the original endpoint samples. The transform pair above uses its own
Cartesian array normalization; it is not a substitute for a high-order encoding
operator or density-weighted reconstruction.

## Storage and precision

`gpu` uses the current CUDA device and can fall back to returning its input when
CUDA is unavailable. Its `CUDA.cu` conversion may also change precision. Use an
explicit `CuArray` conversion when you need to preserve a chosen numeric type.
`f32`/`f64` change supported leaf precision; they do not select a device.

## `gpu`

<!--@include: ../../.generated/api/gpu.md-->

## `cpu`

<!--@include: ../../.generated/api/cpu.md-->

## `f32`

<!--@include: ../../.generated/api/f32.md-->

## `f64`

<!--@include: ../../.generated/api/f64.md-->

## Integration and differentiation

No gyromagnetic ratio or physical-unit conversion is included. For magnetic
field gradients in T/m, account for the appropriate conversion before treating
the integral as cycles/m. See [Symbols and notation](/theory/symbols).

## `grad2traj`

<!--@include: ../../.generated/api/grad2traj.md-->

## `traj2grad`

<!--@include: ../../.generated/api/traj2grad.md-->

## Resizing and central regions

`imresize_complex` interpolates magnitude and wrapped phase separately. Phase
near the -pi/pi branch cut can therefore interpolate incorrectly; this is not a
phase-unwrapping or geometry-aware resampling method.

`get_center_crop` currently returns all zeros if either requested axis equals
the corresponding input axis. Its example above deliberately uses a strict crop
on both axes. The limitation is retained in the implementation and documented
below. `normalization` and `standardization` likewise produce nonfinite results
for constant arrays because their denominators vanish.

## `imresize_real`

<!--@include: ../../.generated/api/imresize-real.md-->

## `imresize_complex`

<!--@include: ../../.generated/api/imresize-complex.md-->

## `get_center_range`

<!--@include: ../../.generated/api/get-center-range.md-->

## `get_center_crop`

<!--@include: ../../.generated/api/get-center-crop.md-->

## `get_factors`

<!--@include: ../../.generated/api/get-factors.md-->

## `normalization`

<!--@include: ../../.generated/api/normalization.md-->

## `standardization`

<!--@include: ../../.generated/api/standardization.md-->

## `get_mask`

<!--@include: ../../.generated/api/get-mask.md-->

## Array transforms and metadata

## `convert_fft`

<!--@include: ../../.generated/api/convert-fft.md-->

## `convert_ifft`

<!--@include: ../../.generated/api/convert-ifft.md-->

## `struct_to_dict`

<!--@include: ../../.generated/api/struct-to-dict.md-->

## Scope

The repository's `src/utils/ISMRMRD.jl` and `src/utils/ShiftFOV.jl` are not loaded
by the package module and are not supported public entry points. This migration
does not activate them or invent API documentation for unfinished code.
