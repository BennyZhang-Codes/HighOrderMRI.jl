# Array and signal utilities

Common exported helpers used for array conversion, trajectory-gradient conversion, resizing, cropping, and factor selection.

## CPU/GPU and precision conversion

```julia
gpu(x)
cpu(x)
f32(x)
f64(x)
```

Use these helpers when moving arrays between CPU/CUDA storage or normalizing floating-point precision in scripts.

## Gradient and trajectory conversion

```julia
grad2traj(...)
traj2grad(...)
```

These functions convert between gradient waveforms and accumulated trajectory/phase coefficients. The gyromagnetic ratio, time step, and physical units must be consistent with the acquisition convention used by the encoding operator.

## Image resizing

```julia
imresize_real(...)
imresize_complex(...)
```

Convenience wrappers for resizing real and complex arrays.

## Cropping and factors

```julia
get_center_range(...)
get_center_crop(...)
get_factors(...)
```

Utilities for centred array ranges/crops and integer-factor selection.

For scientific reconstruction code, unit-sensitive trajectory conversion should be documented together with the phase convention described in [Symbols and notation](/theory/symbols).

[Source: `src/utils`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/tree/docs-modern-ui/src/utils)
