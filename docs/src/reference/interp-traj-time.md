# `InterpTrajTime`

Interpolate trajectory or field-coefficient samples to requested acquisition times after applying a temporal offset.

## Two-dimensional form

```julia
InterpTrajTime(
    traj::AbstractArray{T,2},
    dt::Real,
    delTime::Real,
    datatime::AbstractVector{T};
    intermode=AkimaMonotonicInterpolation(),
) where T<:Real
```

`traj` is shaped `(nFieldSample, nTerm)`. The input sample-time grid is formed as

$$
t_n = n\,dt - \mathrm{delTime}.
$$

Values outside the sampled interval are extrapolated as zero.

## Convenience form

```julia
InterpTrajTime(traj, dt, delTime; intermode=...)
```

uses the original equally spaced sampling times as `datatime`.

## Multi-shot form

```julia
InterpTrajTime(
    traj::AbstractArray{T,3},
    dt::Real,
    delTime::AbstractVector{T},
    datatime::AbstractArray{T,2};
    intermode=AkimaMonotonicInterpolation(),
) where T<:Real
```

The three-dimensional input is shaped `(nShot, nFieldSample, nTerm)` with one temporal offset per shot.

## Returns

Interpolated coefficients with field terms in the last dimension.

## Example

```julia
kspha_at_adc = InterpTrajTime(
    kspha_samples,
    field_dt,
    start_time + delay,
    adc_times,
)
```

Inspect the requested interval explicitly because extrapolated leading or trailing values are zero.

[Source: `InterpTraj.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Synchronization/InterpTraj.jl)
