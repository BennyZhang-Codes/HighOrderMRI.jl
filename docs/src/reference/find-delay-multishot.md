# `FindDelay_multishot`

Estimate one synchronization delay shared across multiple shots while retaining the shot-specific start-time offsets.

```julia
FindDelay_multishot(
    gridding::Grid{T},
    data::AbstractArray{Complex{T},2},
    kspha::AbstractArray{T,3},
    datatime::AbstractArray{T,2},
    StartTime::AbstractVector{T},
    dt::T;
    JumpFact=3,
    Δτ_min=T(0.005),
    intermode=AkimaMonotonicInterpolation(),
    fieldmap=zeros(T, gridding.nX, gridding.nY),
    csm=ones(Complex{T}, gridding.nX, gridding.nY, 1),
    recon_terms="111",
    nBlock=50,
    use_gpu=true,
    solver="cgnr",
    reg="L2",
    iter_max=10,
    λ=zero(T),
    verbose=false,
) where T<:AbstractFloat
```

A convenience overload omits `datatime` and constructs equally spaced sample times within each shot.

## Arguments

- `gridding`: 2D reconstruction grid.
- `data`: concatenated complex data `(nSampleTotal, nCha)`.
- `kspha`: field coefficients `(nShot, nFieldSample, nTerm)`.
- `datatime`: ADC times `(nShot, nSamplePerShot)`.
- `StartTime`: one starting time per shot.
- `dt`: field-coefficient sampling interval.

## Keywords

The keyword meanings match [`FindDelay`](/reference/find-delay).

## Returns

One scalar delay shared by all shots.

The image update uses all shots jointly. The derivative residuals are then evaluated shot by shot and combined into one scalar least-squares delay update. The method therefore does **not** estimate independent shot-specific synchronization delays.

## Example

```julia
delay = FindDelay_multishot(
    grid,
    data,
    field_coefficients_by_shot,
    adc_times_by_shot,
    start_times,
    field_dt;
    csm,
)
```

See [Field preprocessing and synchronization](/guide/field-preprocessing#model-based-delay-estimation) for the shared-delay interpretation.

[Source: `FindDelay_multishot.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Synchronization/FindDelay_multishot.jl)
