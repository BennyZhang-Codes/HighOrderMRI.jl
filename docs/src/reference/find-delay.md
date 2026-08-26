# `FindDelay`

Estimate a scalar synchronization delay between MRI data and dynamic field/trajectory coefficients using the model-based update implemented in HighOrderMRI.

```julia
FindDelay(
    gridding::Grid{T},
    data::AbstractArray{Complex{T},2},
    kspha::AbstractArray{T,2},
    datatime::AbstractVector{T},
    StartTime::T,
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

A convenience overload omits `datatime` and assumes equally spaced ADC samples separated by `dt`.

## Arguments

- `gridding`: 2D reconstruction grid used during the iterative image update.
- `data`: complex data `(nSample, nCha)`.
- `kspha`: measured or predicted field coefficients `(nFieldSample, nTerm)`.
- `datatime`: ADC sample times.
- `StartTime`: time origin of the coefficient series relative to the acquisition timing convention.
- `dt`: coefficient sampling interval.

`StartTime`, `dt`, `datatime`, and `Δτ_min` must use the same time unit.

## Keywords

- `JumpFact`: acceleration factor applied to the scalar delay update; reduced after a sign change.
- `Δτ_min`: stopping threshold for the delay increment.
- `intermode`: interpolation scheme.
- `fieldmap`, `csm`, `recon_terms`, `nBlock`: explicit encoding inputs.
- `use_gpu`: use CUDA-backed `HighOrderOp` during the image update.
- `solver`, `reg`, `iter_max`, `λ`: reconstruction settings used inside each synchronization iteration.
- `verbose`: show intermediate reconstruction output and progress information.

## Returns

A scalar delay `τ`. The current function returns only the final value; it does not expose an iteration history or convergence-status object.

At each outer iteration, the implementation reconstructs an image with the current encoding operator and updates the delay from the complex residual and derivative forward product. See [Field preprocessing and synchronization](/guide/field-preprocessing#model-based-delay-estimation) for the update equation, sign convention, and convergence limitations.

## Example

```julia
delay = FindDelay(
    grid,
    data,
    field_coefficients,
    adc_times,
    start_time,
    field_dt;
    csm,
    Δτ_min=delay_tolerance,
)
```

[Source: `FindDelay.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Synchronization/FindDelay.jl)
