"""
    FindDelay(
        gridding :: Grid{T} ,
        data :: AbstractArray{Complex{T}, 2} ,
        kspha :: AbstractArray{T, 2} ,
        datatime :: AbstractVector{T} ,
        StartTime :: T ,
        dt :: T ;
        JumpFact :: Int64 = 3 ,
        Δτ_min :: T = 0.005 ,
        intermode :: Interpolations.InterpolationType = AkimaMonotonicInterpolation() ,
        fieldmap :: AbstractArray{T, 2} = zeros(T,(gridding.nX, gridding.nY)) ,
        csm :: Array{Complex{T}, 3} = ones(Complex{T},(gridding.nX, gridding.nY)..., 1) ,
        recon_terms :: String ="111" ,
        nBlock :: Int64 = 50 ,
        arrayType :: Type{<:AbstractArray}= Array ,
        solver :: Type{<:AbstractLinearSolver} = CGNR ,
        reg :: AbstractParameterizedRegularization = L2Regularization(zero(T)) ,
        iter_max :: Int64 = 10 ,
        verbose :: Bool = false ,
    ) ::T where {T<:AbstractFloat}

    FindDelay(
        gridding :: Grid{T} ,
        data :: AbstractArray{Complex{T}, 2} ,
        kspha :: AbstractArray{T, 2} ,
        StartTime :: T ,
        dt :: T ;
        JumpFact :: Int64 = 3 ,
        Δτ_min :: T = 0.005 ,
        intermode :: Interpolations.InterpolationType = AkimaMonotonicInterpolation() ,
        fieldmap :: AbstractArray{T, 2} = zeros(T,(gridding.nX, gridding.nY)) ,
        csm :: Array{Complex{T}, 3} = ones(Complex{T},(gridding.nX, gridding.nY)..., 1) ,
        recon_terms :: String = "111" ,
        nBlock :: Int64 = 50 ,
        arrayType :: Type{<:AbstractArray}= Array ,
        solver :: Type{<:AbstractLinearSolver} = CGNR ,
        reg :: AbstractParameterizedRegularization = L2Regularization(zero(T)) ,
        iter_max :: Int64 = 10 ,
        verbose :: Bool = false ,
    ) ::T where {T<:AbstractFloat}

Estimate a scalar synchronization delay between complex MRI data and sampled field coefficients for a two-dimensional reconstruction.

The implementation alternates a density-weighted image reconstruction with an
unweighted scalar delay update from the data residual and the derivative forward
product. It follows the model-based approach of Dubovan and Baron, with the
weighting and accelerated update implemented in this package.

# Arguments

All real-valued inputs share `T<:AbstractFloat`; `data` and `csm` use `Complex{T}`.

- `gridding::Grid{T}`: reconstruction grid with `nZ == 1`.
- `data::AbstractArray{Complex{T},2}`: MRI samples, shaped `(nSample, nCha)`.
- `kspha::AbstractArray{T,2}`: time-integrated field coefficients, shaped
  `(nFieldSample, nTerm)`, with `nTerm` equal to 9 or 16. Use the solid-harmonic
  order and cycles-based phase units expected by `HighOrderOp`.
- `datatime::AbstractVector{T}`: `nSample` ADC times, in seconds. This vector
  may have a different sampling interval from the field coefficients.
- `StartTime::T`: offset subtracted from the field-coefficient time grid,
  in seconds. At the current delay `τ`, coefficient sample `j` is placed at
  `(j - 1) * dt - (StartTime + τ)` before interpolation.
- `dt::T`: positive field-coefficient sampling interval, in seconds.

The overload without `datatime` constructs `T.(dt * (0:nSample-1))` for the
ADC times. Use it only when the ADC interval equals the coefficient interval
and the first ADC time is zero in the chosen time reference.

# Keywords

- `JumpFact::Int64=3`: multiplier of the scalar delay update. It is reduced
  after a change in the update's sign.
- `Δτ_min::T=0.005`: stopping threshold for the absolute delay increment, in
  the same time unit as `dt`. The literal default is `0.005`, not a conversion
  from microseconds; with second-based inputs it represents 5 ms. Pass a
  finite positive tolerance appropriate to the acquisition explicitly.
- `intermode::Interpolations.InterpolationType=AkimaMonotonicInterpolation()`:
  interpolation method for the field coefficients and their time derivatives.
- `fieldmap::AbstractArray{T,2}=zeros(T,(gridding.nX, gridding.nY))`:
  static off-resonance map in Hz, shaped `(nX, nY)`.
- `csm::Array{Complex{T},3}=ones(Complex{T},(gridding.nX, gridding.nY)..., 1)`:
  coil-sensitivity maps, shaped `(nX, nY, nCha)`.
- `recon_terms::String="111"`: order-selection string passed to `HighOrderOp`.
  Use three digits for 9 coefficient columns and four digits for 16 columns.
- `nBlock::Int64=50`: trajectory block count passed to the explicit operators
  to bound temporary storage.
- `arrayType::Type{<:AbstractArray}=Array`: operator backend passed to
  `HighOrderOp`. The default uses the CPU. This wrapper creates density weights
  on the CPU; changing this keyword alone is not an end-to-end GPU setup.
- `solver::Type{<:AbstractLinearSolver}=CGNR`: inner reconstruction solver
  type from `RegularizedLeastSquares`, rather than a solver-name string.
- `reg::AbstractParameterizedRegularization=L2Regularization(zero(T))`:
  inner reconstruction regularizer object. Configure its strength in the
  object; this function has no separate `λ` keyword.
- `iter_max::Int64=10`: iteration limit for each inner image reconstruction.
  This does not limit the number of outer delay updates.
- `verbose::Bool=false`: show intermediate image plots and enable verbosity
  in the explicit operators. Delay-update information is logged even when
  this keyword is `false`.

# Returns

A scalar `τ::T`, in seconds with the input convention above. Apply it as
`StartTime + τ` when calling `InterpTrajTime`. Positive `τ` evaluates the
coefficient series at later source times: `k(t + StartTime + τ)`.

After the final update, the implementation returns
`τ_iter - Δτ / J * (J - 1)`, where `τ_iter` is its accumulated delay and
`J` is the final value of `JumpFact`. No iteration history or convergence
status is returned.

# Limitations

The current `Δτ_min=0.005` keyword default is a `Float64` literal and causes
a method-dispatch error for `Float32` inputs. Pass a same-type value explicitly,
for example `Δτ_min=Float32(5e-9)` for a selected tolerance of 0.005 μs when
all times are in seconds. This numerical tolerance is an example, not a
validated choice for every acquisition.

The outer loop has no iteration limit or non-finite-update guard. A returned
scalar does not establish convergence or timing accuracy. The estimator starts
with zero additional delay; `StartTime` supplies the initial temporal offset.
Interpolation uses zero outside the available coefficient interval, so the
shifted interval must cover the ADC times relevant to the reconstruction.

The iteration log currently prints `Δτ/dt` and `τ/dt` with the label `[us]`.
Those ratios are dimensionless sample offsets, not microseconds. Interpret
the returned value using the input time unit rather than that log label.

See also `InterpTrajTime`, `FindDelay_multishot`, `HighOrderOp`, and `recon_HOOp`.

# Reference

Dubovan PI, Baron CA. Model-based determination of the synchronization delay
between MRI and trajectory data. *Magnetic Resonance in Medicine*.
2023;89:721–728. [doi:10.1002/mrm.29460](https://doi.org/10.1002/mrm.29460).
"""
function FindDelay(
    gridding    :: Grid{T}                                                                  ,
    data        :: AbstractArray{Complex{T}, 2}                                             ,
    kspha       :: AbstractArray{T, 2}                                                      , 
    datatime    :: AbstractVector{T}                                                        ,
    StartTime   :: T                                                                        ,
    dt          :: T                                                                        ;
    JumpFact    :: Int64                = 3                                                 ,
    Δτ_min      :: T                    = 0.005                                             ,
    intermode   :: Interpolations.InterpolationType = AkimaMonotonicInterpolation()         ,
    fieldmap    :: AbstractArray{T, 2}  = zeros(T,(gridding.nX, gridding.nY))               , 
    csm         :: Array{Complex{T}, 3} = ones(Complex{T},(gridding.nX, gridding.nY)..., 1) , 
    recon_terms :: String               ="111"                                              ,
    nBlock      :: Int64                = 50                                                , 
    arrayType   :: Type{<:AbstractArray}= Array                                             ,
    solver      :: Type{<:AbstractLinearSolver} = CGNR                                      ,
    reg         :: AbstractParameterizedRegularization = L2Regularization(zero(T))          ,
    iter_max    :: Int64                = 10                                                ,
    verbose     :: Bool                 = false                                             , 
    ) ::T where {T<:AbstractFloat}
    @assert size(data,2) == size(csm,3) "data and csm must have the same number of coil channels"
    @assert size(data,1) == size(datatime, 1) "data and datatime must have the same number of spatial points"

    nSample, nCha = size(data);

    # 1. Initialize some variables
    τ         = 0;
    nIter     = 1;
    Δτ_prev   = Δτ = Inf;
    Δτ_min    = Δτ_min;        # same time unit as dt (seconds with fieldmap in Hz)
    τ_perIter = Vector{Float64}();
    push!(τ_perIter, τ);
    recParams = Dict{Symbol,Any}()
    recParams[:reconSize]   = (gridding.nX, gridding.nY)
    recParams[:reg]         = reg
    recParams[:iterations]  = iter_max
    recParams[:solver]      = solver


    # 2. Compute kspha_dt, which is "dk/dt"
    kernel = T.([1/8 1/4 0 -1/4 -1/8]');
    kspha_dt = conv(kspha, kernel)[3:end-2,:]/dt;  # crop both sides to match size of kspha
    
    while abs(Δτ) > Δτ_min
        # Interpolate to match datatime
        kspha_τ    = T.(InterpTrajTime(kspha   , dt, τ + StartTime, datatime, intermode=intermode)[1:nSample,:]');
        kspha_dt_τ = T.(InterpTrajTime(kspha_dt, dt, τ + StartTime, datatime, intermode=intermode)[1:nSample,:]');
        
        weight = samplingDensity(kspha_τ[2:3,:], (gridding.nX, gridding.nY));
        HOOp    = HighOrderOp(gridding, kspha_τ, datatime; recon_terms=recon_terms, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=arrayType, verbose=verbose);
        HOOp_dt = HighOrderOp(gridding, kspha_τ, datatime; recon_terms=recon_terms, kspha_dt=kspha_dt_τ, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=arrayType, verbose=verbose);
        
        # Update image
        x = recon_HOOp(HOOp, data, weight, recParams)
        if verbose
            plt_image(abs.(x), title="Iteration $nIter", vmaxp=99.9)
        end
        # Update delay
        y1 = vec(data) - HOOp * vec(x); # Y - Aₚxₚ
        y2 = HOOp_dt * vec(x);       # Bₚxₚ
        Δτ = JumpFact * real(y2 \ y1);
        τ += Δτ;

        @info "Iteration $nIter: Δτ = $(round(Δτ/dt, digits=5)) [us] | τ = $(round(τ/dt, digits=5)) [us] | JumpFact = $JumpFact"
        
        if (nIter>1) && (sign(Δτ_prev) != sign(Δτ))
            JumpFact = maximum([1, JumpFact/2]);
        end
        Δτ_prev = Δτ;
        nIter  += 1;
        push!(τ_perIter, τ);
    end
    return T.(τ - Δτ / JumpFact * (JumpFact - 1))
end

function FindDelay(
    gridding    :: Grid{T}                                                                  ,
    data        :: AbstractArray{Complex{T}, 2}                                             ,
    kspha       :: AbstractArray{T, 2}                                                      , 
    StartTime   :: T                                                                        ,
    dt          :: T                                                                        ;
    JumpFact    :: Int64                = 3                                                 ,
    Δτ_min      :: T                    = 0.005                                             ,
    intermode   :: Interpolations.InterpolationType = AkimaMonotonicInterpolation()         ,
    fieldmap    :: AbstractArray{T, 2}  = zeros(T,(gridding.nX, gridding.nY))               , 
    csm         :: Array{Complex{T}, 3} = ones(Complex{T},(gridding.nX, gridding.nY)..., 1) , 
    recon_terms :: String               = "111"                                             ,
    nBlock      :: Int64                = 50                                                , 
    arrayType   :: Type{<:AbstractArray}= Array                                             ,
    solver      :: Type{<:AbstractLinearSolver} = CGNR                                      ,
    reg         :: AbstractParameterizedRegularization = L2Regularization(zero(T))          ,
    iter_max    :: Int64                = 10                                                ,
    verbose     :: Bool                 = false                                             , 
    ) ::T where {T<:AbstractFloat} 
    nSample, nCha = size(data);
    datatime = T.(collect(dt * (0:nSample-1)));
    return FindDelay(
        gridding, data, kspha, datatime, StartTime, dt; 
        JumpFact=JumpFact, Δτ_min=Δτ_min, intermode=intermode, 
        fieldmap=fieldmap, csm=csm, recon_terms=recon_terms, nBlock=nBlock, arrayType=arrayType, 
        solver=solver, reg=reg, iter_max=iter_max, verbose=verbose);
end
