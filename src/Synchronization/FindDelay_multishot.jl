"""
    FindDelay_multishot(gridding::Grid{T}, data::AbstractArray{Complex{T}, 2}, kspha::AbstractArray{T, 3}, datatime::AbstractArray{T, 2}, StartTime::AbstractVector{T}, dt::T; JumpFact::Int64 = 3, Δτ_min::T = 0.005, intermode::Interpolations.InterpolationType = AkimaMonotonicInterpolation(), fieldmap::AbstractArray{T, 2} = zeros(T, (gridding.nX, gridding.nY)), csm::Array{Complex{T}, 3} = ones(Complex{T}, (gridding.nX, gridding.nY)..., 1), recon_terms::String = "111", nBlock::Int64 = 50, arrayType::Type{<:AbstractArray} = Array, solver::Type{<:AbstractLinearSolver} = CGNR, reg::AbstractParameterizedRegularization = L2Regularization(zero(T)), iter_max::Int64 = 10, verbose::Bool = false)::T where T <: AbstractFloat

    FindDelay_multishot(gridding::Grid{T}, data::AbstractArray{Complex{T}, 2}, kspha::AbstractArray{T, 3}, StartTime::AbstractVector{T}, dt::T; JumpFact::Int64 = 3, Δτ_min::T = 0.005, intermode::Interpolations.InterpolationType = AkimaMonotonicInterpolation(), fieldmap::AbstractArray{T, 2} = zeros(T, (gridding.nX, gridding.nY)), csm::Array{Complex{T}, 3} = ones(Complex{T}, (gridding.nX, gridding.nY)..., 1), recon_terms::String = "111", nBlock::Int64 = 50, arrayType::Type{<:AbstractArray} = Array, solver::Type{<:AbstractLinearSolver} = CGNR, reg::AbstractParameterizedRegularization = L2Regularization(zero(T)), iter_max::Int64 = 10, verbose::Bool = false)::T where T <: AbstractFloat

Estimate one scalar synchronization delay shared by all shots of a 2D image.
The image update uses all shots; per-shot residual and derivative-forward
products are stacked into one real scalar least-squares update. This is not
an estimator of independent shot delays. The method follows the model-based
synchronization framework of Dubovan and Baron (doi:10.1002/mrm.29460), with
this package's density-weighted image update and JumpFact acceleration.

# Inputs and units

Use a `Grid{T}` with one z plane, complex data `(nShot*nADCSample,nCha)`,
coefficients `(nShot,nFieldSample,nTerm)`, ADC times `(nShot,nADCSample)`, and
`StartTime::AbstractVector{T}` of length nShot. Samples within each shot vary
fastest in data. `fieldmap` is `(nX,nY)` and `csm` is `(nX,nY,nCha)`. Match
the coefficient order and spatial units to the encoding basis. Nine terms
with `recon_terms="111"` are the ordinary second-order configuration.

Use seconds for `dt`, `datatime`, `StartTime`, `Δτ_min` and the returned delay
when `fieldmap` is in Hz. Shot i is sampled as
`k_i(t_ADC+StartTime[i]+τ)` under the interpolation convention. No integration
or gyromagnetic-ratio conversion is performed by this estimator.

# Options and iteration

`arrayType=Array` selects CPU execution; `CuArray` selects the GPU path of the
legacy explicit operator. `solver` is a solver type such as `CGNR`, and `reg`
is a regularization object such as `L2Regularization(zero(T))`. There are no
`use_gpu` or `λ` keywords. `iter_max` limits each inner image reconstruction,
not the outer delay loop. `nBlock` is passed to `HighOrderOp`.

At each step, `Δτ = J*real(b\\r)`, with residual r and derivative product b.
The update is unweighted although image reconstruction uses density weights.
J starts at `JumpFact`, is reduced after an update-sign change, and the loop
tests `abs(Δτ)>Δτ_min`. The result is converted to `T` from
`τ-Δτ/J*(J-1)` using the final J. It is a scalar without convergence status.
The current log prints `Δτ/dt` and `τ/dt` labelled as microseconds; those ratios
are sample offsets, not microseconds.

# Current restrictions

- The convenience overload without `datatime` constructs the wrong time-array
  shape. Use the explicit-time overload; this documentation does not repair it.
- The per-shot residual operators currently receive the complete concatenated
  time vector and therefore reuse the first shot's times. Different shot ADC
  time vectors with a nonzero field map can produce incorrect residuals. Do
  not use this estimator for that case; identical per-shot times or a zero
  field map avoid this specific defect, without guaranteeing convergence.
- `Δτ_min::T=0.005` is the actual Float64 default literal and fails for
  Float32 inputs when omitted. Supply a same-type threshold explicitly.
  In seconds, 0.005 means 5 ms; `Float32(5e-9)` represents 0.005 microseconds.
  A later JumpFact reduction can also promote Float32 delay state; Float64
  is the conservative current choice for multishot estimator calls.
- The outer loop has no iteration cap or reliable breakdown-status return.
  Zero/poorly identifiable derivative information, extrapolated zeros and
  noisy data require independent assessment. Initial τ is fixed at zero;
  no caller-provided initial-delay keyword is available.

See [Field preprocessing](/guide/field-preprocessing) for the update equations
and [FindDelay](/reference/find-delay) for single-shot configuration.
"""
function FindDelay_multishot(
    gridding    :: Grid{T}                                                                  ,
    data        :: AbstractArray{Complex{T}, 2}                                             ,
    kspha       :: AbstractArray{T, 3}                                                      , 
    datatime    :: AbstractArray{T, 2}                                                      ,
    StartTime   :: AbstractVector{T}                                                        ,
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
    @assert size(kspha, 1) == size(datatime, 1) "kspha and datatime must have the same number of shots"

    nSample, nCha = size(data);
    nShot, _, nTerm = size(kspha);
    nShot, nSamplePerInterleave = size(datatime);


    # 1. Initialize some variables
    τ         = 0;
    nIter     = 1;
    Δτ_prev   = Δτ = Inf;
    Δτ_min    = Δτ_min;        # [us]
    τ_perIter = Vector{T}();
    push!(τ_perIter, τ);
    recParams = Dict{Symbol,Any}()
    recParams[:reconSize]   = (gridding.nX, gridding.nY)
    recParams[:reg]         = reg
    recParams[:iterations]  = iter_max
    recParams[:solver]      = solver

    # 2. Compute kspha_dt, which is "dk/dt"
    kernel = T.([1/8 1/4 0 -1/4 -1/8]');
    kspha_dt = zeros(T, size(kspha));
    for ishot = 1:nShot
        kspha_dt[ishot,:,:] = conv(kspha[ishot,:,:], kernel)[3:end-2,:]/dt;  # crop both sides to match size of kspha
    end

    while maximum(abs.(Δτ)) > Δτ_min
        kspha_τ    = T.(InterpTrajTime(kspha   , dt, τ .+ StartTime, datatime, intermode=intermode));
        kspha_dt_τ = T.(InterpTrajTime(kspha_dt, dt, τ .+ StartTime, datatime, intermode=intermode));
        kspha_τ    = permutedims(kspha_τ   , [2, 1, 3]); # [nSamplePerInterleave, nShot, nTerm]
        kspha_dt_τ = permutedims(kspha_dt_τ, [2, 1, 3]); # [nSamplePerInterleave, nShot, nTerm]
        
        kspha_τ_r  = T.(reshape(kspha_τ   , :, nTerm)'); # reshape to (nSample, nTerm)
        datatime_recon = reshape(permutedims(datatime, [2, 1]), :);
        weight = samplingDensity(kspha_τ_r[2:3,:], (gridding.nX, gridding.nY));
        HOOp    = HighOrderOp(gridding, kspha_τ_r, datatime_recon; recon_terms=recon_terms, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=arrayType, verbose=verbose);
        # Update image
        x = recon_HOOp(HOOp, data, weight, recParams)
        if verbose
            plt_image(abs.(x), title="Iteration $nIter", vmaxp=99.9)
        end

        kdata = reshape(data, nSamplePerInterleave, nShot, nCha);
        y1 = zeros(Complex{T}, nSamplePerInterleave, nShot, nCha);
        y2 = zeros(Complex{T}, nSamplePerInterleave, nShot, nCha);
        for ishot = 1:nShot
            HOOp_τ    = HighOrderOp(gridding, Matrix(kspha_τ[:,ishot,:]'), datatime_recon; recon_terms=recon_terms, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=arrayType, verbose=verbose);
            HOOp_dt_τ = HighOrderOp(gridding, Matrix(kspha_τ[:,ishot,:]'), datatime_recon; recon_terms=recon_terms,
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=arrayType, verbose=verbose,
                    kspha_dt=Matrix(kspha_dt_τ[:,ishot,:]'));
            # Update delay
            y1_τ = vec(kdata[:,ishot,:]) - HOOp_τ * vec(x); # Y - Aₚxₚ
            y2_τ = HOOp_dt_τ * vec(x);       # Bₚxₚ

            y1[:,ishot,:] = reshape(y1_τ, nSamplePerInterleave, nCha);
            y2[:,ishot,:] = reshape(y2_τ, nSamplePerInterleave, nCha);
        end
        Δτ = JumpFact * real(vec(y2) \ vec(y1));
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

function FindDelay_multishot(
    gridding    :: Grid{T}                                                                  ,
    data        :: AbstractArray{Complex{T}, 2}                                             ,
    kspha       :: AbstractArray{T, 3}                                                      , 
    StartTime   :: AbstractVector{T}                                                        ,
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
    nShot, nSamplePerInterleave, nTerm = size(kspha);
    datatime = T.(collect(dt * (0:nSamplePerInterleave-1)));
    datatime = repeat(datatime, nShot, 1);
    return FindDelay_multishot(
        gridding, data, kspha, datatime, StartTime, dt; 
        JumpFact=JumpFact, Δτ_min=Δτ_min, intermode=intermode, 
        fieldmap=fieldmap, csm=csm, recon_terms=recon_terms, nBlock=nBlock, arrayType=arrayType, 
        solver=solver, reg=reg, iter_max=iter_max, verbose=verbose);
end
