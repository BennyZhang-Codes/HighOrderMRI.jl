"""
    τ = FindDelay(gridding, data, kspha, datatime, StartTime, dt, ...)

This is a modified version, minor differences are: 
1. apply density weighting in the estimation of `x`; 
2. final results are returned with (τ - Δτ / JumpFact * (JumpFact - 1)).

# Description
A Julia implementation of the model-based synchronization delay estimation algorithm.
The algorithm is based on the paper "Model-based determination of the synchronization delay between MRI and trajectory data"
by Paul Dubovan and Corey Baron (2022, https://doi.org/10.1002/mrm.29460).

This implementation utilizes our `HighOrderOp` to realize the expanded encoding model with dynamic fields.

# Arguments
- `gridding::Grid{T}`: gridding object containing the grid parameters
- `data::AbstractArray{T, 2} [nSample, nCha]`: signal data
- `kspha::AbstractArray{T, 2} [nSample, nTerm]`: coefficients of spherical harmonics basis functions
- `datatime::AbstractVector [nSample]`: sampling time points of the signal data
- `StartTime::T`: the time point of the first sample
- `dt::T`: sampling interval

# Keywords
- `JumpFact::Int64 = 3`: scaling factor for acceleration of convergence
- `Δτ_min::T = 0.005`: [us] minimum delay increment
- `intermode::Interpolations.InterpolationType = AkimaMonotonicInterpolation()`: interpolation method
- `fieldmap::AbstractArray{T, 2} [nX, nY]`: field map
- `csm::Array{Complex{T}, 3} [nX, nY, nCha]`: coil sensitivity map
- `recon_terms:: = "111"`: digits flag (e.g. "111") to indicate terms to be used in the HOOp.
- `nBlock::Int64 = 50`: number of blocks for parallel processing
- `use_gpu::Bool = true`: whether to use GPU acceleration
- `solver::String = "cgnr"`: solver for the inverse problem
- `reg::String = "L2"`: regularization method
- `iter_max::Int64 = 10`: maximum number of iterations for the solver
- `λ::T = 0.`: regularization parameter
- `verbose::Bool = false`: whether to print verbose output

# Returns
- `τ::T`: synchronization delay

# Example
```julia
julia> τ = FindDelay(gridding, data, kspha, datatime, StartTime, dt, ...)
```
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
    use_gpu     :: Bool                 = true                                              , 
    solver      :: String               = "cgnr"                                            ,
    reg         :: String               = "L2"                                              ,
    iter_max    :: Int64                = 10                                                ,
    λ           :: T                    = 0.                                                ,
    verbose     :: Bool                 = false                                             , 
    ) ::T where {T<:AbstractFloat}
    @assert size(data,2) == size(csm,3) "data and csm must have the same number of coil channels"
    @assert size(data,1) == size(datatime, 1) "data and datatime must have the same number of spatial points"

    nSample, nCha = size(data);

    # 1. Initialize some variables
    τ         = 0;
    nIter     = 1;
    Δτ_prev   = Δτ = Inf;
    Δτ_min    = Δτ_min;        # [us]
    τ_perIter = Vector{Float64}();
    push!(τ_perIter, τ);
    recParams = Dict{Symbol,Any}()
    recParams[:reconSize]      = (gridding.nX, gridding.nY)
    recParams[:regularization] = reg
    recParams[:λ]              = λ
    recParams[:iterations]     = iter_max
    recParams[:solver]         = solver


    # 2. Compute kspha_dt, which is "dk/dt"
    kernel = T.([1/8 1/4 0 -1/4 -1/8]');
    kspha_dt = conv(kspha, kernel)[3:end-2,:]/dt;  # crop both sides to match size of kspha
    
    while abs(Δτ) > Δτ_min
        # Interpolate to match datatime
        kspha_τ    = T.(InterpTrajTime(kspha   , dt, τ + StartTime, datatime, intermode=intermode)[1:nSample,:]');
        kspha_dt_τ = T.(InterpTrajTime(kspha_dt, dt, τ + StartTime, datatime, intermode=intermode)[1:nSample,:]');
        
        weight = SampleDensity(kspha_τ[2:3,:], (gridding.nX, gridding.nY));
        HOOp    = HighOrderOp(gridding, kspha_τ, datatime; recon_terms=recon_terms, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=(use_gpu ? CuArray : Array), verbose=verbose);
        HOOp_dt = HighOrderOp(gridding, kspha_τ, datatime; recon_terms=recon_terms, kspha_dt=kspha_dt_τ, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, arrayType=(use_gpu ? CuArray : Array), verbose=verbose);
        
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
    use_gpu     :: Bool                 = true                                              , 
    solver      :: String               = "cgnr"                                            ,
    reg         :: String               = "L2"                                              ,
    iter_max    :: Int64                = 10                                                ,
    λ           :: T                    = 0.                                                ,
    verbose     :: Bool                 = false                                             , 
    ) ::T where {T<:AbstractFloat} 
    nSample, nCha = size(data);
    datatime = T.(collect(dt * (0:nSample-1)));
    return FindDelay(
        gridding, data, kspha, datatime, StartTime, dt; 
        JumpFact=JumpFact, Δτ_min=Δτ_min, intermode=intermode, 
        fieldmap=fieldmap, csm=csm, recon_terms=recon_terms, nBlock=nBlock, use_gpu=use_gpu, 
        solver=solver, reg=reg, iter_max=iter_max, λ=λ, verbose=verbose);
end
