"""
    τ = FindDelay_multishot(gridding, data, kspha, datatime, StartTime, dt, ...)

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
- `kspha::AbstractArray{T, 3} [nShot, nSamplePerInterleave, nTerm]`: coefficients of spherical harmonics basis functions
- `datatime::AbstractArray{T, 2} [nShot, nSamplePerInterleave]`: sampling time points of the signal data
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
julia> τ = FindDelay_multishot(gridding, data, kspha, datatime, StartTime, dt, ...)
```
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
    use_gpu     :: Bool                 = true                                              , 
    solver      :: String               = "cgnr"                                            ,
    reg         :: String               = "L2"                                              ,
    iter_max    :: Int64                = 10                                                ,
    λ           :: T                    = 0.                                                ,
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
    recParams[:reconSize]      = (gridding.nX, gridding.nY)
    recParams[:regularization] = reg
    recParams[:λ]              = λ
    recParams[:iterations]     = iter_max
    recParams[:solver]         = solver

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
        weight = SampleDensity(kspha_τ_r[2:3,:], (gridding.nX, gridding.nY));
        HOOp    = HighOrderOp(gridding, kspha_τ_r, datatime_recon; recon_terms=recon_terms, 
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, use_gpu=use_gpu, verbose=verbose);
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
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, use_gpu=use_gpu, verbose=verbose);
            HOOp_dt_τ = HighOrderOp(gridding, Matrix(kspha_τ[:,ishot,:]'), datatime_recon; recon_terms=recon_terms,
                    nBlock=nBlock, csm=csm, fieldmap=fieldmap, use_gpu=use_gpu, verbose=verbose,
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
    use_gpu     :: Bool                 = true                                              , 
    solver      :: String               = "cgnr"                                            ,
    reg         :: String               = "L2"                                              ,
    iter_max    :: Int64                = 10                                                ,
    λ           :: T                    = 0.                                                ,
    verbose     :: Bool                 = false                                             , 
    ) ::T where {T<:AbstractFloat} 
    nSample, nCha = size(data);
    nShot, nSamplePerInterleave, nTerm = size(kspha);
    datatime = T.(collect(dt * (0:nSamplePerInterleave-1)));
    datatime = repeat(datatime, nShot, 1);
    return FindDelay_multishot(
        gridding, data, kspha, datatime, StartTime, dt; 
        JumpFact=JumpFact, Δτ_min=Δτ_min, intermode=intermode, 
        fieldmap=fieldmap, csm=csm, recon_terms=recon_terms, nBlock=nBlock, use_gpu=use_gpu, 
        solver=solver, reg=reg, iter_max=iter_max, λ=λ, verbose=verbose);
end