export HighOrderOp3D

"""
# A Julia implementation of the expanded signal encoding model.
- This implementation using GPU with CUDA.jl to accelerate the calculation.
- If the GPU memory is not enough, the calculation can be divided into blocks.
"""
mutable struct HighOrderOp3D{T,F1,F2} <: HOOp{T}
    nrow       :: Int
    ncol       :: Int
    symmetric  :: Bool
    hermitian  :: Bool
    prod!      :: Function
    tprod!     :: F1
    ctprod!    :: F2
    nprod      :: Int
    ntprod     :: Int
    nctprod    :: Int
    args5      :: Bool
    use_prod5! :: Bool
    allocated5 :: Bool
    Mv5        :: Vector{T}
    Mtu5       :: Vector{T}
end
LinearOperators.storage_type(op::HighOrderOp3D) = typeof(op.Mv5)


"""
    HighOrderOp3D(grid::Grid{T}, kspha::AbstractArray{T, 2}, times::AbstractVector{T}; kwargs...)

# Description
    generates a `HighOrderOp3D` which explicitely evaluates the MRI Fourier HighOrder encoding operator.

# Arguments:
* `grid::Grid{T}`                   - grid object.
* `kspha::AbstractArray{T, 2}`      - [nSam, nTerm], Coefficients of field dynamics.
* `times::AbstractVector{T}`        - [nSam], time points for trajectory.

# Keywords:
* `fieldmap::Matrix{T}`             - [nX, nY], fieldmap for off-resonance correction.
* `csm::Array{Complex{T}, 3}`       - [nX, nY, nCha], coil sensitivity map.
* `recon_terms::String`             - digits flag (e.g. "111") to indicate terms to be used in the HOOp.
* `k_nominal::AbstractArray{T, 2}`  - [nSam, 3], nominal kspace trajectory.
* `kspha_dt`                        - [nSam, nTerm], time-derivative of the coefficients of field dynamics.
* `nBlock::Int64`                   - split trajectory into `nBlock` blocks to avoid memory overflow.
* `use_gpu::Bool`                   - use GPU for HighOrder encoding/decoding(default: `true`).
* `verbose::Bool`                   - print progress information(default: `false`).
"""
function HighOrderOp3D(
    grid        :: Grid{T}                                                                   ,
    kspha       :: AbstractArray{T, 2}                                                       , 
    times       :: AbstractVector{T}                                                         ;
    fieldmap    :: AbstractArray{T, 3}  = zeros(T,(grid.nX, grid.nY, grid.nZ))               , 
    csm         :: Array{Complex{T}, 4} = ones(Complex{T},(grid.nX, grid.nY, grid.nZ)..., 1) , 
    recon_terms :: String               = nothing                                            ,
    k_nominal   :: AbstractArray{T, 2}  = kspha[2:4, :]                                      ,
    kspha_dt                            = nothing                                            ,
    nBlock      :: Int64                = 50                                                 , 
    use_gpu     :: Bool                 = true                                               , 
    verbose     :: Bool                 = false                                              , 
    ) where {T<:AbstractFloat}

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm, 4)
    nRow = nSam * nCha
    nCol = nVox = prod(grid.matrixSize)
    if verbose
        @info "HighOrderOp3D nRow=$nRow, nCol=$nCol, nSam=$nSam, nCha=$nCha, nBlock=$nBlock, use_gpu=$use_gpu"
    end

    @assert nTerm              in [9, 16]       "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3             "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY, nZ)  "FieldMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(csm)[1:3]     == (nX, nY, nZ)  "Coil-SensitivityMap must have same size as $((nX, nY, nZ)) in grid"
    
    # prepare data 
    kspha    = prep_kspha_3D(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    csm      = reshape(csm, :, nCha)      # [nX * nY, nCha]
    fieldmap = vec(fieldmap)                  # [nVox]

    # divide the calculation into blocks (nBlock) to avoid memory overflow
    nBlock = nBlock > nSam  ? nSam  : nBlock  # nBlock must be <= k
    n      = nSam ÷ nBlock                    # number of sampless per block
    parts  = [n for i=1:nBlock]               # number of samples per block
    parts  = [1+n*(i-1):n*i for i=1:nBlock]
    if nSam%nBlock!= 0
        push!(parts, n*nBlock+1:nSam)
    end
    
    # if use_gpu, move all the variables to GPU
    if use_gpu
        kspha       = kspha       |> gpu
        kspha_dt    = kspha_dt    |> gpu
        grid        = grid        |> gpu
        times       = times       |> gpu
        fieldmap    = fieldmap    |> gpu
        csm         = csm         |> gpu
    end

    # compute basis functions (spherical harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, collect(1:nTerm))

    if isnothing(kspha_dt)
        func_prod = (res,xm)->(res .= prod_HighOrderOp3D(xm, bf, nVox, nSam, nCha, kspha, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod = (res,xm)->(res .= prod_dt_HighOrderOp3D(xm, bf, nVox, nSam, nCha, kspha, kspha_dt, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    end
    func_ctprod = (res,ym)->(res .= ctprod_HighOrderOp3D(ym, bf, nVox, nSam, nCha, kspha, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    
    return HighOrderOp3D{Complex{T},Nothing,Function}(
                        nRow, nCol, 
                        false, false,
                        func_prod, nothing, func_ctprod,
                        0, 0, 0, 
                        false, false, false, 
                        Complex{T}[], Complex{T}[])
end


function prep_kspha_3D(
    kspha         :: AbstractArray{T, 2} , 
    k_nominal     :: AbstractArray{T, 2} , 
    nTerm         :: Int64               ;
    recon_terms   :: String = nothing    ,
    verbose       :: Bool   = false      , 
    ) where T<:AbstractFloat
    if isnothing(recon_terms)
        recon_terms = nTerm == 9 ? "111" : "1111"
    end
    if nTerm == 9
        @assert length(recon_terms) == 3 "recon_terms must be 3 digits for up to 2nd order terms"
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = false
    elseif nTerm == 16
        @assert length(recon_terms) == 4 "recon_terms must be 4 digits for up to 3rd order terms"
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = Bool(parse(Int64, recon_terms[4]))
    else
        @error "nTerm must be 9 or 16"
    end

    if t0 == false
        kspha[1, :] = kspha[1, :] .* 0
    end
    if t1 == false
        kspha[2:4, :] = k_nominal[:, :]
    end
    if t2 == false
        kspha[5:9, :] = kspha[5:9, :] .* 0
    end
    if t3 == false && nTerm == 16
        kspha[10:16, :] = kspha[10:16, :] .* 0
    end
    if verbose
        @info "kspha prepared for flag: $(recon_terms)" zeroth=t0 first=t1 second=t2 third=t3
    end
    return kspha
end


"""
    prod_dt_HighOrderOp3D
    for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
"""
function prod_dt_HighOrderOp3D(
    x         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               ,
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    kspha     :: AbstractArray{D, 2}                 , 
    kspha_dt  :: AbstractArray{D, 2}                 ,
    times     :: AbstractVector{D}                   , 
    fieldmap  :: AbstractVector{D}                   ,
    csm       :: AbstractArray{Complex{D}, 2}        ;
    nBlock    :: Int64                    = 1        , 
    parts     :: Vector{UnitRange{Int64}} = [1:nSam] , 
    use_gpu   :: Bool                     = false    , 
    verbose   :: Bool                     = false    ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    x = Vector(x)
    if verbose
        @info "HighOrderOp3D prod_dt nBlock=$nBlock, use_gpu=$use_gpu"
    end
    if use_gpu
        x   = x |> gpu
        out = CUDA.zeros(Complex{D}, nSam, nCha)
    else
        out = zeros(Complex{D}, nSam, nCha)
    end
    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:,p]))'
        e = exp.(2*1im*pi*ϕ) .* (bf * @view(kspha_dt[:,p]))' .* (2*1im*pi)
        out[p, :] =  e * (x .* csm)
        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
        if use_gpu
            CUDA.unsafe_free!(ϕ)
            CUDA.unsafe_free!(e)
        end
    end
    if use_gpu
        CUDA.unsafe_free!(x)
    end
    out = out ./ sqrt(nVox)
    if use_gpu
        out = out |> cpu
    end
    return vec(out)
end


"""
    Forward operator for HighOrderOp3D
"""
function prod_HighOrderOp3D(
    x         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               ,
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    kspha     :: AbstractArray{D, 2}                 , 
    times     :: AbstractVector{D}                   , 
    fieldmap  :: AbstractVector{D}                   ,
    csm       :: AbstractArray{Complex{D}, 2}        ;
    nBlock    :: Int64                    = 1        , 
    parts     :: Vector{UnitRange{Int64}} = [1:nSam] , 
    use_gpu   :: Bool                     = false    , 
    verbose   :: Bool                     = false    ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    x = Vector(x)
    if verbose
        @info "HighOrderOp3D prod nBlock=$nBlock, use_gpu=$use_gpu"
    end
    if use_gpu
        x   = x |> gpu
        out = CUDA.zeros(Complex{D}, nSam, nCha)
    else
        out = zeros(Complex{D}, nSam, nCha)
    end
    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:,p]))'
        e = exp.(2*1im*pi*ϕ)
        out[p, :] =  e * (x .* csm)
        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
        if use_gpu
            CUDA.unsafe_free!(ϕ)
            CUDA.unsafe_free!(e)
        end
    end
    if use_gpu
        CUDA.unsafe_free!(x)
    end
    out = out ./ sqrt(nVox)
    if use_gpu
        out = out |> cpu
    end
    return vec(out)
end


"""
    Adjoint of prod_HighOrderOp3D
"""
function ctprod_HighOrderOp3D(
    y         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               , 
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    kspha     :: AbstractArray{D, 2}                 , 
    times     :: AbstractVector{D}                   , 
    fieldmap  :: AbstractVector{D}                   ,
    csm       :: AbstractArray{Complex{D}, 2}        ;
    nBlock    :: Int64                    = 1        , 
    parts     :: Vector{UnitRange{Int64}} = [1:nSam] , 
    use_gpu   :: Bool                     = false    , 
    verbose   :: Bool                     = false    ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    csmC = conj.(csm)
    y    = reshape(y, nSam, nCha)

    if verbose
        @info "HighOrderOp3D ctprod nBlock=$nBlock, use_gpu=$use_gpu"
    end
    
    if use_gpu
        y   = y |> gpu
        out = CUDA.zeros(Complex{D}, nVox, nCha)
    else
        out = zeros(Complex{D}, nVox, nCha)
    end

    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        ϕ =  fieldmap .* @view(times[p])' .+ (bf * @view(kspha[:,p]))
        e = exp.(2*1im*pi*ϕ)
        out +=  conj(e) * y[p, :]
        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
        if use_gpu
            CUDA.unsafe_free!(ϕ)
            CUDA.unsafe_free!(e)
        end
    end
    if use_gpu
        CUDA.unsafe_free!(y)
    end
    out = out ./ sqrt(nVox)
    out = out .* csmC
    if use_gpu
        out = out |> cpu
    end
  return vec(sum(out, dims=2))
end


function Base.adjoint(op::HighOrderOp3D{T}) where T
  return LinearOperator{T}(
                            op.ncol, 
                            op.nrow, 
                            op.symmetric, 
                            op.hermitian,
                            op.ctprod!, 
                            nothing, 
                            op.prod!)
end