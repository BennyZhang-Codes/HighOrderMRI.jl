export HighOrderOp_Kernel

"""
# A Julia implementation of the expanded signal encoding model.
- This implementation using GPU with CUDA.jl to accelerate the calculation.
- If the GPU memory is not enough, the calculation can be divided into blocks.
"""
mutable struct HighOrderOp_Kernel{T,F1,F2} <: HOOp{T}
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
LinearOperators.storage_type(op::HighOrderOp_Kernel) = typeof(op.Mv5)


"""
    HighOrderOp_Kernel(grid::Grid{T}, kspha::AbstractArray{T, 2}, times::AbstractVector{T}; kwargs...)

# Description
    generates a `HighOrderOp_Kernel` which explicitely evaluates the MRI Fourier HighOrder encoding operator.

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
* `gpus::Vector{Int}`               - GPU device ids to be used (default: `[0]`).
* `verbose::Bool`                   - print progress information(default: `false`).
"""
function HighOrderOp_Kernel(
    grid        :: Grid{T}                                                          ,
    kspha       :: AbstractArray{T, 2}                                              , 
    times       :: AbstractVector{T}                                                ;
    fieldmap    :: AbstractArray{T, 2}  = zeros(T,(grid.nX, grid.nY))               , 
    csm         :: Array{Complex{T}, 3} = ones(Complex{T},(grid.nX, grid.nY)..., 1) , 
    recon_terms :: String               = nothing                                   ,
    k_nominal   :: AbstractArray{T, 2}  = kspha[2:4, :]                             ,
    kspha_dt                            = nothing                                   ,
    nBlock      :: Int64                = 50                                        , 
    use_gpu     :: Bool                 = true                                      , 
    gpus        :: Vector{Int}          = [0]                                       ,
    verbose     :: Bool                 = false                                     , 
    ) where {T<:AbstractFloat}

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm, 3)
    nRow = nSam * nCha
    nCol = nVox = prod(grid.matrixSize)
    if verbose
        @info "HighOrderOp_Kernel nRow=$nRow, nCol=$nCol, nSam=$nSam, nCha=$nCha, nBlock=$nBlock, use_gpu=$use_gpu"
    end

    @assert nTerm              in [9, 16]   "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3         "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY)  "FieldMap must have same size as $((nX, nY)) in grid"
    @assert size(csm)[1:2]     == (nX, nY)  "Coil-SensitivityMap must have same size as $((nX, nY)) in grid"
    
    # prepare data 
    kspha    = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    csm      = reshape(csm, nX*nY, nCha)      # [nX * nY, nCha]
    fieldmap = vec(fieldmap)                  # [nVox]

    # divide the calculation into blocks (nBlock) to avoid memory overflow
    nBlock = nBlock > nSam  ? nSam  : nBlock  # nBlock must be <= k
    n      = nSam ÷ nBlock                    # number of sampless per block
    parts  = [n for i=1:nBlock]               # number of samples per block
    parts  = [1+n*(i-1):n*i for i=1:nBlock]
    if nSam%nBlock!= 0
        push!(parts, n*nBlock+1:nSam)
    end

    # compute basis functions (spherical harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, collect(1:nTerm))

    # if use_gpu, move all the variables to GPU
    if use_gpu
        kspha       = kspha       |> gpu
        kspha_dt    = kspha_dt    |> gpu
        grid        = grid        |> gpu
        times       = times       |> gpu
        fieldmap    = fieldmap    |> gpu
        csm         = csm         |> gpu
    end

    if use_gpu
        bf_d       = Dict{Int, CuArray{T, 2}}()
        kspha_d    = Dict{Int, CuArray{T, 2}}()
        times_d    = Dict{Int, CuArray{T, 1}}()
        fieldmap_d = Dict{Int, CuArray{T, 1}}()
        csm_d      = Dict{Int, CuArray{Complex{T}, 2}}()

        for gpu_id in gpus
            CUDA.device!(gpu_id)
            bf_d[gpu_id]       = bf       |> gpu
            kspha_d[gpu_id]    = kspha    |> gpu
            times_d[gpu_id]    = times    |> gpu
            fieldmap_d[gpu_id] = fieldmap |> gpu
            csm_d[gpu_id]      = csm      |> gpu
        end
    end
    if isnothing(kspha_dt)
        func_prod = (res,xm)->(res .= prod_HighOrderOp_Kernel(xm, csm_d, times_d, fieldmap_d, bf_d, kspha_d, nSam, nCha, nTerm, nVox; 
                                            gpus=gpus, verbose=verbose))
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod = (res,xm)->(res .= prod_dt_HighOrderOp_Kernel(xm, bf, nVox, nSam, nCha, nTerm, kspha, kspha_dt, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    end
    func_ctprod = (res,ym)->(res .= ctprod_HighOrderOp_Kernel(ym, csm_d, times_d, fieldmap_d, bf_d, kspha_d, nSam, nCha, nTerm, nVox; 
                                            gpus=gpus, verbose=verbose))
    
    return HighOrderOp_Kernel{Complex{T},Nothing,Function}(
                        nRow, nCol, 
                        false, false,
                        func_prod, nothing, func_ctprod,
                        0, 0, 0, 
                        false, false, false, 
                        Complex{T}[], Complex{T}[])
end


"""
    prod_dt_HighOrderOp_Kernel
    for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
"""
function prod_dt_HighOrderOp_Kernel(
    x         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               ,
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    nTerm     :: Int64                               ,
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
        @info "HighOrderOp_Kernel prod_dt nBlock=$nBlock, use_gpu=$use_gpu"
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
    Forward operator for HighOrderOp_Kernel
"""
function prod_HighOrderOp_Kernel(
    x         :: AbstractVector{T}                         ,   # [nVox] 
    csm       :: Dict{Int, <:AbstractArray{Complex{D}, 2}} ,   # [nVox, nCha]
    times     :: Dict{Int, <:AbstractVector{D}}            ,   # [nSam] 
    fieldmap  :: Dict{Int, <:AbstractVector{D}}            ,   # [nVox]
    bf        :: Dict{Int, <:AbstractArray{D, 2}}          ,   # [nVox, nTerm]
    kspha     :: Dict{Int, <:AbstractArray{D, 2}}          ,   # [nTerm, nSam] 
    nSam      :: Int64                                     , 
    nCha      :: Int64                                     ,
    nTerm     :: Int64                                     ,
    nVox      :: Int64                                     ;
    gpus      :: Vector{Int} = [0]                         ,
    verbose   :: Bool = false                              ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    if verbose
        @info "HighOrderOp: Kernel-based prod"
    end
    t_total = @elapsed begin
        x = Vector(x)

        out = zeros(Complex{D}, nSam, nCha)
        nGPU = length(gpus)
        nSamplePerGPU = cld(nSam, nGPU)
        tasks = []

        for (i, gpu_id) in enumerate(gpus)
            i_s = (i-1)*nSamplePerGPU + 1
            i_e = min(i*nSamplePerGPU, nSam)
            if i_s > nSam
                continue
            end

            push!(tasks, Threads.@spawn begin
                CUDA.device!(gpu_id)
                
                x_i        = x |> gpu
                csm_i      = csm[gpu_id] 
                times_i    = @view times[gpu_id][i_s:i_e] 
                fieldmap_i = fieldmap[gpu_id]
                bf_i       = bf[gpu_id]
                kspha_i    = @view kspha[gpu_id][:, i_s:i_e]
                out_i      = CUDA.zeros(Complex{D}, i_e-i_s+1, nCha)

                CUDA.@sync out_i = HighOrderMRI.run_kernel_prod!(
                    out_i, x_i, csm_i, times_i, fieldmap_i, bf_i, kspha_i,
                    i_e-i_s+1, nCha, nTerm, nVox)
                out[i_s:i_e, :] .= Array(out_i)
            end)
        end

        for t in tasks
            fetch(t)
        end

        out ./= sqrt(nVox)
    end
    if verbose
        println("runtime: $(round(t_total, digits=5)) [s]")
    end
    return vec(out)
end


"""
    Adjoint of prod_HighOrderOp_Kernel
"""
function ctprod_HighOrderOp_Kernel(
    y         :: AbstractVector{T}                         ,   # [nSam, nCha] 
    csm       :: Dict{Int, <:AbstractArray{Complex{D}, 2}} ,   # [nVox, nCha] 
    times     :: Dict{Int, <:AbstractVector{D}}            ,   # [nSam] 
    fieldmap  :: Dict{Int, <:AbstractVector{D}}            ,   # [nVox]
    bf        :: Dict{Int, <:AbstractArray{D, 2}}          ,   # [nVox, nTerm]
    kspha     :: Dict{Int, <:AbstractArray{D, 2}}          ,   # [nTerm, nSam] 
    nSam      :: Int64                                     , 
    nCha      :: Int64                                     ,
    nTerm     :: Int64                                     ,
    nVox      :: Int64                                     ;
    gpus      :: Vector{Int} = [0]                         ,
    verbose   :: Bool = false                              ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    if verbose
        @info "HighOrderOp: Kernel-based ctprod"
    end
    t_total = @elapsed begin
        csmC = conj.(Array(csm[first(keys(csm))]))
        y    = reshape(y, nSam, nCha)

        out = zeros(Complex{D}, nVox, nCha)
        nGPU = length(gpus)
        nVoxPerGPU = cld(nVox, nGPU)
        tasks = []

        for (i, gpu_id) in enumerate(gpus)
            i_s = (i-1)*nVoxPerGPU + 1
            i_e = min(i*nVoxPerGPU, nVox)
            if i_s > nVox
                continue
            end

            push!(tasks, Threads.@spawn begin
                CUDA.device!(gpu_id)
                
                y_i        = y |> gpu
                csm_i      = @view csm[gpu_id][i_s:i_e, :]
                times_i    = times[gpu_id]
                fieldmap_i = @view fieldmap[gpu_id][i_s:i_e]
                bf_i       = @view bf[gpu_id][i_s:i_e, :]
                kspha_i    = kspha[gpu_id]
                out_i      = CUDA.zeros(Complex{D}, i_e-i_s+1, nCha)

                CUDA.@sync out_i = run_kernel_ctprod!(
                    out_i, y_i, csm_i, times_i, fieldmap_i, bf_i, kspha_i,
                    nSam, nCha, nTerm, i_e-i_s+1)
                out[i_s:i_e, :] .= Array(out_i)
            end)
        end

        for t in tasks
            fetch(t)
        end

        out = out ./ sqrt(nVox)
        out = out .* csmC
    end
    if verbose
        println("runtime: $(round(t_total, digits=5)) [s]")
    end
    return vec(sum(out, dims=2))
end


function Base.adjoint(op::HighOrderOp_Kernel{T}) where T
  return LinearOperator{T}(
                            op.ncol, 
                            op.nrow, 
                            op.symmetric, 
                            op.hermitian,
                            op.ctprod!, 
                            nothing, 
                            op.prod!)
end