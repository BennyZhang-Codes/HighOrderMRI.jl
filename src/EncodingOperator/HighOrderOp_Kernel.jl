export HighOrderOp_Kernel

"""
Kernel-based implementation of the explicit high-order signal encoding model.

The forward phase convention is `exp(+2πim * phase)`, the operator
normalization is `1 / sqrt(nVox)`, and the vectorized output ordering is
samples first and then channels.

- Supports 2D or 3D reconstruction with up to third-order dynamic fields.
- Supports CUDA kernel acceleration and multiple GPUs.
- Supports multiple receive channels, static off-resonance, and masking.
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
    Mv         :: Vector{T}
    Mtu        :: Vector{T}
end
LinearOperators.storage_type(op::HighOrderOp_Kernel) = typeof(op.Mv)


"""
    HighOrderOp_Kernel(grid::Grid{T}, kspha::AbstractArray{T, 2}, times::AbstractVector{T}; kwargs...)

# Description

Construct a `HighOrderOp_Kernel` that explicitly evaluates the MRI high-order
Fourier encoding operator.

# Arguments

* `grid::Grid{T}`                      - Cartesian reconstruction grid.
* `kspha::AbstractArray{T, 2}`         - [nTerm, nSam], coefficients of field
  dynamics. `nTerm` must be `9` (up to second order) or `16` (up to third
  order).
* `times::AbstractVector{T}`           - [nSam], sampling time points.

# Keywords

* `fieldmap::AbstractArray{T}`          - [nX, nY, nZ], off-resonance map.
  [nX, nY] is accepted when `nZ == 1`.
* `csm::AbstractArray{Complex{T}}`      - [nX, nY, nZ, nCha], complex coil
  sensitivity maps. [nX, nY, nCha] is accepted when `nZ == 1`.
* `mask::AbstractArray{Bool}`           - [nX, nY, nZ], reconstruction mask.
  [nX, nY] is accepted when `nZ == 1`.
* `recon_terms::Union{Nothing, AbstractString}` - Binary order-selection
  string. Use three digits for `nTerm == 9` and four digits for
  `nTerm == 16`; the digits select zeroth-, first-, second-, and third-order
  terms. The default is `"111"` or `"1111"`.
* `k_nominal::AbstractArray{T, 2}`      - [3, nSam], nominal first-order
  trajectory ordered as [kx, ky, kz]. It replaces the measured first-order
  coefficients when first-order error is disabled by `recon_terms`.
* `kspha_dt`                            - [nTerm, nSam], optional time
  derivative of the field-dynamic coefficients.
* `nBlock::Int64`                       - Number of trajectory blocks used by
  the derivative path to limit temporary memory.
* `use_gpu::Bool`                       - Use CUDA execution; default is
  `true`.
* `gpus::Vector{Int}`                   - Zero-based CUDA device IDs.
* `verbose::Bool`                       - Print progress information; default
  is `false`.
"""
function HighOrderOp_Kernel(
    grid        :: Grid{T}                                                             ,
    kspha       :: AbstractArray{T, 2}                                                 , 
    times       :: AbstractVector{T}                                                   ;
    fieldmap    :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm         :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask        :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    recon_terms :: Union{Nothing,AbstractString} = nothing                            ,
    k_nominal   :: AbstractArray{T, 2}       = kspha[2:4, :]                           ,
    kspha_dt                                 = nothing                                 ,
    nBlock      :: Int64                     = 50                                      , 
    use_gpu     :: Bool                      = true                                    , 
    gpus        :: Vector{Int}               = [0]                                     ,
    verbose     :: Bool                      = false                                   , 
    ) where {T<:AbstractFloat}

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm)[end]
    nRow = nSam * nCha
    nCol = prod(grid.matrixSize)
    nVox = sum(mask)

    fieldmap = ndims(fieldmap) == 2 ? reshape(fieldmap, nX, nY, 1) : fieldmap
    csm      = ndims(csm) == 3      ? reshape(csm, nX, nY, 1, nCha) : csm
    mask     = ndims(mask) == 2     ? reshape(mask, nX, nY, 1) : mask

    @info "HighOrderOp_Kernel nRow=$nRow [nSam*nCha=$nSam*$nCha], nCol=$nCol [prod(matrixSize=$((nX,nY,nZ))], nVox in mask=$nVox, nBlock=$nBlock, gpus=$gpus"

    @assert nTerm              in [9, 16]       "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3             "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY, nZ)  "FieldMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(csm)[1:3]     == (nX, nY, nZ)  "Coil-SensitivityMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(mask)         == (nX, nY, nZ)  "Mask must have same size as $((nX, nY, nZ)) in grid"
    
    mask     = vec(mask)                               # [prod(MatrixSize)]

    # prepare data 
    kspha    = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    csm      = reshape(csm, :, nCha)[mask.!=0, :]      # [nVox, nCha]
    fieldmap = vec(fieldmap)[mask.!=0]                 # [nVox]

    # divide the calculation into blocks (nBlock) to avoid memory overflow
    nBlock = nBlock > nSam  ? nSam  : nBlock  # nBlock must be <= k
    n      = nSam ÷ nBlock                    # number of sampless per block
    parts  = [n for i=1:nBlock]               # number of samples per block
    parts  = [1+n*(i-1):n*i for i=1:nBlock]
    if nSam%nBlock!= 0
        push!(parts, n*nBlock+1:nSam)
    end
    
    # compute basis functions (spherical harmonics)
    bf = basisfunc_spha(grid.x[mask.!=0], grid.y[mask.!=0], grid.z[mask.!=0], collect(1:nTerm))

    # if use_gpu, move all the variables to GPU
    if use_gpu
        kspha       = kspha       |> gpu
        kspha_dt    = kspha_dt    |> gpu
        bf          = bf          |> gpu
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
        func_prod = (res,xm)->(res .= prod_HighOrderOp_Kernel(xm, mask, csm_d, times_d, fieldmap_d, bf_d, kspha_d, nSam, nCha, nTerm, nVox; 
                                            gpus=gpus, verbose=verbose))
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod = (res,xm)->(res .= prod_dt_HighOrderOp(xm, mask, bf, nVox, nSam, nCha, kspha, kspha_dt, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    end
    func_ctprod = (res,ym)->(res .= ctprod_HighOrderOp_Kernel(ym, mask, csm_d, times_d, fieldmap_d, bf_d, kspha_d, nSam, nCha, nTerm, nVox; 
                                            gpus=gpus, verbose=verbose))
    
    return HighOrderOp_Kernel{Complex{T},Nothing,Function}(
                        nRow, nCol, 
                        false, false,
                        func_prod, nothing, func_ctprod,
                        0, 0, 0, 
                        Complex{T}[], Complex{T}[])
end


"""
    Forward operator for HighOrderOp_Kernel
"""
function prod_HighOrderOp_Kernel(
    x         :: AbstractVector{T}                         ,   # [prod(MatrixSize)] 
    mask      :: AbstractVector{Bool}                      ,   # [prod(MatrixSize)]
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
        x = Vector(x)[mask.!=0]

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
    mask      :: AbstractVector{Bool}                      ,   # [prod(MatrixSize)]
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
    x = zeros(Complex{D}, size(mask))
    x[mask] .= vec(sum(out, dims=2))
    return x
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
