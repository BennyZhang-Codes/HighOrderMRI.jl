using CUDA
using LinearAlgebra
using NFFT

export HighOrderLowRankOp

mutable struct HighOrderLowRankWorkspace{AV}
    grid_flat_prod   :: AV
    grid_flat_ctprod :: AV
    k_signal         :: AV
    k_weighted       :: AV
    x_weighted       :: AV
    x_masked         :: AV
    x_l              :: AV
end

"""
HighOrderLowRankOp implements a high-order field encoding operator based on SVD low-rank approximation.
Automatically separates 0th-order (outer demodulation) and 1st-order (NFFT) terms, and performs extremely fast SVD dimensionality reduction only on the remaining smooth high-order errors.
Supports automatic switching between CPU and GPU compute backends.
"""
mutable struct HighOrderLowRankOp{
    T, F1, F2, 
    P<:AbstractNFFTPlan, 
    AM<:AbstractArray{Complex{T}, 3}, 
    AN<:AbstractArray{Complex{T}, 2},
    W<:HighOrderLowRankWorkspace,
    S<:AbstractVector{Complex{T}}
    } <: HOOp{Complex{T}}
    nSam       :: Int
    nVox       :: Int
    nCha       :: Int
    nDyn       :: Int
    L          :: Int                    # SVD truncation rank
    
    u          :: AM                     # [nSam, L, nDyn]
    v_star     :: AM                     # [nVox, L, nDyn]
    v          :: AM                     # [nVox, L, nDyn]
    csm        :: AN                     # [nVox, nCha]
    
    nfftplan   :: Vector{P}              # NFFT plan adapted for CPU/GPU
    mask_idx   :: AbstractVector{Int32}  # Extracted 3D mask indices
    grid_size  :: Tuple                  # gridding size

    workspace  :: W

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
    Mv         :: S                  # Dynamically adapted CPU/GPU Vector
    Mtu        :: S                  # Dynamically adapted CPU/GPU Vector
end
LinearOperators.storage_type(op::HighOrderLowRankOp) = typeof(op.Mv)


function HighOrderLowRankOp(
    grid        :: Grid{T}                                                             ,
    kspha       :: AbstractArray{T, 2}                                                 , 
    times       :: AbstractArray{T, 1}                                                 ;
    kwargs...          
    ) where T <: AbstractFloat
    kspha = reshape(kspha, size(kspha, 1), size(kspha, 2), 1)
    times = reshape(times, :, 1)
    return HighOrderLowRankOp(grid, kspha, times; kwargs...)
end

"""
kspha: [nTerm, nSam, nDyn]
times: [nSam, nDyn]
"""
function HighOrderLowRankOp(
    grid            :: Grid{T}                                                             ,
    kspha           :: AbstractArray{T, 3}                                                 , 
    times           :: AbstractArray{T, 2}                                                 ;
    fieldmap        :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm             :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask            :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    recon_terms     :: String                    = nothing                                 ,
    k_nominal       :: AbstractArray{T, 3}       = kspha[2:4, :, :]                        ,
    kspha_dt                                     = nothing                                 ,
    nBlock          :: Int64                     = 50                                      , 
    
    gpus            :: Vector{Int}               = [0]                                     ,
    L_rank          :: Int                       = 15                                      , 
    arrayType       :: Type{<:AbstractArray}     = Array                                   ,
    rsvd_seed       :: Int                       = 1234                                    ,
    rsvd_chunk      :: Int                       = 4096                                    , 
    rsvd_oversample :: Int                       = 5                                       ,
    verbose         :: Bool                      = false                                   ,   
    
    kwargs...          
    ) where T <: AbstractFloat

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam, nDyn = size(kspha)
    nCha = size(csm)[end]
    nRow = nSam * nCha * nDyn
    nCol = prod(grid.matrixSize)
    nVox = sum(mask)

    fieldmap = ndims(fieldmap) == 2 ? reshape(fieldmap, nX, nY, 1) : fieldmap
    csm      = ndims(csm) == 3      ? reshape(csm, nX, nY, 1, nCha) : csm
    mask     = ndims(mask) == 2     ? reshape(mask, nX, nY, 1) : mask

    if verbose @info "HighOrderLowRankOp Setup: ArrayType=$arrayType, nRow=$(nSam*nCha), nCol=$(prod(grid.matrixSize)), nVox in mask=$nVox, gpus=$gpus" end

    @assert nTerm              in [9, 16]       "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3             "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY, nZ)  "FieldMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(csm)[1:3]     == (nX, nY, nZ)  "Coil-SensitivityMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(mask)         == (nX, nY, nZ)  "Mask must have same size as $((nX, nY, nZ)) in grid"
    @assert size(times)        == (nSam, nDyn)  "times must have size $((nSam, nDyn))"
    
    # prepare data 
    mask     = vec(mask)                               # [prod(MatrixSize)]
    kspha    = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    csm      = reshape(csm, :, nCha)[mask.!=0, :]      # [nVox, nCha]
    fieldmap = vec(fieldmap)[mask.!=0]                 # [nVox]
    bf       = basisfunc_spha(grid.x[mask.!=0], grid.y[mask.!=0], grid.z[mask.!=0], collect(1:nTerm))  # compute basis functions (spherical harmonics)

    mask_idx = Int32.(findall(mask .!= 0))
    mask_idx = arrayType(mask_idx)

    times     = arrayType(times)
    fieldmap  = arrayType(fieldmap)
    bf        = arrayType(bf)

    csm       = arrayType(csm)


    if nTerm >= 5
        kspha_err = arrayType(kspha[5:end, :, :])
        bf_err    = arrayType(bf[:, 5:end])
    else
        kspha_err = arrayType(zeros(T, 1, nSam, nDyn))
        bf_err    = arrayType(zeros(T, nVox, 1))
    end
    @info "nSam=$nSam, nVox=$nVox, nDyn=$nDyn, nTerm=$nTerm, nCha=$nCha, nBlock=$nBlock"
    u      = arrayType(zeros(Complex{T}, nSam, L_rank, nDyn))
    v      = arrayType(zeros(Complex{T}, nVox, L_rank, nDyn))
    v_star = arrayType(zeros(Complex{T}, nVox, L_rank, nDyn))

    @assert rsvd_chunk > 0 "rsvd_chunk must be positive for chunked rSVD"
    L_total = L_rank + rsvd_oversample
    rsvd_chunk = min(rsvd_chunk, nVox)
    rsvd_workspace = RSVDWorkspace(u, T, nSam, nVox, L_total, rsvd_chunk)
    for dyn = 1:nDyn
        times_dyn     = times[:, dyn]
        kspha_err_dyn = kspha_err[:, :, dyn]
        u_trunc, s_trunc, v_trunc = perform_rsvd(times_dyn, fieldmap, bf_err, kspha_err_dyn, nVox, nSam, L_rank, rsvd_chunk, rsvd_workspace; 
            seed=rsvd_seed + dyn - 1, p_oversample=rsvd_oversample)
        v_scaled = v_trunc * Diagonal(s_trunc)

        @views u[:, :, dyn]      .= arrayType(u_trunc)
        @views v[:, :, dyn]      .= arrayType(v_scaled)
        @views v_star[:, :, dyn] .= arrayType(conj.(v_scaled))
    end


    if nZ == 1
        MatrixSize = (nX, nY)
        scale = 1 / min(grid.Δx, grid.Δy)
        k_range = 2:3
    else
        MatrixSize = (nX, nY, nZ)
        scale = 1 / min(grid.Δx, grid.Δy, grid.Δz)
        k_range = 2:4
    end

    k1_traj_1 = kspha[k_range, :, 1] ./ scale
    nfftplan_1 = plan_nfft(arrayType, k1_traj_1, MatrixSize; m=3, σ=1.25, precompute=NFFT.TENSOR)
    nfftplan = Vector{typeof(nfftplan_1)}(undef, nDyn)
    nfftplan[1] = nfftplan_1
    for dyn = 2:nDyn
        k1_traj = kspha[k_range, :, dyn] ./ scale
        nfftplan[dyn] = plan_nfft(arrayType, k1_traj, MatrixSize; m=3, σ=1.25, precompute=NFFT.TENSOR)
    end

    nGrid = prod(grid.matrixSize)
    workspace = HighOrderLowRankWorkspace(
        fill!(similar(u, Complex{T}, nGrid), 0),
        similar(u, Complex{T}, nGrid),
        similar(u, Complex{T}, nSam),
        similar(u, Complex{T}, nSam),
        similar(u, Complex{T}, nVox),
        similar(u, Complex{T}, nVox),
        similar(u, Complex{T}, nVox),
    )

    Mv = Mtu = arrayType(Vector{Complex{T}}(undef, 0))  
    return HighOrderLowRankOp{T, Nothing, typeof(ctprod!), typeof(nfftplan_1), typeof(u), typeof(csm), typeof(workspace), typeof(Mv)}(
        nSam, nVox, nCha, nDyn, L_rank,
        u, v_star, v, csm,
        nfftplan, mask_idx, grid.matrixSize, workspace,
        nRow, nCol, false, false, prod!, nothing, ctprod!,
        0, 0, 0, 
        Mv, Mtu,
    )
end

# -------------------------------------------------------------------------
# Fused Kernels
# -------------------------------------------------------------------------
# Kernel fusion: Computes x_masked * v_star * csm and writes to the grid in a single step, saving an array read/write!
function kernel_scatter_csm!(grid_flat, x_w, csm, c, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds idx = mask_idx[v]
        @inbounds grid_flat[idx] = x_w[v] * csm[v, c]
    end
    return nothing
end

function kernel_gather_csm!(x_l, img_flat, csm, c, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds idx = mask_idx[v]
        @inbounds x_l[v] += img_flat[idx] * conj(csm[v, c])
    end
    return nothing
end


# -------------------------------------------------------------------------
# Zero-allocation prod! and ctprod!
# -------------------------------------------------------------------------
function prod!(y::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T

    workspace  = op.workspace
    grid_flat  = workspace.grid_flat_prod
    k_signal   = workspace.k_signal
    x_weighted = workspace.x_weighted

    x_masked = length(x) == prod(op.grid_size) ? view(x, op.mask_idx) : x
    y = reshape(y, op.nSam, op.nCha, op.nDyn)
    fill!(y, 0) 
    
    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
    nfft_img  = reshape(grid_flat, nfft_dims)
    
    threads = 256
    blocks_vox = cld(op.nVox, threads)

    for dyn = 1:op.nDyn
        for l = 1:op.L
            @views @. x_weighted = x_masked * op.v_star[:, l, dyn]
            
            for c = 1:op.nCha
                if grid_flat isa CuArray
                    @cuda threads=threads blocks=blocks_vox kernel_scatter_csm!(
                        grid_flat, x_weighted, op.csm, c, op.mask_idx, op.nVox
                    )
                else
                    @views @. grid_flat[op.mask_idx] = x_weighted * op.csm[:, c]
                end
                
                mul!(k_signal, op.nfftplan[dyn], nfft_img)
                @views @. y[:, c, dyn] += k_signal * op.u[:, l, dyn]
            end
        end
    end
    
    return y
end

function ctprod!(x::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, y::AbstractVector{Complex{T}}) where T

    y = reshape(y, op.nSam, op.nCha, op.nDyn)

    workspace  = op.workspace
    grid_flat  = workspace.grid_flat_ctprod
    k_weighted = workspace.k_weighted
    x_masked   = workspace.x_masked
    x_l        = workspace.x_l

    fill!(x_masked, 0)

    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
    nfft_img  = reshape(grid_flat, nfft_dims)

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    for dyn = 1:op.nDyn
        for l = 1:op.L
            fill!(x_l, 0)
            
            for c = 1:op.nCha
                @views @. k_weighted = y[:, c, dyn] * conj(op.u[:, l, dyn])
                mul!(nfft_img, adjoint(op.nfftplan[dyn]), k_weighted)
                
                if grid_flat isa CuArray
                    @cuda threads=threads blocks=blocks_vox kernel_gather_csm!(
                        x_l, grid_flat, op.csm, c, op.mask_idx, op.nVox
                    )
                else
                    @views @. x_l += grid_flat[op.mask_idx] * conj(op.csm[:, c])
                end
            end
            @views @. x_masked += x_l * op.v[:, l, dyn]
        end
    end
    
    if length(x) == prod(op.grid_size)
        fill!(x, 0)
        view(x, op.mask_idx) .= x_masked
    else
        x .= x_masked
    end
    
    return x
end

# -------------------------------------------------------------------------
# Adjoint Operator Definition
# -------------------------------------------------------------------------
Base.eltype(::HighOrderLowRankOp{T}) where T = Complex{T}

function Base.adjoint(op::HighOrderLowRankOp{T}) where T
    return LinearOperator{Complex{T}}(
                              op.ncol, 
                              op.nrow, 
                              op.symmetric, 
                              op.hermitian,
                              (res, y) -> ctprod!(res, op, y), 
                              nothing, 
                              (res, x) -> prod!(res, op, x),
                              S=typeof(op.Mv))
  end

function Base.size(op::HighOrderLowRankOp)
    return (op.nSam * op.nCha * op.nDyn, prod(op.grid_size))
end

function Base.size(op::HighOrderLowRankOp, dim::Int)
    return dim == 1 ? op.nSam * op.nCha * op.nDyn : prod(op.grid_size)
end

import LinearAlgebra: mul!
function mul!(y::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T
    prod!(y, op, x)
end

function mul!(
    y::AbstractVector{Complex{T}},
    op::HighOrderLowRankOp{T},
    x::AbstractVector{Complex{T}},
    α,
    β,
) where T
    if iszero(β)
        prod!(y, op, x)
        α == one(α) || (@. y = α * y)
    else
        y_tmp = similar(y)
        prod!(y_tmp, op, x)
        @. y = α * y_tmp + β * y
    end

    return y
end

function Base.:*(op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T
    y = fill!(similar(op.v, Complex{T}, op.nSam * op.nCha * op.nDyn), 0)
    mul!(y, op, x)
    return y
end

