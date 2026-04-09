using CUDA
using LinearAlgebra
using NFFT

export HighOrderNFFTOp

"""
HighOrderNFFTOp implements a high-order field encoding operator based on SVD low-rank approximation.
Automatically separates 0th-order (outer demodulation) and 1st-order (NFFT) terms, and performs extremely fast SVD dimensionality reduction only on the remaining smooth high-order errors.
Supports automatic switching between CPU and GPU compute backends.
"""
mutable struct HighOrderNFFTOp{T, F1, F2, P<:AbstractNFFTPlan, AM<:AbstractMatrix{Complex{T}}, S<:AbstractVector{Complex{T}}} <: HOOp{Complex{T}}
    nSam::Int
    nVox::Int
    nCha::Int
    L::Int                           # SVD truncation rank
    
    u::AM                            # [nSam, L]
    v_star::AM                       # [nVox, L]
    v::AM                            # [nVox, L]
    csm::AM                          # [nVox, nCha]
    
    nfftplan::P                      # NFFT plan adapted for CPU/GPU
    mask_idx::AbstractVector{Int32}  # Extracted 3D mask indices
    grid_size::Tuple                 # gridding size

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
LinearOperators.storage_type(op::HighOrderNFFTOp) = typeof(op.Mv)


function HighOrderNFFTOp(
    grid        :: Grid{T}                                                             ,
    kspha       :: AbstractArray{T, 2}                                                 , 
    times       :: AbstractVector{T}                                                   ;
    fieldmap    :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm         :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask        :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    recon_terms :: String                    = nothing                                 ,
    k_nominal   :: AbstractArray{T, 2}       = kspha[2:4, :]                           ,
    kspha_dt                                 = nothing                                 ,
    nBlock      :: Int64                     = 50                                      , 
    
    gpus        :: Vector{Int}               = [0]                                     ,
    L_rank      :: Int                       = 15                                      , 
    arrayType   :: Type{<:AbstractArray}     = Array                                   ,
    verbose     :: Bool                      = false                                   ,   
    
    kwargs...          
) where T

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm)[end]
    nRow = nSam * nCha
    nCol = prod(grid.matrixSize)
    nVox = sum(mask)

    fieldmap = ndims(fieldmap) == 2 ? reshape(fieldmap, nX, nY, 1) : fieldmap
    csm      = ndims(csm) == 3      ? reshape(csm, nX, nY, 1, nCha) : csm
    mask     = ndims(mask) == 2     ? reshape(mask, nX, nY, 1) : mask

    if verbose @info "HighOrderNFFTOp Setup: ArrayType=$arrayType, nRow=$(nSam*nCha), nCol=$(prod(grid.matrixSize)), nVox in mask=$nVox, gpus=$gpus" end

    @assert nTerm              in [9, 16]       "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3             "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY, nZ)  "FieldMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(csm)[1:3]     == (nX, nY, nZ)  "Coil-SensitivityMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(mask)         == (nX, nY, nZ)  "Mask must have same size as $((nX, nY, nZ)) in grid"
    
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
        kspha_err = arrayType(kspha[5:end, :])
        bf_err    = arrayType(bf[:, 5:end])
    else
        kspha_err = arrayType(zeros(T, 1, nSam))
        bf_err    = arrayType(zeros(T, nVox, 1))
    end

    u_trunc, s_trunc, v_trunc = perform_rsvd(times, fieldmap, bf_err, kspha_err, nVox, nSam, L_rank)
    v_scaled = v_trunc * Diagonal(s_trunc)

    if nZ == 1
        k1_traj = kspha[2:3, :] ./ (1/min(grid.Δx, grid.Δy)); MatrixSize = (nX, nY);
    else
        k1_traj = kspha[2:4, :] ./ (1/min(grid.Δx, grid.Δy, grid.Δz)); MatrixSize = (nX, nY, nZ);
    end

    nfftplan = arrayType <: CuArray ? plan_nfft(CuArray, k1_traj, MatrixSize; m=3, σ=1.25, precompute=NFFT.TENSOR) : 
    plan_nfft(k1_traj, MatrixSize; m=3, σ=1.25, precompute=NFFT.TENSOR)

    Mv = Mtu = arrayType(Vector{Complex{T}}(undef, 0))  
    return HighOrderNFFTOp{T, Nothing, typeof(ctprod!), typeof(nfftplan), typeof(csm), typeof(Mv)}(
        nSam, nVox, nCha, L_rank,
        arrayType(u_trunc), arrayType(conj.(v_scaled)), arrayType(v_scaled), csm,
        nfftplan, mask_idx, grid.matrixSize,
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
function prod!(y::AbstractVector{Complex{T}}, op::HighOrderNFFTOp{T}, x::AbstractVector{Complex{T}}) where T
    x_masked = length(x) == prod(op.grid_size) ? view(x, op.mask_idx) : x
    y = reshape(y, op.nSam, op.nCha)
    fill!(y, 0) 
    
    grid_flat = fill!(similar(op.v, Complex{T}, prod(op.grid_size)), 0)
    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
    nfft_img  = reshape(grid_flat, nfft_dims)
    
    k_signal   = similar(op.v, Complex{T}, op.nSam)
    x_weighted = similar(op.v, Complex{T}, op.nVox)

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    for l = 1:op.L
        @views @. x_weighted = x_masked * op.v_star[:, l]
        
        for c = 1:op.nCha
            if grid_flat isa CuArray
                @cuda threads=threads blocks=blocks_vox kernel_scatter_csm!(
                    grid_flat, x_weighted, op.csm, c, op.mask_idx, op.nVox
                )
            else
                @views @. grid_flat[op.mask_idx] = x_weighted * op.csm[:, c]
            end
            
            mul!(k_signal, op.nfftplan, nfft_img)
            @views @. y[:, c] += k_signal * op.u[:, l]
        end
    end
    
    return y
end

function ctprod!(x::AbstractVector{Complex{T}}, op::HighOrderNFFTOp{T}, y::AbstractVector{Complex{T}}) where T
    y = reshape(y, op.nSam, op.nCha)
    x_masked = fill!(similar(op.v, Complex{T}, op.nVox), 0)
    
    grid_flat = similar(op.v, Complex{T}, prod(op.grid_size))
    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
    nfft_img  = reshape(grid_flat, nfft_dims)
    
    k_weighted = similar(op.v, Complex{T}, op.nSam)
    x_l        = similar(op.v, Complex{T}, op.nVox)

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    for l = 1:op.L
        fill!(x_l, 0)
        
        for c = 1:op.nCha
            @views @. k_weighted = y[:, c] * conj(op.u[:, l])
            mul!(nfft_img, adjoint(op.nfftplan), k_weighted)
            
            if grid_flat isa CuArray
                @cuda threads=threads blocks=blocks_vox kernel_gather_csm!(
                    x_l, grid_flat, op.csm, c, op.mask_idx, op.nVox
                )
            else
                @views @. x_l += grid_flat[op.mask_idx] * conj(op.csm[:, c])
            end
        end
        @views @. x_masked += x_l * op.v[:, l]
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
struct AdjointHighOrderNFFTOp{T, P, AM, AV}
    op::HighOrderNFFTOp{T, P, AM, AV}
end

Base.adjoint(op::HighOrderNFFTOp) = AdjointHighOrderNFFTOp(op)

Base.eltype(::HighOrderNFFTOp{T}) where T = Complex{T}
Base.eltype(::AdjointHighOrderNFFTOp{T}) where T = Complex{T}

function Base.size(op::HighOrderNFFTOp)
    return (op.nSam * op.nCha, prod(op.grid_size))
end

function Base.size(op::HighOrderNFFTOp, dim::Int)
    return dim == 1 ? op.nSam * op.nCha : prod(op.grid_size)
end

function Base.size(adj::AdjointHighOrderNFFTOp, dim::Int)
    return dim == 1 ? prod(adj.op.grid_size) : adj.op.nSam * adj.op.nCha
end

import LinearAlgebra: mul!
function mul!(y::AbstractVector{Complex{T}}, op::HighOrderNFFTOp{T}, x::AbstractVector{Complex{T}}) where T
    prod!(y, op, x)
end

function mul!(x::AbstractVector{Complex{T}}, adj::AdjointHighOrderNFFTOp{T}, y::AbstractVector{Complex{T}}) where T
    ctprod!(x, adj.op, y)
end

function Base.:*(op::HighOrderNFFTOp{T}, x::AbstractVector{Complex{T}}) where T
    y = fill!(similar(op.v, Complex{T}, op.nSam * op.nCha), 0)
    mul!(y, op, x)
    return y
end

function Base.:*(adj::AdjointHighOrderNFFTOp{T}, y::AbstractVector{Complex{T}}) where T
    x = fill!(similar(adj.op.v, Complex{T}, prod(adj.op.grid_size)), 0)
    mul!(x, adj, y)
    return x
end
