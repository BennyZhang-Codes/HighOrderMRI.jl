using CUDA
using LinearAlgebra
using NFFT
using Random

"""
phase         [nSam, chunk_size] Float32
encoding      [nSam, chunk_size] ComplexF32
omega         [nVox, L_total] ComplexF32
W             [nSam, L_total] ComplexF32
B_adj         [nVox, L_total] ComplexF32
gram          [L_total, L_total]
right_vectors [L_total, L_total]
"""
mutable struct RSVDWorkspace{RM<:AbstractMatrix, CM<:AbstractMatrix}
    phase         :: RM  # [nSam, chunk_size]
    encoding      :: CM  # [nSam, chunk_size]
    omega         :: CM  # [nVox, L_total]
    W             :: CM  # [nSam, L_total]
    B_adj         :: CM  # [nVox, L_total]
    gram          :: CM  # [L_total, L_total]
    right_vectors :: CM  # [L_total, L_total]
end

function RSVDWorkspace(
    prototype,
    ::Type{T},
    nSam      :: Int,
    nVox      :: Int,
    L_total   :: Int,
    chunk_size:: Int,
    ) where T <: AbstractFloat

    return RSVDWorkspace(
        similar(prototype, T, nSam, chunk_size),
        similar(prototype, Complex{T}, nSam, chunk_size),
        similar(prototype, Complex{T}, nVox, L_total),
        similar(prototype, Complex{T}, nSam, L_total),
        similar(prototype, Complex{T}, nVox, L_total),
        similar(prototype, Complex{T}, L_total, L_total),
        similar(prototype, Complex{T}, L_total, L_total),
    )
end

function finalize_rsvd_svd_scaled!(
    v_scaled :: AbstractMatrix{Complex{T}},
    Q        :: AbstractMatrix{Complex{T}},
    B_adj    :: AbstractMatrix{Complex{T}},
    L_rank   :: Int,
) where T <: AbstractFloat

    F = svd!(B_adj; full=false)

    Z = @view F.V[:, 1:L_rank]
    P = @view F.U[:, 1:L_rank]
    s = @view F.S[1:L_rank]

    # U = Q * Z
    u_trunc = Q * Z

    # Vscaled = P * Diagonal(s)
    copyto!(v_scaled, P)
    v_scaled .*= reshape(s, 1, L_rank)

    total_energy = T(real(dot(s, s)))

    return u_trunc, total_energy
end

function finalize_rsvd_gram!(
    v_scaled :: AbstractMatrix{Complex{T}},
    Q        :: AbstractMatrix{Complex{T}},
    B_adj    :: AbstractMatrix{Complex{T}},
    workspace:: RSVDWorkspace,
    L_rank   :: Int;
    allow_fallback::Bool = true
) where T <: AbstractFloat

    gram = workspace.gram

    # G = BᴴB
    mul!(gram, adjoint(B_adj), B_adj)

    gram_cpu = Array(gram)

    gram_cpu = (gram_cpu + adjoint(gram_cpu)) * T(0.5)

    eig = eigen(Hermitian(gram_cpu))

    order = sortperm(real.(eig.values); rev=true)

    values = T.(real.(eig.values[order]))
    vectors_cpu = eig.vectors[:, order]

    if !all(isfinite, values) || !all(isfinite, vectors_cpu)
        if !allow_fallback
            error("Gram finalization failed numerical validation")
        end
        @warn "Non-finite Gram eigendecomposition; falling back to SVD" maxlog=1
        return finalize_rsvd_svd_scaled!(v_scaled, Q, B_adj, L_rank)
    end

    λmax = maximum(abs, values)

    if λmax <= zero(T)
        if !allow_fallback
            error("Gram finalization failed numerical validation")
        end
        @warn "Degenerate Gram matrix; falling back to SVD" maxlog=1
        return finalize_rsvd_svd_scaled!(v_scaled, Q, B_adj, L_rank)
    end

    negative_tol = T(10) * eps(T) * T(size(gram, 1)) * λmax

    if minimum(values) < -negative_tol
        if !allow_fallback
            error("Gram finalization failed numerical validation")
        end
        @warn "Gram matrix has significant negative eigenvalues; falling back to SVD" minimum_eigenvalue=minimum(values) negative_tol maxlog=1
        return finalize_rsvd_svd_scaled!(v_scaled, Q, B_adj, L_rank)
    end

    values .= max.(values, zero(T))

    copyto!(workspace.right_vectors, vectors_cpu)

    Z = @view workspace.right_vectors[:, 1:L_rank]

    # U = QZ
    u_trunc = Q * Z

    # Vscaled = BZ = PΣ
    mul!(v_scaled, B_adj, Z)

    # λ = σ²
    total_energy = T(sum(@view values[1:L_rank]))

    return u_trunc, total_energy
end


function perform_rsvd(
    times         :: Array{T}, 
    fieldmap      :: Array{T}, 
    bf            :: Array{T, 2}, 
    kspha_err     :: Array{T, 2}, 
    nVox          :: Int, 
    nSam          :: Int, 
    L_rank        :: Int,
    chunk_size    :: Int,
    workspace     :: RSVDWorkspace;
    seed          :: Int = 0,
    p_oversample  :: Int = 5,
    rsvd_finalize :: Symbol = :svd,
    rsvd_backend  :: Symbol = :chunked,
    v_scaled                = nothing,
    gram_allow_fallback::Bool = true,
    verbose       :: Bool = false,
    ) where T <: AbstractFloat
    rsvd_backend === :chunked || throw(ArgumentError("rsvd_backend=$rsvd_backend is currently supported only for CuArray"))

    if verbose @info "Performing chunked rSVD on CPU..." backend=rsvd_backend end

    L_total = L_rank + p_oversample
    @assert L_total <= min(nSam, nVox) "rSVD rank exceeds matrix dimensions"
    @assert chunk_size > 0 "chunk_size must be positive"
    chunk_size = min(chunk_size, nVox)

    phase_workspace = workspace.phase
    E_workspace     = workspace.encoding
    Ω_cpu           = workspace.omega
    W_cpu           = workspace.W
    B_adj_cpu       = workspace.B_adj

    fill!(W_cpu, 0)

    rng = Random.Xoshiro(seed)
    randn!(rng, Ω_cpu)

    times_mat = reshape(times, :, 1)

    # First pass: W = E * Ω
    for vox_start = 1:chunk_size:nVox
        vox_stop = min(vox_start + chunk_size - 1, nVox)
        vox_range = vox_start:vox_stop
        nChunk = length(vox_range)

        phase_chunk = @view phase_workspace[:, 1:nChunk]
        E_chunk = @view E_workspace[:, 1:nChunk]

        fieldmap_chunk = @view fieldmap[vox_range]
        bf_chunk = @view bf[vox_range, :]
        Ω_chunk = @view Ω_cpu[vox_range, :]

        mul!(phase_chunk, times_mat, reshape(fieldmap_chunk, 1, :))

        mul!(phase_chunk, transpose(kspha_err), transpose(bf_chunk), one(T), one(T))

        @. E_chunk = cis(T(2π) * phase_chunk)

        mul!(W_cpu, E_chunk, Ω_chunk, one(Complex{T}), one(Complex{T}))
    end

    Q_cpu = Matrix(qr(W_cpu).Q)

    # Second pass: B_adj = E' * Q
    # B_adj_cpu = zeros(Complex{T}, nVox, L_total)

    for vox_start = 1:chunk_size:nVox
        vox_stop = min(vox_start + chunk_size - 1, nVox)
        vox_range = vox_start:vox_stop
        nChunk = length(vox_range)

        phase_chunk = @view phase_workspace[:, 1:nChunk]
        E_chunk = @view E_workspace[:, 1:nChunk]

        fieldmap_chunk = @view fieldmap[vox_range]
        bf_chunk = @view bf[vox_range, :]
        B_adj_chunk = @view B_adj_cpu[vox_range, :]

        mul!(phase_chunk, times_mat, reshape(fieldmap_chunk, 1, :))

        mul!(phase_chunk, transpose(kspha_err), transpose(bf_chunk), one(T), one(T))

        @. E_chunk = cis(T(2π) * phase_chunk)

        mul!(B_adj_chunk, adjoint(E_chunk), Q_cpu)
    end


    if rsvd_finalize === :svd

        F = svd!(B_adj_cpu; full=false)
    
        u_trunc = Q_cpu * F.V[:, 1:L_rank]
        s_trunc = F.S[1:L_rank]
        v_trunc = F.U[:, 1:L_rank]

        return u_trunc, s_trunc, v_trunc
    
    elseif rsvd_finalize === :gram
        v_scaled === nothing && throw(ArgumentError("v_scaled must be supplied when rsvd_finalize=:gram"))
    
        @assert size(v_scaled) == (nVox, L_rank)
    
        return finalize_rsvd_gram!(v_scaled, Q_cpu, B_adj_cpu, workspace, L_rank, allow_fallback=gram_allow_fallback)
    else
        throw(ArgumentError("Unsupported rsvd_finalize=$rsvd_finalize; " * "expected :svd or :gram"))
    end
end


function kernel_phase_to_encoding!(
    encoding,
    phase_highorder,
    times,
    fieldmap,
)
    iSam = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iVox = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if iSam <= size(encoding, 1) && iVox <= size(encoding, 2)
        @inbounds begin
            phase = phase_highorder[iSam, iVox] + times[iSam] * fieldmap[iVox]

            encoding[iSam, iVox] = cis(eltype(phase_highorder)(2π) * phase)
        end
    end

    return nothing
end



"""
phase_chunk: [nSam, chunk_size] Float32
E_chunk:     [nSam, chunk_size] ComplexF32
Ω:           [nVox, L_total]
W:           [nSam, L_total]
B_adj:       [nVox, L_total]
"""
function perform_rsvd(
    times         :: CuArray{T}, 
    fieldmap      :: CuArray{T}, 
    bf            :: CuArray{T, 2}, 
    kspha_err     :: CuArray{T, 2}, 
    nVox          :: Int, 
    nSam          :: Int, 
    L_rank        :: Int,
    chunk_size    :: Int,
    workspace     :: RSVDWorkspace;
    seed          :: Int = 0,
    p_oversample  :: Int = 5,
    rsvd_finalize :: Symbol = :svd,
    rsvd_backend  :: Symbol = :kernel,
    v_scaled              = nothing,
    gram_allow_fallback::Bool = true,
    verbose::Bool = false,
    ) where T <: AbstractFloat

    rsvd_backend in (:chunked, :kernel) || throw(ArgumentError(
        "Unsupported rsvd_backend=$rsvd_backend; expected :chunked or :kernel",
    ))

    if verbose @info "Performing rSVD on GPU..." backend=rsvd_backend end

    L_total = L_rank + p_oversample
    @assert L_total <= min(nSam, nVox) "rSVD rank exceeds matrix dimensions"
    @assert chunk_size > 0 "chunk_size must be positive"
    chunk_size = min(chunk_size, nVox)

    phase_workspace = workspace.phase
    E_workspace     = workspace.encoding
    Ω_d             = workspace.omega
    W_d             = workspace.W
    B_adj_d         = workspace.B_adj

    CUDA.seed!(seed)
    randn!(Ω_d)

    kernel_threads = (32, 8)

    # First pass: W = E * Ω
    if rsvd_backend === :kernel
        run_kernel_rsvd_forward!(
            W_d, Ω_d, times, fieldmap, bf, kspha_err;
            threads=128,
        )
    else
        fill!(W_d, zero(Complex{T}))
        for vox_start = 1:chunk_size:nVox
            vox_stop = min(vox_start + chunk_size - 1, nVox)
            vox_range = vox_start:vox_stop
            nChunk = length(vox_range)

            phase_chunk = @view phase_workspace[:, 1:nChunk]
            E_chunk = @view E_workspace[:, 1:nChunk]

            fieldmap_chunk = @view fieldmap[vox_range]
            bf_chunk = @view bf[vox_range, :]
            Ω_chunk = @view Ω_d[vox_range, :]

            mul!(phase_chunk, transpose(kspha_err), transpose(bf_chunk), one(T), zero(T))
            kernel_blocks = (cld(nSam, kernel_threads[1]), cld(nChunk, kernel_threads[2]))
            @cuda threads=kernel_threads blocks=kernel_blocks kernel_phase_to_encoding!(E_chunk, phase_chunk, times, fieldmap_chunk)

            mul!(W_d, E_chunk, Ω_chunk, one(Complex{T}), one(Complex{T}))
        end
    end

    # W = Q * R
    qr_W = qr!(W_d)
    Q_d = CuArray(qr_W.Q)

    # Second pass: B_adj = E' * Q
    # B_adj_d = CUDA.zeros(Complex{T}, nVox, L_total)

    if rsvd_backend === :kernel
        run_kernel_rsvd_adjoint!(
            B_adj_d, Q_d, times, fieldmap, bf, kspha_err;
            threads=256,
        )
    else
        for vox_start = 1:chunk_size:nVox
            vox_stop = min(vox_start + chunk_size - 1, nVox)
            vox_range = vox_start:vox_stop
            nChunk = length(vox_range)

            phase_chunk = @view phase_workspace[:, 1:nChunk]
            E_chunk = @view E_workspace[:, 1:nChunk]

            fieldmap_chunk = @view fieldmap[vox_range]
            bf_chunk = @view bf[vox_range, :]
            B_adj_chunk = @view B_adj_d[vox_range, :]

            mul!(phase_chunk, transpose(kspha_err), transpose(bf_chunk), one(T), zero(T))
            kernel_blocks = (cld(nSam, kernel_threads[1]), cld(nChunk, kernel_threads[2]))
            @cuda threads=kernel_threads blocks=kernel_blocks kernel_phase_to_encoding!(E_chunk, phase_chunk, times, fieldmap_chunk)

            mul!(B_adj_chunk, adjoint(E_chunk), Q_d)
        end
    end

    if rsvd_finalize === :svd

        F = svd!(B_adj_d; full=false)
    
        u_trunc = Q_d * F.V[:, 1:L_rank]
        s_trunc = F.S[1:L_rank]
        v_trunc = F.U[:, 1:L_rank]
    
        return u_trunc, s_trunc, v_trunc
    
    elseif rsvd_finalize === :gram
        v_scaled === nothing && throw(ArgumentError("v_scaled must be supplied when rsvd_finalize=:gram"))
    
        @assert size(v_scaled) == (nVox, L_rank)
    
        return finalize_rsvd_gram!(v_scaled, Q_d, B_adj_d, workspace, L_rank, allow_fallback=gram_allow_fallback)
    else
        throw(ArgumentError("Unsupported rsvd_finalize=$rsvd_finalize; " * "expected :svd or :gram"))
    end
end
