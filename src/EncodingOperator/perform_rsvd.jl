using CUDA
using LinearAlgebra
using NFFT
using Random

# ==============================================================================
# 1. CPU Backend
# ==============================================================================
function perform_rsvd(
    times     :: Array{T}, 
    fieldmap  :: Array{T}, 
    bf        :: Array{T, 2}, 
    kspha_err :: Array{T, 2}, 
    nVox      :: Int, 
    nSam      :: Int, 
    L_rank    :: Int;
    seed      :: Int = 0
    ) where T <: AbstractFloat
    @info "Performing rSVD on CPU (Multi-threaded)..."
    p_oversample = 5
    L_total = L_rank + p_oversample
    rng = Random.Xoshiro(seed)
    Ω_cpu = randn(rng, Complex{T}, nVox, L_total) 

    Y_cpu = zeros(Complex{T}, nSam, L_total)
    block_size = 5000 
    for i in 1:block_size:nVox
        idx = i:min(i+block_size-1, nVox)
        Phase_block = fieldmap[idx] * times' .+ bf[idx, :] * kspha_err
        H_block = exp.(1im .* T(2π) .* Phase_block)
        Y_cpu .+= transpose(H_block) * Ω_cpu[idx, :]
    end

    Q_cpu, _ = qr(Y_cpu)
    Q_cpu = Matrix(Q_cpu)

    B_adj_cpu = zeros(Complex{T}, nVox, L_total)
    for i in 1:block_size:nSam
        idx = i:min(i+block_size-1, nSam)
        Phase_t_block = fieldmap * times[idx]' .+ bf * kspha_err[:, idx]
        H_t_block = exp.(-1im .* T(2π) .* Phase_t_block)
        B_adj_cpu .+= H_t_block * Q_cpu[idx, :]
    end

    F = svd(B_adj_cpu)
    u_trunc  = Q_cpu * F.V[:, 1:L_rank]
    s_trunc  = F.S[1:L_rank]
    v_trunc  = F.U[:, 1:L_rank]

    return u_trunc, s_trunc, v_trunc
end

# ==============================================================================
# 2. GPU Backend: Shared Memory Kernels + Covariance Trick
# ==============================================================================
# ---------------------------------------------------------
# Kernel 1: Y = H * Ω 
# ---------------------------------------------------------
function kernel_rsvd_forward_opt!(Y, times, fieldmap, bf, kspha_err, Omega, nVox, nSam, nTerm, L)
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    t = threadIdx().x

    shmem = CuDynamicSharedArray(eltype(Y), (blockDim().x, L))

    if s <= nSam
        for l = 1:L
            shmem[t, l] = zero(eltype(Y))
        end

        for v = 1:nVox
            phase = fieldmap[v] * times[s]
            for term = 1:nTerm
                phase += bf[v, term] * kspha_err[term, s]
            end
            
            h_val = cis( eltype(fieldmap)(2π) * phase ) 

            for l = 1:L
                shmem[t, l] += h_val * Omega[v, l]
            end
        end

        for l = 1:L
            Y[s, l] = shmem[t, l]
        end
    end
    return nothing
end

# ---------------------------------------------------------
# Kernel 2: B_adj = H' * Q
# ---------------------------------------------------------
function kernel_rsvd_adjoint_opt!(B_adj, times, fieldmap, bf, kspha_err, Q, nVox, nSam, nTerm, L)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    t = threadIdx().x

    shmem = CuDynamicSharedArray(eltype(B_adj), (blockDim().x, L))

    if v <= nVox
        for l = 1:L
            shmem[t, l] = zero(eltype(B_adj))
        end

        for s = 1:nSam
            phase = fieldmap[v] * times[s]
            for term = 1:nTerm
                phase += bf[v, term] * kspha_err[term, s]
            end
            
            h_val_adj = cis( -eltype(fieldmap)(2π) * phase ) 

            for l = 1:L
                shmem[t, l] += h_val_adj * Q[s, l]
            end
        end

        for l = 1:L
            B_adj[v, l] = shmem[t, l]
        end
    end
    return nothing
end

function perform_rsvd(
    times     :: CuArray{T}, 
    fieldmap  :: CuArray{T}, 
    bf        :: CuArray{T, 2}, 
    kspha_err :: CuArray{T, 2}, 
    nVox      :: Int, 
    nSam      :: Int, 
    L_rank    :: Int;
    seed      :: Int = 0
    ) where T <: AbstractFloat
    @info "Performing rSVD on GPU (Fused Kernels)..."
    p_oversample = 5
    L_total = L_rank + p_oversample
    nTerm_err = size(kspha_err, 1)
    CUDA.seed!(seed)
    Ω_d = CUDA.randn(Complex{T}, nVox, L_total) 
    
    Y_d = CUDA.zeros(Complex{T}, nSam, L_total)
    threads_Y = 256
    blocks_Y = cld(nSam, threads_Y)
    
    shmem_size_Y = threads_Y * L_total * sizeof(Complex{T})
    
    @cuda threads=threads_Y blocks=blocks_Y shmem=shmem_size_Y kernel_rsvd_forward_opt!(
        Y_d, times, fieldmap, bf, kspha_err, Ω_d, nVox, nSam, nTerm_err, L_total
    )
    
    Q_cpu, _ = qr(Array(Y_d))
    Q_d = CuArray(Matrix(Q_cpu))
    
    B_adj_d = CUDA.zeros(Complex{T}, nVox, L_total)
    threads_B = 256
    blocks_B = cld(nVox, threads_B)
    
    shmem_size_B = threads_B * L_total * sizeof(Complex{T})
    
    @cuda threads=threads_B blocks=blocks_B shmem=shmem_size_B kernel_rsvd_adjoint_opt!(
        B_adj_d, times, fieldmap, bf, kspha_err, Q_d, nVox, nSam, nTerm_err, L_total
    )
    
    B_adj_cpu = Array(B_adj_d)
    F = svd(B_adj_cpu)
    
    u_trunc  = Matrix(Q_cpu) * F.V[:, 1:L_rank]
    s_trunc  = F.S[1:L_rank]
    v_trunc  = F.U[:, 1:L_rank]
    
    CUDA.unsafe_free!(Ω_d); CUDA.unsafe_free!(Y_d); CUDA.unsafe_free!(Q_d); CUDA.unsafe_free!(B_adj_d);

    return u_trunc, s_trunc, v_trunc
end




"""
phase:    [nSam, chunk_size] Float32
encoding: [nSam, chunk_size] ComplexF32
omega:    [nVox, L_total] ComplexF32
W:        [nSam, L_total] ComplexF32
B_adj:    [nVox, L_total] ComplexF32
"""
mutable struct RSVDWorkspace{RM<:AbstractMatrix, CM<:AbstractMatrix}
    phase    :: RM  # [nSam, chunk_size]
    encoding :: CM  # [nSam, chunk_size]
    omega    :: CM  # [nVox, L_total]
    W        :: CM  # [nSam, L_total]
    B_adj    :: CM  # [nVox, L_total]
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
    )
end


function perform_rsvd(
    times        :: Array{T}, 
    fieldmap     :: Array{T}, 
    bf           :: Array{T, 2}, 
    kspha_err    :: Array{T, 2}, 
    nVox         :: Int, 
    nSam         :: Int, 
    L_rank       :: Int,
    chunk_size   :: Int,
    workspace    :: RSVDWorkspace;
    seed         :: Int = 0,
    p_oversample :: Int = 5,
    ) where T <: AbstractFloat

    @info "Performing chunked rSVD on CPU..."

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

    # rng = Random.Xoshiro(seed)
    # Ω_cpu = randn(rng, Complex{T}, nVox, L_total)
    # W_cpu = zeros(Complex{T}, nSam, L_total)

    # phase_workspace = zeros(T, nSam, chunk_size)
    # E_workspace = zeros(Complex{T}, nSam, chunk_size)

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

    F = svd!(B_adj_cpu)

    u_trunc = Q_cpu * F.V[:, 1:L_rank]
    s_trunc = F.S[1:L_rank]
    v_trunc = F.U[:, 1:L_rank]

    return u_trunc, s_trunc, v_trunc
end


function kernel_phase_to_encoding!(E, phase)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= length(phase)
        @inbounds E[idx] = cis(eltype(phase)(2π) * phase[idx])
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
    times        :: CuArray{T}, 
    fieldmap     :: CuArray{T}, 
    bf           :: CuArray{T, 2}, 
    kspha_err    :: CuArray{T, 2}, 
    nVox         :: Int, 
    nSam         :: Int, 
    L_rank       :: Int,
    chunk_size   :: Int,
    workspace    :: RSVDWorkspace;
    seed         :: Int = 0,
    p_oversample :: Int = 5,
    ) where T <: AbstractFloat

    @info "Performing chunked rSVD on GPU..."

    L_total = L_rank + p_oversample
    @assert L_total <= min(nSam, nVox) "rSVD rank exceeds matrix dimensions"
    @assert chunk_size > 0 "chunk_size must be positive"
    chunk_size = min(chunk_size, nVox)

    phase_workspace = workspace.phase
    E_workspace     = workspace.encoding
    Ω_d             = workspace.omega
    W_d             = workspace.W
    B_adj_d         = workspace.B_adj

    fill!(W_d, 0)

    CUDA.seed!(seed)
    randn!(Ω_d)

    # CUDA.seed!(seed)

    # Ω_d = CUDA.randn(Complex{T}, nVox, L_total) 
    # W_d = CUDA.zeros(Complex{T}, nSam, L_total)

    # phase_workspace = CUDA.zeros(T, nSam, chunk_size)
    # E_workspace = CUDA.zeros(Complex{T}, nSam, chunk_size)

    times_mat = reshape(times, :, 1)
    threads = 256

    # First pass: W = E * Ω
    for vox_start = 1:chunk_size:nVox
        vox_stop = min(vox_start + chunk_size - 1, nVox)
        vox_range = vox_start:vox_stop
        nChunk = length(vox_range)

        phase_chunk = @view phase_workspace[:, 1:nChunk]
        E_chunk = @view E_workspace[:, 1:nChunk]

        fieldmap_chunk = @view fieldmap[vox_range]
        bf_chunk = @view bf[vox_range, :]
        Ω_chunk = @view Ω_d[vox_range, :]

        # phase_chunk = times * fieldmap_chunk' + kspha_err' * bf_chunk'
        mul!(phase_chunk, times_mat, reshape(fieldmap_chunk, 1, :))

        mul!(phase_chunk, transpose(kspha_err), transpose(bf_chunk), one(T), one(T))

        blocks = cld(length(phase_chunk), threads)
        @cuda threads=threads blocks=blocks kernel_phase_to_encoding!(E_chunk, phase_chunk)

        mul!(W_d, E_chunk, Ω_chunk, one(Complex{T}), one(Complex{T}))
    end

    # W = Q * R
    qr_W = qr!(W_d)
    Q_d = CuArray(qr_W.Q)

    # Second pass: B_adj = E' * Q
    # B_adj_d = CUDA.zeros(Complex{T}, nVox, L_total)

    for vox_start = 1:chunk_size:nVox
        vox_stop = min(vox_start + chunk_size - 1, nVox)
        vox_range = vox_start:vox_stop
        nChunk = length(vox_range)

        phase_chunk = @view phase_workspace[:, 1:nChunk]
        E_chunk = @view E_workspace[:, 1:nChunk]

        fieldmap_chunk = @view fieldmap[vox_range]
        bf_chunk = @view bf[vox_range, :]
        B_adj_chunk = @view B_adj_d[vox_range, :]

        mul!(phase_chunk, times_mat, reshape(fieldmap_chunk, 1, :))

        mul!(phase_chunk, transpose(kspha_err), transpose(bf_chunk), one(T), one(T))

        blocks = cld(length(phase_chunk), threads)
        @cuda threads=threads blocks=blocks kernel_phase_to_encoding!(E_chunk, phase_chunk)

        mul!(B_adj_chunk, adjoint(E_chunk), Q_d)
    end

    F = svd!(B_adj_d; full=false)

    u_trunc = Q_d * F.V[:, 1:L_rank]
    s_trunc = F.S[1:L_rank]
    v_trunc = F.U[:, 1:L_rank]

    return u_trunc, s_trunc, v_trunc
end
