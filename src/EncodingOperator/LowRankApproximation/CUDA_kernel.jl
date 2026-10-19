@inline function shuffle_down_add_tuple(values::NTuple{L,T}, offset::Int32) where {L,T}

    return ntuple(Val(L)) do i
        @inbounds values[i] + CUDA.shfl_down_sync( CUDA.FULL_MASK, values[i], offset, Int32(32))
    end
end


@inline function reduce_warp_tuple(values::NTuple{L,T}) where {L,T}

    offset = Int32(16)

    while offset > 0
        values = shuffle_down_add_tuple(values, offset)
        offset >>= 1
    end
    return values
end

@inline function accumulate_phase_tuple(
    acc        :: NTuple{L,Complex{T}},
    phase_real :: T,
    phase_imag :: T,
    matrix     :: CuDeviceMatrix{Complex{T}},
    irow       :: I,
) where {L,T<:AbstractFloat,I<:Integer}

    neg_imag = -phase_imag

    return ntuple(Val(L)) do irank
        @inbounds begin
            acc_real, acc_imag = reim(acc[irank])
            value_real, value_imag = reim(matrix[irow, irank])

            out_real = muladd(phase_real, value_real, muladd(neg_imag, value_imag, acc_real))
            out_imag = muladd(phase_imag, value_real, muladd(phase_real, value_imag, acc_imag))
            Complex{T}(out_real, out_imag)
        end
    end
end


@inline phase_sincos(theta::T) where {T<:AbstractFloat} = sincos(theta)


# Each fused kernel keeps one accumulator per sketch column in registers.
# Compiling a single launch with a much larger `Val(L)` causes excessive
# register pressure and very long compilation.  Wider sketches are therefore
# split into independent column batches while retaining the same E * Ω and
# E' * Q mathematics.
const RSVD_KERNEL_RANK_BATCH = 16


@inline function load_kspha(
    kspha::CuDeviceMatrix{T},
    iterm::I,
    isam::J,
    ::Val{false},
) where {T,I<:Integer,J<:Integer}
    return @inbounds kspha[iterm, isam]
end


@inline function load_kspha(
    kspha::CuDeviceMatrix{T},
    iterm::I,
    isam::J,
    ::Val{true},
) where {T,I<:Integer,J<:Integer}
    return @inbounds kspha[isam, iterm]
end


function run_kernel_rsvd_forward!(
    W        :: CuMatrix{Complex{T}},
    omega    :: CuMatrix{Complex{T}},
    times    :: CuVector{T},
    fieldmap :: CuVector{T},
    bf       :: CuMatrix{T},
    kspha    :: CuMatrix{T};
    threads  :: Int = 128,
    kspha_transposed::Bool = false,
) where {T<:AbstractFloat}

    nSam = size(W, 1)
    L    = size(W, 2)
    nVox = size(omega, 1)
    M    = kspha_transposed ? size(kspha, 2) : size(kspha, 1)

    @assert size(omega) == (nVox, L)
    @assert size(times) == (nSam,)
    @assert size(fieldmap) == (nVox,)
    @assert size(bf) == (nVox, M)
    expected_kspha_size = kspha_transposed ? (nSam, M) : (M, nSam)
    @assert size(kspha) == expected_kspha_size
    blocks  = nSam
    nWarp   = threads ÷ 32

    M_pad = M + (M % 2)

    for first_rank = 1:RSVD_KERNEL_RANK_BATCH:L
        last_rank = min(first_rank + RSVD_KERNEL_RANK_BATCH - 1, L)
        ranks = first_rank:last_rank
        batch_rank = length(ranks)
        W_batch = @view W[:, ranks]
        omega_batch = @view omega[:, ranks]
        shmem_bytes = M_pad * sizeof(T) + nWarp * batch_rank * sizeof(Complex{T})

        @cuda always_inline=true threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_rsvd_forward!(
                W_batch, omega_batch, times, fieldmap, bf, kspha, Int32(nSam), Int32(nVox),
                Val(M), Val(batch_rank), Val(kspha_transposed))
    end

    return W
end


function run_kernel_rsvd_adjoint!(
    B_adj    :: CuMatrix{Complex{T}},
    Q        :: CuMatrix{Complex{T}},
    times    :: CuVector{T},
    fieldmap :: CuVector{T},
    bf       :: CuMatrix{T},
    kspha    :: CuMatrix{T};
    threads  :: Int = 128,
    kspha_transposed::Bool = false,
    lanes_per_voxel::Int = 32,
    voxels_per_group::Int = 1,
) where {T<:AbstractFloat}

    nSam = size(Q, 1)
    L    = size(Q, 2)
    nVox = size(B_adj, 1)
    M    = kspha_transposed ? size(kspha, 2) : size(kspha, 1)

    @assert size(B_adj) == (nVox, L)
    @assert size(times) == (nSam,)
    @assert size(fieldmap) == (nVox,)
    @assert size(bf) == (nVox, M)
    expected_kspha_size = kspha_transposed ? (nSam, M) : (M, nSam)
    @assert size(kspha) == expected_kspha_size

    @assert 32 <= threads <= 1024
    @assert threads % 32 == 0 "threads must be a multiple of warp size"

    lanes_per_voxel in (1, 2, 4, 8, 16, 32) || throw(ArgumentError(
        "lanes_per_voxel must be a power of two between 1 and 32",
    ))
    voxels_per_group in (1, 2) || throw(ArgumentError(
        "voxels_per_group must be 1 or 2",
    ))
    voxels_per_group == 1 || lanes_per_voxel == 2 || throw(ArgumentError(
        "Paired voxels require lanes_per_voxel=2",
    ))
    groups_per_block = threads ÷ lanes_per_voxel

    blocks = cld(nVox, groups_per_block * voxels_per_group)

    M_pad = lanes_per_voxel == 32 ? M : (iseven(M) ? M + 1 : M)
    shmem_bytes = groups_per_block * M_pad * sizeof(T)

    for first_rank = 1:RSVD_KERNEL_RANK_BATCH:L
        last_rank = min(first_rank + RSVD_KERNEL_RANK_BATCH - 1, L)
        ranks = first_rank:last_rank
        batch_rank = length(ranks)
        B_adj_batch = @view B_adj[:, ranks]
        Q_batch = @view Q[:, ranks]

        if voxels_per_group == 2
            @cuda always_inline=true threads=threads blocks=blocks CUDA_kernel_rsvd_adjoint_paired!(
                B_adj_batch, Q_batch, times, fieldmap, bf, kspha, Int32(nSam), Int32(nVox),
                Val(M), Val(batch_rank), Val(kspha_transposed))
        elseif lanes_per_voxel == 32
            @cuda always_inline=true threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_rsvd_adjoint!(
                B_adj_batch, Q_batch, times, fieldmap, bf, kspha, Int32(nSam), Int32(nVox),
                Val(M), Val(batch_rank), Val(kspha_transposed))
        else
            @cuda always_inline=true threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_rsvd_adjoint_grouped!(
                B_adj_batch, Q_batch, times, fieldmap, bf, kspha, Int32(nSam), Int32(nVox),
                Val(M), Val(batch_rank), Val(kspha_transposed), Val(lanes_per_voxel))
        end
    end

    return B_adj
end

function CUDA_kernel_rsvd_forward!(
    W        :: CuDeviceMatrix{Complex{T}},
    omega    :: CuDeviceMatrix{Complex{T}},
    times    :: CuDeviceVector{T},
    fieldmap :: CuDeviceVector{T},
    bf       :: CuDeviceMatrix{T},
    kspha    :: CuDeviceMatrix{T},
    nSam     :: Int32,
    nVox     :: Int32,
    ::Val{M},
    ::Val{L},
    kspha_transposed::Val{K},
) where {T<:AbstractFloat,M,L,K}

    isam = blockIdx().x

    if isam > nSam
        return nothing
    end

    tid    = threadIdx().x
    stride = blockDim().x

    lane  = (tid - 1) % Int32(32)
    iwarp = (tid - 1) ÷ Int32(32)
    nWarp = blockDim().x ÷ Int32(32)

    M_pad = M + (M % 2)

    s_kspha = CuDynamicSharedArray(T, M)
    s_partial = CuDynamicSharedArray(Complex{T}, nWarp * L, M_pad * sizeof(T))

    iterm = tid
    while iterm <= M
        @inbounds s_kspha[iterm] = load_kspha(kspha, iterm, isam, kspha_transposed)
        iterm += stride
    end

    sync_threads()

    time = @inbounds times[isam]
    acc = ntuple(_ -> zero(Complex{T}), Val(L))

    ivox = tid
    while ivox <= nVox
        phase = @inbounds time * fieldmap[ivox]

        @inbounds for iterm = 1:M
            phase = muladd(bf[ivox, iterm], s_kspha[iterm], phase)
        end

        phase_sin, phase_cos = phase_sincos(T(2π) * phase)
        acc = accumulate_phase_tuple(acc, phase_cos, phase_sin, omega, ivox)

        ivox += stride
    end

    acc = reduce_warp_tuple(acc)

    if lane == 0
        offset = iwarp * L
        @inbounds for irank = 1:L
            s_partial[offset + irank] = acc[irank]
        end
    end

    sync_threads()

    if tid <= L
        total = zero(Complex{T})

        @inbounds for iwarp = 0:(nWarp - 1)
            total += s_partial[iwarp * L + tid]
        end

        @inbounds W[isam, tid] = total
    end

    return nothing
end


function CUDA_kernel_rsvd_adjoint!(
    B_adj    :: CuDeviceMatrix{Complex{T}},
    Q        :: CuDeviceMatrix{Complex{T}},
    times    :: CuDeviceVector{T},
    fieldmap :: CuDeviceVector{T},
    bf       :: CuDeviceMatrix{T},
    kspha    :: CuDeviceMatrix{T},
    nSam     :: Int32,
    nVox     :: Int32,
    ::Val{M},
    ::Val{L},
    kspha_transposed::Val{K},
) where {T<:AbstractFloat,M,L,K}

    tid = threadIdx().x

    lane  = (tid - Int32(1)) % Int32(32)
    iwarp = (tid - Int32(1)) ÷ Int32(32)
    nWarp = blockDim().x ÷ Int32(32)

    ivox = (blockIdx().x - Int32(1)) * nWarp + iwarp + Int32(1)
    valid_voxel = ivox <= nVox

    s_bf = CuDynamicSharedArray(T, nWarp * M)

    bf_offset = iwarp * M

    if valid_voxel
        iterm = lane + Int32(1)

        while iterm <= M
            @inbounds s_bf[bf_offset + iterm] = bf[ivox, iterm]

            iterm += Int32(32)
        end
    end

    sync_threads()

    if valid_voxel
        b0 = @inbounds fieldmap[ivox]

        acc = ntuple(_ -> zero(Complex{T}), Val(L))

        # lane+1, lane+33, lane+65, ...
        isam = lane + Int32(1)

        while isam <= nSam
            phase = @inbounds times[isam] * b0

            @inbounds for iterm = 1:M
                phase = muladd(
                    s_bf[bf_offset + iterm],
                    load_kspha(kspha, iterm, isam, kspha_transposed),
                    phase,
                )
            end

            phase_sin, phase_cos = phase_sincos(T(2π) * phase)

            # conj(E) = cos(phase) - i*sin(phase)
            acc = accumulate_phase_tuple(acc, phase_cos, -phase_sin, Q, isam)

            isam += Int32(32)
        end

        acc = reduce_warp_tuple(acc)

        if lane == 0
            @inbounds for irank = 1:L
                B_adj[ivox, irank] = acc[irank]
            end
        end
    end

    return nothing
end


function run_kernel_rsvd_forward_tiled!(
    W::CuMatrix{Complex{T}},
    partials::CuArray{Complex{T},3},
    omega::CuMatrix{Complex{T}},
    times::CuVector{T},
    fieldmap::CuVector{T},
    bf::CuMatrix{T},
    kspha::CuMatrix{T};
    threads::Int=128,
    kspha_transposed::Bool=false,
) where {T<:AbstractFloat}
    nSam, L = size(W)
    nVox = size(omega, 1)
    M = kspha_transposed ? size(kspha, 2) : size(kspha, 1)
    nParts = size(partials, 3)
    @assert nParts > 0
    @assert size(partials) == (nSam, L, nParts)
    @assert size(omega) == (nVox, L)
    @assert length(times) == nSam && length(fieldmap) == nVox
    @assert size(bf) == (nVox, M)
    @assert size(kspha) == (kspha_transposed ? (nSam, M) : (M, nSam))
    @assert 32 <= threads <= 1024 && threads % 32 == 0
    # Align starts of consecutive tiles to coalesced voxel accesses. Any
    # trailing empty tile still writes zero, so all partials are overwritten.
    chunk = cld(cld(nVox, nParts), threads) * threads
    M_pad = M + M % 2
    for first_rank in 1:RSVD_KERNEL_RANK_BATCH:L
        ranks = first_rank:min(first_rank + RSVD_KERNEL_RANK_BATCH - 1, L)
        batch_rank = length(ranks)
        W_batch = @view W[:, ranks]
        omega_batch = @view omega[:, ranks]
        parts_batch = @view partials[:, ranks, :]
        shmem_bytes = M_pad * sizeof(T) + (threads ÷ 32) * batch_rank * sizeof(Complex{T})
        @cuda always_inline=true threads=threads blocks=(nSam, nParts) shmem=shmem_bytes CUDA_kernel_rsvd_forward_tiled!(
            parts_batch, omega_batch, times, fieldmap, bf, kspha,
            Int32(nSam), Int32(nVox), Int32(chunk),
            Val(M), Val(batch_rank), Val(kspha_transposed))
        @cuda threads=256 blocks=cld(nSam * batch_rank, 256) kernel_sum_rsvd_partials!(
            W_batch, parts_batch, Int32(nSam), Int32(batch_rank))
    end
    return W
end


function CUDA_kernel_rsvd_forward_tiled!(
    partials::AbstractArray{Complex{T},3}, omega::CuDeviceMatrix{Complex{T}},
    times::CuDeviceVector{T}, fieldmap::CuDeviceVector{T},
    bf::CuDeviceMatrix{T}, kspha::CuDeviceMatrix{T},
    nSam::Int32, nVox::Int32, chunk::Int32,
    ::Val{M}, ::Val{L}, transposed::Val{K},
) where {T<:AbstractFloat,M,L,K}
    sam, part = blockIdx().x, blockIdx().y
    tid, stride = threadIdx().x, blockDim().x
    lane = (tid - Int32(1)) % Int32(32)
    warp = (tid - Int32(1)) ÷ Int32(32)
    nWarp = stride ÷ Int32(32)
    M_pad = M + M % 2
    coefficients = CuDynamicSharedArray(T, M)
    warp_sums = CuDynamicSharedArray(Complex{T}, nWarp * L, M_pad * sizeof(T))
    term = tid
    while term <= M
        @inbounds coefficients[term] = load_kspha(kspha, term, sam, transposed)
        term += stride
    end
    sync_threads()
    time = @inbounds times[sam]
    acc = ntuple(_ -> zero(Complex{T}), Val(L))
    v = (part - Int32(1)) * chunk + tid
    last_voxel = min(part * chunk, nVox)
    while v <= last_voxel
        phase = @inbounds time * fieldmap[v]
        @inbounds for term in 1:M
            phase = muladd(bf[v, term], coefficients[term], phase)
        end
        s, c = phase_sincos(T(2π) * phase)
        acc = accumulate_phase_tuple(acc, c, s, omega, v)
        v += stride
    end
    acc = reduce_warp_tuple(acc)
    if lane == 0
        @inbounds for r in 1:L
            warp_sums[warp * L + r] = acc[r]
        end
    end
    sync_threads()
    if tid <= L
        value = zero(Complex{T})
        @inbounds for w in 0:(nWarp - 1)
            value += warp_sums[w * L + tid]
        end
        @inbounds partials[sam, tid, part] = value
    end
    return nothing
end


function kernel_sum_rsvd_partials!(W, partials, nSam::Int32, L::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= nSam * L
        sam = (i - Int32(1)) % nSam + Int32(1)
        rank = (i - Int32(1)) ÷ nSam + Int32(1)
        value = zero(eltype(W))
        @inbounds for part in axes(partials, 3)
            value += partials[sam, rank, part]
        end
        @inbounds W[sam, rank] = value
    end
    return nothing
end


@inline function shuffle_down_add_group(values::NTuple{L,T}, offset::Int32, ::Val{G}) where {L,T,G}
    return ntuple(Val(L)) do r
        @inbounds values[r] + CUDA.shfl_down_sync(CUDA.FULL_MASK, values[r], offset, Int32(G))
    end
end


@inline function reduce_group_tuple(values::NTuple{L,T}, ::Val{G}) where {L,T,G}
    offset = Int32(G ÷ 2)
    while offset > 0
        values = shuffle_down_add_group(values, offset, Val(G))
        offset >>= 1
    end
    return values
end


function CUDA_kernel_rsvd_adjoint_grouped!(
    B_adj::CuDeviceMatrix{Complex{T}}, Q::CuDeviceMatrix{Complex{T}},
    times::CuDeviceVector{T}, fieldmap::CuDeviceVector{T},
    bf::CuDeviceMatrix{T}, kspha::CuDeviceMatrix{T},
    nSam::Int32, nVox::Int32, ::Val{M}, ::Val{L}, transposed::Val{K}, ::Val{G},
) where {T<:AbstractFloat,M,L,K,G}
    tid = threadIdx().x
    lane = (tid - Int32(1)) % Int32(G)
    group = (tid - Int32(1)) ÷ Int32(G)
    nGroups = blockDim().x ÷ Int32(G)
    v = (blockIdx().x - Int32(1)) * nGroups + group + Int32(1)
    M_pad = iseven(M) ? M + 1 : M
    coefficients = CuDynamicSharedArray(T, nGroups * M_pad)
    offset = group * M_pad
    valid = v <= nVox
    if valid
        term = lane + Int32(1)
        while term <= M
            @inbounds coefficients[offset + term] = bf[v, term]
            term += Int32(G)
        end
    end
    sync_threads()
    acc = ntuple(_ -> zero(Complex{T}), Val(L))
    if valid
        b0 = @inbounds fieldmap[v]
        sam = lane + Int32(1)
        while sam <= nSam
            phase = @inbounds times[sam] * b0
            @inbounds for term in 1:M
                phase = muladd(coefficients[offset + term], load_kspha(kspha, term, sam, transposed), phase)
            end
            s, c = phase_sincos(T(2π) * phase)
            acc = accumulate_phase_tuple(acc, c, -s, Q, sam)
            sam += Int32(G)
        end
    end
    # All lanes participate, including invalid voxel groups in a tail warp.
    acc = reduce_group_tuple(acc, Val(G))
    if valid && lane == 0
        @inbounds for r in 1:L
            B_adj[v, r] = acc[r]
        end
    end
    return nothing
end


# Reuse temporal Q/trajectory loads for two independent voxels. Keeping the
# spatial coefficients in registers avoids a shared-memory load per phase term.
@generated function rsvd_phase_pair(
    phase0::T, phase1::T,
    coeffs0::NTuple{M,T}, coeffs1::NTuple{M,T},
    kspha, sample, transposed,
) where {M,T}
    updates = Expr[]
    for term in 1:M
        push!(updates, :(coefficient = load_kspha(kspha, $term, sample, transposed)))
        push!(updates, :(phase0 = muladd(coeffs0[$term], coefficient, phase0)))
        push!(updates, :(phase1 = muladd(coeffs1[$term], coefficient, phase1)))
    end
    return quote
        @inbounds begin
            $(updates...)
        end
        return phase0, phase1
    end
end

@inline function accumulate_rsvd_pair(
    acc0::NTuple{L,Complex{T}}, acc1::NTuple{L,Complex{T}},
    c0, s0, c1, s1, Q, sample,
) where {L,T}
    pairs = ntuple(Val(L)) do rank
        @inbounds begin
            re, im = reim(Q[sample, rank])
            z0, z1 = acc0[rank], acc1[rank]
        end
        value0 = Complex{T}(
            muladd(c0, re, muladd(-s0, im, real(z0))),
            muladd(s0, re, muladd(c0, im, imag(z0))),
        )
        value1 = Complex{T}(
            muladd(c1, re, muladd(-s1, im, real(z1))),
            muladd(s1, re, muladd(c1, im, imag(z1))),
        )
        return value0, value1
    end
    return ntuple(rank -> pairs[rank][1], Val(L)), ntuple(rank -> pairs[rank][2], Val(L))
end

function CUDA_kernel_rsvd_adjoint_paired!(
    B_adj, Q, times, fieldmap, bf, kspha,
    nSam::Int32, nVox::Int32,
    ::Val{M}, ::Val{L}, transposed::Val{K},
) where {M,L,K}
    T = eltype(times)
    tid = threadIdx().x
    lane = (tid - Int32(1)) % Int32(2)
    group = (tid - Int32(1)) ÷ Int32(2)
    nGroups = blockDim().x ÷ Int32(2)
    voxel0 = (blockIdx().x - Int32(1)) * (Int32(2) * nGroups) + group + Int32(1)
    voxel1 = voxel0 + nGroups
    coeffs0 = ntuple(term -> voxel0 <= nVox ? (@inbounds bf[voxel0, term]) : zero(T), Val(M))
    coeffs1 = ntuple(term -> voxel1 <= nVox ? (@inbounds bf[voxel1, term]) : zero(T), Val(M))
    b0 = voxel0 <= nVox ? (@inbounds fieldmap[voxel0]) : zero(T)
    b1 = voxel1 <= nVox ? (@inbounds fieldmap[voxel1]) : zero(T)
    acc0 = ntuple(_ -> zero(Complex{T}), Val(L))
    acc1 = acc0

    if voxel0 <= nVox
        sample = lane + Int32(1)
        while sample <= nSam
            time = @inbounds times[sample]
            phase0, phase1 = rsvd_phase_pair(time * b0, time * b1, coeffs0, coeffs1, kspha, sample, transposed)
            s0, c0 = phase_sincos(T(2π) * phase0)
            s1, c1 = phase_sincos(T(2π) * phase1)
            acc0, acc1 = accumulate_rsvd_pair(acc0, acc1, c0, -s0, c1, -s1, Q, sample)
            sample += Int32(2)
        end
    end

    # Include invalid tail groups in the warp shuffle. The two-lane sample
    # reduction order is identical to the existing one-voxel grouped kernel.
    acc0 = reduce_group_tuple(acc0, Val(2))
    acc1 = reduce_group_tuple(acc1, Val(2))
    if lane == 0
        if voxel0 <= nVox
            @inbounds for rank in 1:L
                B_adj[voxel0, rank] = acc0[rank]
            end
        end
        if voxel1 <= nVox
            @inbounds for rank in 1:L
                B_adj[voxel1, rank] = acc1[rank]
            end
        end
    end
    return nothing
end
