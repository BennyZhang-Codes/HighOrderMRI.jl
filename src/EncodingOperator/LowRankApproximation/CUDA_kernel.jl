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
    @assert L <= 32 "Fused rSVD kernel currently supports L_total <= 32"

    blocks  = nSam
    nWarp   = threads ÷ 32

    M_pad = M + (M % 2)
    shmem_bytes = M_pad * sizeof(T) + nWarp * L * sizeof(Complex{T})

    @cuda threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_rsvd_forward!(
            W, omega, times, fieldmap, bf, kspha, Int32(nSam), Int32(nVox),
            Val(M), Val(L), Val(kspha_transposed))

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

    @assert L <= 32 "Fused rSVD kernel currently supports L_total <= 32"

    @assert 32 <= threads <= 1024
    @assert threads % 32 == 0 "threads must be a multiple of warp size"

    warps_per_block = threads ÷ 32

    blocks = cld(nVox, warps_per_block)

    shmem_bytes = warps_per_block * M * sizeof(T)

    @cuda threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_rsvd_adjoint!(
            B_adj, Q, times, fieldmap, bf, kspha, Int32(nSam), Int32(nVox),
            Val(M), Val(L), Val(kspha_transposed))

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
