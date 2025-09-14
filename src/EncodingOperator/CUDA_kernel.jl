function run_kernel_prod!(    
    out      ::CuMatrix{Complex{T}},   # [nSam, nCha]
    x        ::CuVector{Complex{T}},   # [nVox]
    csm      ::CuMatrix{Complex{T}},   # [nVox, nCha]
    times    ::CuVector{T}         ,   # [nSam]
    fieldmap ::CuVector{T}         ,   # [nVox]
    bf       ::CuMatrix{T}         ,   # [nVox, nTerm]
    kspha    ::CuMatrix{T}         ,   # [nTerm, nSam]
    nSam::D, nCha::D, nTerm::D, nVox::D,
    ) where {T<:AbstractFloat, D<:Integer}
    threads = (256) 
    blocks = (cld(nSam, 1))
    shmem_bytes = nTerm * sizeof(T) + nCha * cld(threads[1],32) * sizeof(T) * 2
    @cuda threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_prod_HighOrderOp!(out, x, csm, times, fieldmap, bf, kspha, Int32(nSam), Int32(nCha), Int32(nTerm), Int32(nVox));
end

function run_kernel_ctprod!(
    out      ::CuMatrix{Complex{T}},   # [nVox, nCha]
    y        ::CuMatrix{Complex{T}},   # [nSam, nCha]  
    csm      ::CuMatrix{Complex{T}},   # [nVox, nCha]
    times    ::CuVector{T}         ,   # [nSam]
    fieldmap ::CuVector{T}         ,   # [nVox]
    bf       ::CuMatrix{T}         ,   # [nVox, nTerm]
    kspha    ::CuMatrix{T}         ,   # [nTerm, nSam]
    nSam::D, nCha::D, nTerm::D, nVox::D,
    ) where {T<:AbstractFloat, D<:Integer}
    threads = (512) 
    blocks = (cld(nVox, 1))
    shmem_bytes = nTerm * sizeof(T) + nCha * cld(threads[1],32) * sizeof(T) * 2
    @cuda threads=threads blocks=blocks shmem=shmem_bytes CUDA_kernel_ctprod_HighOrderOp!(out, y, csm, times, fieldmap, bf, kspha, Int32(nSam), Int32(nCha), Int32(nTerm), Int32(nVox));
end


@inline function reduce_warp(
    val1::T, val2::T, val3::T, val4::T, val5::T, val6::T, val7::T, val8::T) where {T}
    @inbounds for k=0:4
        offset = Int32(1 << k)
        val1 += CUDA.shfl_down_sync(0xffffffff, val1, offset, 32)
        val2 += CUDA.shfl_down_sync(0xffffffff, val2, offset, 32)
        val3 += CUDA.shfl_down_sync(0xffffffff, val3, offset, 32)
        val4 += CUDA.shfl_down_sync(0xffffffff, val4, offset, 32)
        val5 += CUDA.shfl_down_sync(0xffffffff, val5, offset, 32)
        val6 += CUDA.shfl_down_sync(0xffffffff, val6, offset, 32)
        val7 += CUDA.shfl_down_sync(0xffffffff, val7, offset, 32)
        val8 += CUDA.shfl_down_sync(0xffffffff, val8, offset, 32)
    end
    return val1, val2, val3, val4, val5, val6, val7, val8
end


function CUDA_kernel_prod_HighOrderOp!(
    out      ::CuDeviceMatrix{Complex{T}},   # [nSam, nCha]
    x        ::CuDeviceVector{Complex{T}},   # [nVox]
    csm      ::CuDeviceMatrix{Complex{T}},   # [nVox, nCha]
    times    ::CuDeviceVector{T}         ,   # [nSam]
    fieldmap ::CuDeviceVector{T}         ,   # [nVox]
    bf       ::CuDeviceMatrix{T}         ,   # [nVox, nTerm]
    kspha    ::CuDeviceMatrix{T}         ,   # [nTerm, nSam]
    nSam::Int32, nCha::Int32, nTerm::Int32, nVox::Int32,
) where {T<:AbstractFloat}
    @fastmath begin
        isam = blockIdx().x
        stride = blockDim().x
        tid = threadIdx().x
        iterm = tid
        icha = tid
        ivox = tid
        is = tid

        lane = (threadIdx().x - 1) % 32
        iwarp = (threadIdx().x - 1) ÷ 32
        nWarp = (blockDim().x + 31) ÷ 32

        if isam > nSam 
            return
        end
        x_r   = zero(T)
        x_i   = zero(T)
        tmp_r = zero(T)
        tmp_i = zero(T)
        csm_r = zero(T)
        csm_i = zero(T) 
        t     = zero(T)
        ϕ     = zero(T)
        ϕ_r   = zero(T)
        ϕ_i   = zero(T)
        TWOPI = 2f0 * π
        @inbounds t = times[isam]

        sig_r1 = zero(T)
        sig_r2 = zero(T)
        sig_r3 = zero(T)
        sig_r4 = zero(T)
        sig_r5 = zero(T)
        sig_r6 = zero(T)
        sig_r7 = zero(T)
        sig_r8 = zero(T)
        sig_r9 = zero(T)
        sig_r10 = zero(T)
        sig_r11 = zero(T)
        sig_r12 = zero(T)
        sig_r13 = zero(T)
        sig_r14 = zero(T)
        sig_r15 = zero(T)
        sig_r16 = zero(T)
        sig_r17 = zero(T)
        sig_r18 = zero(T)
        sig_r19 = zero(T)
        sig_r20 = zero(T)
        sig_r21 = zero(T)
        sig_r22 = zero(T)
        sig_r23 = zero(T)
        sig_r24 = zero(T)
        sig_r25 = zero(T)
        sig_r26 = zero(T)
        sig_r27 = zero(T)
        sig_r28 = zero(T)
        sig_r29 = zero(T)
        sig_r30 = zero(T)
        sig_r31 = zero(T)
        sig_r32 = zero(T)
        sig_i1 = zero(T)
        sig_i2 = zero(T)
        sig_i3 = zero(T)
        sig_i4 = zero(T)
        sig_i5 = zero(T)
        sig_i6 = zero(T)
        sig_i7 = zero(T)
        sig_i8 = zero(T)
        sig_i9 = zero(T)
        sig_i10 = zero(T)
        sig_i11 = zero(T)
        sig_i12 = zero(T)
        sig_i13 = zero(T)
        sig_i14 = zero(T)
        sig_i15 = zero(T)
        sig_i16 = zero(T)
        sig_i17 = zero(T)
        sig_i18 = zero(T)
        sig_i19 = zero(T)
        sig_i20 = zero(T)
        sig_i21 = zero(T)
        sig_i22 = zero(T)
        sig_i23 = zero(T)
        sig_i24 = zero(T)
        sig_i25 = zero(T)
        sig_i26 = zero(T)
        sig_i27 = zero(T)
        sig_i28 = zero(T)
        sig_i29 = zero(T)
        sig_i30 = zero(T)
        sig_i31 = zero(T)
        sig_i32 = zero(T)
        # sig_r = @MVector zeros(T, 32)
        # sig_i = @MVector zeros(T, 32)
        # sig_r = MVector(0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0)
        # sig_i = MVector(0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0)

        s_kspha = CuDynamicSharedArray(T, nTerm)       
        s_r     = CuDynamicSharedArray(T, nCha * nWarp, sizeof(T) * nTerm)
        s_i     = CuDynamicSharedArray(T, nCha * nWarp, sizeof(T) * (nCha * nWarp + nTerm))

        while is <= (nCha * nWarp)
            @inbounds s_r[is] = 0
            @inbounds s_i[is] = 0
            is += stride
        end

        # s_kspha = @cuDynamicSharedMem(T, nTerm)
        while iterm <= nTerm
            @inbounds s_kspha[iterm] = kspha[iterm, isam]
            iterm += stride
        end

        sync_threads()

        while ivox <= nVox
            ϕ = t * fieldmap[ivox]
            @inbounds for iterm = 1:nTerm
                ϕ += bf[ivox, iterm] * s_kspha[iterm]
            end
            # ϕ_r, ϕ_i = reim(cis(TWOPI * ϕ))
            ϕ_i, ϕ_r = sincos(TWOPI * ϕ)

            @inbounds x_r, x_i = reim(x[ivox])
            
            tmp_r = x_r * ϕ_r - x_i * ϕ_i
            tmp_i = x_r * ϕ_i + x_i * ϕ_r

            csm_r, csm_i = reim(csm[ivox, 1])
            sig_r1 += tmp_r * csm_r - tmp_i * csm_i
            sig_i1 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 2])
            sig_r2 += tmp_r * csm_r - tmp_i * csm_i
            sig_i2 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 3])
            sig_r3 += tmp_r * csm_r - tmp_i * csm_i
            sig_i3 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 4])
            sig_r4 += tmp_r * csm_r - tmp_i * csm_i
            sig_i4 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 5])
            sig_r5 += tmp_r * csm_r - tmp_i * csm_i
            sig_i5 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 6])
            sig_r6 += tmp_r * csm_r - tmp_i * csm_i
            sig_i6 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 7])
            sig_r7 += tmp_r * csm_r - tmp_i * csm_i
            sig_i7 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 8])
            sig_r8 += tmp_r * csm_r - tmp_i * csm_i
            sig_i8 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 9])
            sig_r9 += tmp_r * csm_r - tmp_i * csm_i
            sig_i9 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 10])
            sig_r10 += tmp_r * csm_r - tmp_i * csm_i
            sig_i10 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 11])
            sig_r11 += tmp_r * csm_r - tmp_i * csm_i
            sig_i11 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 12])
            sig_r12 += tmp_r * csm_r - tmp_i * csm_i
            sig_i12 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 13])
            sig_r13 += tmp_r * csm_r - tmp_i * csm_i
            sig_i13 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 14])
            sig_r14 += tmp_r * csm_r - tmp_i * csm_i
            sig_i14 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 15])
            sig_r15 += tmp_r * csm_r - tmp_i * csm_i
            sig_i15 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 16])
            sig_r16 += tmp_r * csm_r - tmp_i * csm_i
            sig_i16 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 17])
            sig_r17 += tmp_r * csm_r - tmp_i * csm_i
            sig_i17 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 18])
            sig_r18 += tmp_r * csm_r - tmp_i * csm_i
            sig_i18 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 19])
            sig_r19 += tmp_r * csm_r - tmp_i * csm_i
            sig_i19 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 20])
            sig_r20 += tmp_r * csm_r - tmp_i * csm_i
            sig_i20 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 21])
            sig_r21 += tmp_r * csm_r - tmp_i * csm_i
            sig_i21 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 22])
            sig_r22 += tmp_r * csm_r - tmp_i * csm_i
            sig_i22 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 23])
            sig_r23 += tmp_r * csm_r - tmp_i * csm_i
            sig_i23 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 24])
            sig_r24 += tmp_r * csm_r - tmp_i * csm_i
            sig_i24 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 25])
            sig_r25 += tmp_r * csm_r - tmp_i * csm_i
            sig_i25 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 26])
            sig_r26 += tmp_r * csm_r - tmp_i * csm_i
            sig_i26 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 27])
            sig_r27 += tmp_r * csm_r - tmp_i * csm_i
            sig_i27 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 28])
            sig_r28 += tmp_r * csm_r - tmp_i * csm_i
            sig_i28 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 29])
            sig_r29 += tmp_r * csm_r - tmp_i * csm_i
            sig_i29 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 30])
            sig_r30 += tmp_r * csm_r - tmp_i * csm_i
            sig_i30 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 31])
            sig_r31 += tmp_r * csm_r - tmp_i * csm_i
            sig_i31 += tmp_r * csm_i + tmp_i * csm_r
            csm_r, csm_i = reim(csm[ivox, 32])
            sig_r32 += tmp_r * csm_r - tmp_i * csm_i
            sig_i32 += tmp_r * csm_i + tmp_i * csm_r
            ivox += stride
        end


        sig_i1, sig_i2, sig_i3, sig_i4, sig_i5, sig_i6, sig_i7, sig_i8 = reduce_warp(sig_i1, sig_i2, sig_i3, sig_i4, sig_i5, sig_i6, sig_i7, sig_i8)
        sig_i9, sig_i10, sig_i11, sig_i12, sig_i13, sig_i14, sig_i15, sig_i16 = reduce_warp(sig_i9, sig_i10, sig_i11, sig_i12, sig_i13, sig_i14, sig_i15, sig_i16)
        sig_i17, sig_i18, sig_i19, sig_i20, sig_i21, sig_i22, sig_i23, sig_i24 = reduce_warp(sig_i17, sig_i18, sig_i19, sig_i20, sig_i21, sig_i22, sig_i23, sig_i24)
        sig_i25, sig_i26, sig_i27, sig_i28, sig_i29, sig_i30, sig_i31, sig_i32 = reduce_warp(sig_i25, sig_i26, sig_i27, sig_i28, sig_i29, sig_i30, sig_i31, sig_i32)
        sig_r1, sig_r2, sig_r3, sig_r4, sig_r5, sig_r6, sig_r7, sig_r8 = reduce_warp(sig_r1, sig_r2, sig_r3, sig_r4, sig_r5, sig_r6, sig_r7, sig_r8)
        sig_r9, sig_r10, sig_r11, sig_r12, sig_r13, sig_r14, sig_r15, sig_r16 = reduce_warp(sig_r9, sig_r10, sig_r11, sig_r12, sig_r13, sig_r14, sig_r15, sig_r16)
        sig_r17, sig_r18, sig_r19, sig_r20, sig_r21, sig_r22, sig_r23, sig_r24 = reduce_warp(sig_r17, sig_r18, sig_r19, sig_r20, sig_r21, sig_r22, sig_r23, sig_r24)
        sig_r25, sig_r26, sig_r27, sig_r28, sig_r29, sig_r30, sig_r31, sig_r32 = reduce_warp(sig_r25, sig_r26, sig_r27, sig_r28, sig_r29, sig_r30, sig_r31, sig_r32)
        
        
        sr = (
            sig_r1, sig_r2, sig_r3, sig_r4, sig_r5, sig_r6, sig_r7, sig_r8,
            sig_r9, sig_r10, sig_r11, sig_r12, sig_r13, sig_r14, sig_r15, sig_r16,
            sig_r17, sig_r18, sig_r19, sig_r20, sig_r21, sig_r22, sig_r23, sig_r24,
            sig_r25, sig_r26, sig_r27, sig_r28, sig_r29, sig_r30, sig_r31, sig_r32,
        )

        si = (
            sig_i1, sig_i2, sig_i3, sig_i4, sig_i5, sig_i6, sig_i7, sig_i8, 
            sig_i9, sig_i10, sig_i11, sig_i12, sig_i13, sig_i14, sig_i15, sig_i16, 
            sig_i17, sig_i18, sig_i19, sig_i20, sig_i21, sig_i22, sig_i23, sig_i24,
            sig_i25, sig_i26, sig_i27, sig_i28, sig_i29, sig_i30, sig_i31, sig_i32,
        )
        itmp = iwarp * nCha
        if lane == 0
            for icha in 1:nCha
                @inbounds s_r[itmp+icha] = sr[icha]
                @inbounds s_i[itmp+icha] = si[icha]
            end
        end
    
        sync_threads()

        icha = tid
        while icha <= nCha
            out_r = zero(T)
            out_i = zero(T)
            for iwarp = 0:nWarp-1
                @inbounds out_r += s_r[iwarp*nCha + icha]
                @inbounds out_i += s_i[iwarp*nCha + icha]
            end
            @inbounds out[isam, icha] = Complex(out_r, out_i)
            icha += stride
        end
    end
    return
end


function CUDA_kernel_ctprod_HighOrderOp!(
    out      ::CuDeviceMatrix{Complex{T}},   # [nVox, nCha]
    y        ::CuDeviceMatrix{Complex{T}},   # [nSam, nCha]
    csm      ::CuDeviceMatrix{Complex{T}},   # [nVox, nCha]
    times    ::CuDeviceVector{T}         ,   # [nSam]
    fieldmap ::CuDeviceVector{T}         ,   # [nVox]
    bf       ::CuDeviceMatrix{T}         ,   # [nVox, nTerm]
    kspha    ::CuDeviceMatrix{T}         ,   # [nTerm, nSam]
    nSam::Int32, nCha::Int32, nTerm::Int32, nVox::Int32
) where {T<:AbstractFloat}
    @fastmath begin
        ivox = blockIdx().x
        stride = blockDim().x
        tid = threadIdx().x
        iterm = tid
        icha = tid
        isam = tid
        is = tid

        lane = (threadIdx().x - 1) % 32
        iwarp = (threadIdx().x - 1) ÷ 32
        nWarp = (blockDim().x + 31) ÷ 32

        if ivox > nVox 
            return
        end
        
        b0    = zero(T)
        ϕ     = zero(T)
        ϕ_r   = zero(T)
        ϕ_i   = zero(T)
        TWOPI = 2f0 * π
        @inbounds b0 = fieldmap[ivox]

        sig_r1 = zero(T)
        sig_r2 = zero(T)
        sig_r3 = zero(T)
        sig_r4 = zero(T)
        sig_r5 = zero(T)
        sig_r6 = zero(T)
        sig_r7 = zero(T)
        sig_r8 = zero(T)
        sig_r9 = zero(T)
        sig_r10 = zero(T)
        sig_r11 = zero(T)
        sig_r12 = zero(T)
        sig_r13 = zero(T)
        sig_r14 = zero(T)
        sig_r15 = zero(T)
        sig_r16 = zero(T)
        sig_r17 = zero(T)
        sig_r18 = zero(T)
        sig_r19 = zero(T)
        sig_r20 = zero(T)
        sig_r21 = zero(T)
        sig_r22 = zero(T)
        sig_r23 = zero(T)
        sig_r24 = zero(T)
        sig_r25 = zero(T)
        sig_r26 = zero(T)
        sig_r27 = zero(T)
        sig_r28 = zero(T)
        sig_r29 = zero(T)
        sig_r30 = zero(T)
        sig_r31 = zero(T)
        sig_r32 = zero(T)
        sig_i1 = zero(T)
        sig_i2 = zero(T)
        sig_i3 = zero(T)
        sig_i4 = zero(T)
        sig_i5 = zero(T)
        sig_i6 = zero(T)
        sig_i7 = zero(T)
        sig_i8 = zero(T)
        sig_i9 = zero(T)
        sig_i10 = zero(T)
        sig_i11 = zero(T)
        sig_i12 = zero(T)
        sig_i13 = zero(T)
        sig_i14 = zero(T)
        sig_i15 = zero(T)
        sig_i16 = zero(T)
        sig_i17 = zero(T)
        sig_i18 = zero(T)
        sig_i19 = zero(T)
        sig_i20 = zero(T)
        sig_i21 = zero(T)
        sig_i22 = zero(T)
        sig_i23 = zero(T)
        sig_i24 = zero(T)
        sig_i25 = zero(T)
        sig_i26 = zero(T)
        sig_i27 = zero(T)
        sig_i28 = zero(T)
        sig_i29 = zero(T)
        sig_i30 = zero(T)
        sig_i31 = zero(T)
        sig_i32 = zero(T)
        # sig_r = @MVector zeros(T, 32)
        # sig_i = @MVector zeros(T, 32)
        # sig_r = MVector(0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0)
        # sig_i = MVector(0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0)

        s_bf  = CuDynamicSharedArray(T, nTerm)       
        s_r   = CuDynamicSharedArray(T, nCha * nWarp, sizeof(T) * nTerm)
        s_i   = CuDynamicSharedArray(T, nCha * nWarp, sizeof(T) * (nCha * nWarp + nTerm))

        while is <= (nCha * nWarp)
            @inbounds s_r[is] = 0
            @inbounds s_i[is] = 0
            is += stride
        end

        # s_kspha = @cuDynamicSharedMem(T, nTerm)
        while iterm <= nTerm
            @inbounds s_bf[iterm] = bf[ivox, iterm]
            iterm += stride
        end

        sync_threads()

        while isam <= nSam
            ϕ = times[isam] * b0
            @inbounds for iterm = 1:nTerm
                ϕ += s_bf[iterm] * kspha[iterm, isam]
            end
            # ϕ_r, ϕ_i = reim(cis(TWOPI * ϕ))
            ϕ_i, ϕ_r = sincos(TWOPI * ϕ)

            # @inbounds x_r, x_i = reim(y[isam])
            # ϕ_r = ϕ_r
            ϕ_i = -ϕ_i

            y_r, y_i = reim(y[isam, 1])
            sig_r1 += ϕ_r * y_r - ϕ_i * y_i
            sig_i1 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 2])
            sig_r2 += ϕ_r * y_r - ϕ_i * y_i
            sig_i2 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 3])
            sig_r3 += ϕ_r * y_r - ϕ_i * y_i
            sig_i3 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 4])
            sig_r4 += ϕ_r * y_r - ϕ_i * y_i
            sig_i4 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 5])
            sig_r5 += ϕ_r * y_r - ϕ_i * y_i
            sig_i5 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 6])
            sig_r6 += ϕ_r * y_r - ϕ_i * y_i
            sig_i6 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 7])
            sig_r7 += ϕ_r * y_r - ϕ_i * y_i
            sig_i7 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 8])
            sig_r8 += ϕ_r * y_r - ϕ_i * y_i
            sig_i8 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 9])
            sig_r9 += ϕ_r * y_r - ϕ_i * y_i
            sig_i9 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 10])
            sig_r10 += ϕ_r * y_r - ϕ_i * y_i
            sig_i10 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 11])
            sig_r11 += ϕ_r * y_r - ϕ_i * y_i
            sig_i11 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 12])
            sig_r12 += ϕ_r * y_r - ϕ_i * y_i
            sig_i12 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 13])
            sig_r13 += ϕ_r * y_r - ϕ_i * y_i
            sig_i13 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 14])
            sig_r14 += ϕ_r * y_r - ϕ_i * y_i
            sig_i14 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 15])
            sig_r15 += ϕ_r * y_r - ϕ_i * y_i
            sig_i15 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 16])
            sig_r16 += ϕ_r * y_r - ϕ_i * y_i
            sig_i16 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 17])
            sig_r17 += ϕ_r * y_r - ϕ_i * y_i
            sig_i17 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 18])
            sig_r18 += ϕ_r * y_r - ϕ_i * y_i
            sig_i18 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 19])
            sig_r19 += ϕ_r * y_r - ϕ_i * y_i
            sig_i19 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 20])
            sig_r20 += ϕ_r * y_r - ϕ_i * y_i
            sig_i20 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 21])
            sig_r21 += ϕ_r * y_r - ϕ_i * y_i
            sig_i21 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 22])
            sig_r22 += ϕ_r * y_r - ϕ_i * y_i
            sig_i22 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 23])
            sig_r23 += ϕ_r * y_r - ϕ_i * y_i
            sig_i23 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 24])
            sig_r24 += ϕ_r * y_r - ϕ_i * y_i
            sig_i24 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 25])
            sig_r25 += ϕ_r * y_r - ϕ_i * y_i
            sig_i25 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 26])
            sig_r26 += ϕ_r * y_r - ϕ_i * y_i
            sig_i26 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 27])
            sig_r27 += ϕ_r * y_r - ϕ_i * y_i
            sig_i27 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 28])
            sig_r28 += ϕ_r * y_r - ϕ_i * y_i
            sig_i28 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 29])
            sig_r29 += ϕ_r * y_r - ϕ_i * y_i
            sig_i29 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 30])
            sig_r30 += ϕ_r * y_r - ϕ_i * y_i
            sig_i30 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 31])
            sig_r31 += ϕ_r * y_r - ϕ_i * y_i
            sig_i31 += ϕ_r * y_i + ϕ_i * y_r
            y_r, y_i = reim(y[isam, 32])
            sig_r32 += ϕ_r * y_r - ϕ_i * y_i
            sig_i32 += ϕ_r * y_i + ϕ_i * y_r

            isam += stride
        end


        sig_i1, sig_i2, sig_i3, sig_i4, sig_i5, sig_i6, sig_i7, sig_i8 = reduce_warp(sig_i1, sig_i2, sig_i3, sig_i4, sig_i5, sig_i6, sig_i7, sig_i8)
        sig_i9, sig_i10, sig_i11, sig_i12, sig_i13, sig_i14, sig_i15, sig_i16 = reduce_warp(sig_i9, sig_i10, sig_i11, sig_i12, sig_i13, sig_i14, sig_i15, sig_i16)
        sig_i17, sig_i18, sig_i19, sig_i20, sig_i21, sig_i22, sig_i23, sig_i24 = reduce_warp(sig_i17, sig_i18, sig_i19, sig_i20, sig_i21, sig_i22, sig_i23, sig_i24)
        sig_i25, sig_i26, sig_i27, sig_i28, sig_i29, sig_i30, sig_i31, sig_i32 = reduce_warp(sig_i25, sig_i26, sig_i27, sig_i28, sig_i29, sig_i30, sig_i31, sig_i32)
        sig_r1, sig_r2, sig_r3, sig_r4, sig_r5, sig_r6, sig_r7, sig_r8 = reduce_warp(sig_r1, sig_r2, sig_r3, sig_r4, sig_r5, sig_r6, sig_r7, sig_r8)
        sig_r9, sig_r10, sig_r11, sig_r12, sig_r13, sig_r14, sig_r15, sig_r16 = reduce_warp(sig_r9, sig_r10, sig_r11, sig_r12, sig_r13, sig_r14, sig_r15, sig_r16)
        sig_r17, sig_r18, sig_r19, sig_r20, sig_r21, sig_r22, sig_r23, sig_r24 = reduce_warp(sig_r17, sig_r18, sig_r19, sig_r20, sig_r21, sig_r22, sig_r23, sig_r24)
        sig_r25, sig_r26, sig_r27, sig_r28, sig_r29, sig_r30, sig_r31, sig_r32 = reduce_warp(sig_r25, sig_r26, sig_r27, sig_r28, sig_r29, sig_r30, sig_r31, sig_r32)
        
        
        sr = (
            sig_r1, sig_r2, sig_r3, sig_r4, sig_r5, sig_r6, sig_r7, sig_r8,
            sig_r9, sig_r10, sig_r11, sig_r12, sig_r13, sig_r14, sig_r15, sig_r16,
            sig_r17, sig_r18, sig_r19, sig_r20, sig_r21, sig_r22, sig_r23, sig_r24,
            sig_r25, sig_r26, sig_r27, sig_r28, sig_r29, sig_r30, sig_r31, sig_r32,
        )

        si = (
            sig_i1, sig_i2, sig_i3, sig_i4, sig_i5, sig_i6, sig_i7, sig_i8, 
            sig_i9, sig_i10, sig_i11, sig_i12, sig_i13, sig_i14, sig_i15, sig_i16, 
            sig_i17, sig_i18, sig_i19, sig_i20, sig_i21, sig_i22, sig_i23, sig_i24,
            sig_i25, sig_i26, sig_i27, sig_i28, sig_i29, sig_i30, sig_i31, sig_i32,
        )
        itmp = iwarp * nCha
        if lane == 0
            for icha in 1:nCha
                @inbounds s_r[itmp+icha] = sr[icha]
                @inbounds s_i[itmp+icha] = si[icha]
            end
        end
    
        sync_threads()

        icha = tid
        while icha <= nCha
            out_r = zero(T)
            out_i = zero(T)
            for iwarp = 0:nWarp-1
                @inbounds out_r += s_r[iwarp*nCha + icha]
                @inbounds out_i += s_i[iwarp*nCha + icha]
            end
            @inbounds out[ivox, icha] = Complex(out_r, out_i)
            icha += stride
        end
    end
    return
end