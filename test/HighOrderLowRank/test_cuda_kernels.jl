@testset "rSVD CUDA kernels" begin
    if CUDA.functional()
        data = rsvd_test_problem()
        times_d = CuArray(data.times[:, 1])
        fieldmap_d = CuArray(data.fieldmap_masked)
        bf_err_d = CuArray(data.bf_err)
        kspha_err_d = CuArray(data.kspha_err)
        kspha_err_t_d = CuArray(permutedims(data.kspha_err))

        omega = randn(Complex{T}, data.nVox, data.L_total)
        W_ref = data.E_ref * omega
        omega_d = CuArray(omega)
        W_d = CUDA.zeros(Complex{T}, data.nSam, data.L_total)
        HighOrderMRI.run_kernel_rsvd_forward!(
            W_d, omega_d, times_d, fieldmap_d, bf_err_d, kspha_err_d,
        )
        @test norm(Array(W_d) - W_ref) / max(norm(W_ref), eps(T)) < T(1e-4)

        W_transposed_d = similar(W_d)
        HighOrderMRI.run_kernel_rsvd_forward!(
            W_transposed_d,
            omega_d,
            times_d,
            fieldmap_d,
            bf_err_d,
            kspha_err_t_d;
            kspha_transposed=true,
        )
        @test norm(Array(W_transposed_d) - W_ref) /
              max(norm(W_ref), eps(T)) < T(1e-4)
        @test norm(Array(W_transposed_d - W_d)) /
              max(norm(Array(W_d)), eps(T)) < T(1e-5)

        Q = randn(Complex{T}, data.nSam, data.L_total)
        B_ref = adjoint(data.E_ref) * Q
        Q_d = CuArray(Q)
        B_d = CUDA.zeros(Complex{T}, data.nVox, data.L_total)
        HighOrderMRI.run_kernel_rsvd_adjoint!(
            B_d, Q_d, times_d, fieldmap_d, bf_err_d, kspha_err_d;
            threads=128,
        )
        @test norm(Array(B_d) - B_ref) / max(norm(B_ref), eps(T)) < T(1e-4)

        B_transposed_d = similar(B_d)
        HighOrderMRI.run_kernel_rsvd_adjoint!(
            B_transposed_d,
            Q_d,
            times_d,
            fieldmap_d,
            bf_err_d,
            kspha_err_t_d;
            threads=128,
            kspha_transposed=true,
        )
        @test norm(Array(B_transposed_d) - B_ref) /
              max(norm(B_ref), eps(T)) < T(1e-4)
        @test norm(Array(B_transposed_d - B_d)) /
              max(norm(Array(B_d)), eps(T)) < T(1e-5)

        # Sketches wider than one fused-kernel register batch are evaluated
        # in independent column blocks without changing the matrix product.
        for wide_rank in (
            HighOrderMRI.RSVD_KERNEL_RANK_BATCH + 8,
            2 * HighOrderMRI.RSVD_KERNEL_RANK_BATCH + 16,
            3 * HighOrderMRI.RSVD_KERNEL_RANK_BATCH,
        )
            omega_wide = randn(Complex{T}, data.nVox, wide_rank)
            W_wide_ref = data.E_ref * omega_wide
            W_wide_d = CUDA.zeros(Complex{T}, data.nSam, wide_rank)
            HighOrderMRI.run_kernel_rsvd_forward!(
                W_wide_d, CuArray(omega_wide), times_d, fieldmap_d,
                bf_err_d, kspha_err_d,
            )
            @test norm(Array(W_wide_d) - W_wide_ref) /
                  max(norm(W_wide_ref), eps(T)) < T(1e-4)

            Q_wide = randn(Complex{T}, data.nSam, wide_rank)
            B_wide_ref = adjoint(data.E_ref) * Q_wide
            B_wide_d = CUDA.zeros(Complex{T}, data.nVox, wide_rank)
            HighOrderMRI.run_kernel_rsvd_adjoint!(
                B_wide_d, CuArray(Q_wide), times_d, fieldmap_d,
                bf_err_d, kspha_err_d; threads=128,
            )
            @test norm(Array(B_wide_d) - B_wide_ref) /
                  max(norm(B_wide_ref), eps(T)) < T(1e-4)

        end

        L_rank = 1
        L_total = L_rank + data.p_oversample
        workspace = HighOrderMRI.RSVDWorkspace(
            times_d, T, data.nSam, data.nVox, L_total, 3,
        )
        u_chunked, s_chunked, v_chunked = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            data.nVox, data.nSam, L_rank, 3, workspace;
            seed=17,
            p_oversample=data.p_oversample,
            rsvd_finalize=:svd,
            rsvd_backend=:chunked,
        )
        u_kernel, s_kernel, v_kernel = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            data.nVox, data.nSam, L_rank, 3, workspace;
            seed=17,
            p_oversample=data.p_oversample,
            rsvd_finalize=:svd,
            rsvd_backend=:kernel,
        )
        E_chunked = Array(u_chunked * Diagonal(s_chunked) * adjoint(v_chunked))
        E_kernel = Array(u_kernel * Diagonal(s_kernel) * adjoint(v_kernel))
        @test norm(E_kernel - E_chunked) /
              max(norm(E_chunked), eps(T)) < T(1e-3)
    end
end

@testset "rSVD CUDA rank and voxel tails" begin
    if CUDA.functional()
        for S in (Float32, Float64)
            nSam, nVox, M = 67, 131, 12
            times = S(0.02) .* rand(S, nSam)
            fieldmap = randn(S, nVox)
            bf = S(0.02) .* randn(S, nVox, M)
            kspha = randn(S, M, nSam)
            phase = times * transpose(fieldmap) + transpose(kspha) * transpose(bf)
            E_ref = cis.(S(2π) .* phase)
            tolerance = S === Float32 ? S(1e-4) : S(1e-12)

            times_d, fieldmap_d, bf_d = CuArray(times), CuArray(fieldmap), CuArray(bf)
            for L in (11, 17, 20, 33, 40, 80, 96, 97)
                omega = randn(Complex{S}, nVox, L)
                Q = randn(Complex{S}, nSam, L)
                W_ref = E_ref * omega
                B_ref = adjoint(E_ref) * Q
                gram_ref = adjoint(B_ref) * B_ref
                omega_d, Q_d = CuArray(omega), CuArray(Q)

                for transposed in (false, true)
                    kspha_d = CuArray(transposed ? permutedims(kspha) : kspha)
                    W_d = CUDA.zeros(Complex{S}, nSam, L)
                    B_d = CUDA.zeros(Complex{S}, nVox, L)
                    HighOrderMRI.run_kernel_rsvd_forward!(
                        W_d, omega_d, times_d, fieldmap_d, bf_d, kspha_d;
                        kspha_transposed=transposed,
                    )
                    HighOrderMRI.run_kernel_rsvd_adjoint!(
                        B_d, Q_d, times_d, fieldmap_d, bf_d, kspha_d;
                        threads=256, kspha_transposed=transposed,
                    )
                    @test norm(Array(W_d) - W_ref) / norm(W_ref) < tolerance
                    @test norm(Array(B_d) - B_ref) / norm(B_ref) < tolerance

                    # The full Gram includes off-diagonal blocks between batches.
                    gram = Array(adjoint(B_d) * B_d)
                    @test norm(gram - gram_ref) / norm(gram_ref) < tolerance
                end
            end
        end
    end
end
