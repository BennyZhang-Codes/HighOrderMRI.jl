@testset "rSVD CUDA kernels" begin
    if CUDA.functional()
        data = rsvd_test_problem()
        times_d = CuArray(data.times[:, 1])
        fieldmap_d = CuArray(data.fieldmap_masked)
        bf_err_d = CuArray(data.bf_err)
        kspha_err_d = CuArray(data.kspha_err)
        kspha_err_t_d = CuArray(permutedims(data.kspha_err))

        omega_ref = randn(Complex{T}, data.nVox, data.L_total)
        W_ref = data.E_ref * omega_ref
        omega_d = CuArray(omega_ref)
        W_d = CUDA.zeros(Complex{T}, data.nSam, data.L_total)
        HighOrderMRI.run_kernel_rsvd_forward!(
            W_d, omega_d, times_d, fieldmap_d, bf_err_d, kspha_err_d,
        )
        forward_kernel_error = norm(Array(W_d) - W_ref) / max(norm(W_ref), eps(T))
        @show forward_kernel_error
        @test forward_kernel_error < T(1e-4)

        W_transposed_d = CUDA.zeros(Complex{T}, data.nSam, data.L_total)
        HighOrderMRI.run_kernel_rsvd_forward!(
            W_transposed_d,
            omega_d,
            times_d,
            fieldmap_d,
            bf_err_d,
            kspha_err_t_d;
            kspha_transposed=true,
        )
        W_transposed = Array(W_transposed_d)
        forward_transposed_error =
            norm(W_transposed - W_ref) / max(norm(W_ref), eps(T))
        forward_layout_error =
            norm(W_transposed - Array(W_d)) / max(norm(Array(W_d)), eps(T))
        @show forward_transposed_error forward_layout_error
        @test forward_transposed_error < T(1e-4)
        @test forward_layout_error < T(1e-5)

        W_fast_d = CUDA.zeros(Complex{T}, data.nSam, data.L_total)
        HighOrderMRI.run_kernel_rsvd_forward!(
            W_fast_d,
            omega_d,
            times_d,
            fieldmap_d,
            bf_err_d,
            kspha_err_d;
            fastmath=true,
        )
        W_fast = Array(W_fast_d)
        forward_fastmath_error = norm(W_fast - W_ref) / max(norm(W_ref), eps(T))
        @show forward_fastmath_error
        @test forward_fastmath_error < T(1e-4)

        Q_ref = randn(Complex{T}, data.nSam, data.L_total)
        B_adj_ref = adjoint(data.E_ref) * Q_ref
        Q_d = CuArray(Q_ref)
        B_adj_d = CUDA.zeros(Complex{T}, data.nVox, data.L_total)
        HighOrderMRI.run_kernel_rsvd_adjoint!(
            B_adj_d, Q_d, times_d, fieldmap_d, bf_err_d, kspha_err_d,
        )
        B_adj_kernel = Array(B_adj_d)
        adjoint_kernel_error =
            norm(B_adj_kernel - B_adj_ref) / max(norm(B_adj_ref), eps(T))
        @show adjoint_kernel_error
        @test adjoint_kernel_error < T(1e-4)

        B_adj_warp_d = CUDA.zeros(Complex{T}, data.nVox, data.L_total)
        HighOrderMRI.run_kernel_rsvd_adjoint_warp!(
            B_adj_warp_d, Q_d, times_d, fieldmap_d, bf_err_d, kspha_err_d;
            threads=128,
        )
        B_adj_warp = Array(B_adj_warp_d)
        adjoint_warp_error =
            norm(B_adj_warp - B_adj_ref) / max(norm(B_adj_ref), eps(T))
        adjoint_layout_error =
            norm(B_adj_warp - B_adj_kernel) / max(norm(B_adj_kernel), eps(T))
        @show adjoint_warp_error adjoint_layout_error
        @test adjoint_warp_error < T(1e-4)
        @test adjoint_layout_error < T(1e-4)

        B_adj_transposed_d = CUDA.zeros(Complex{T}, data.nVox, data.L_total)
        HighOrderMRI.run_kernel_rsvd_adjoint_warp!(
            B_adj_transposed_d,
            Q_d,
            times_d,
            fieldmap_d,
            bf_err_d,
            kspha_err_t_d;
            threads=128,
            kspha_transposed=true,
        )
        B_adj_transposed = Array(B_adj_transposed_d)
        adjoint_transposed_error =
            norm(B_adj_transposed - B_adj_ref) / max(norm(B_adj_ref), eps(T))
        adjoint_transposed_layout_error =
            norm(B_adj_transposed - B_adj_warp) / max(norm(B_adj_warp), eps(T))
        @show adjoint_transposed_error adjoint_transposed_layout_error
        @test adjoint_transposed_error < T(1e-4)
        @test adjoint_transposed_layout_error < T(1e-5)

        B_adj_fast_d = CUDA.zeros(Complex{T}, data.nVox, data.L_total)
        HighOrderMRI.run_kernel_rsvd_adjoint_warp!(
            B_adj_fast_d,
            Q_d,
            times_d,
            fieldmap_d,
            bf_err_d,
            kspha_err_d;
            threads=128,
            fastmath=true,
        )
        B_adj_fast = Array(B_adj_fast_d)
        adjoint_fastmath_error =
            norm(B_adj_fast - B_adj_ref) / max(norm(B_adj_ref), eps(T))
        fastmath_adjoint_error =
            abs(dot(W_fast, Q_ref) - dot(omega_ref, B_adj_fast)) /
            max(abs(dot(W_fast, Q_ref)), abs(dot(omega_ref, B_adj_fast)), eps(T))
        @show adjoint_fastmath_error fastmath_adjoint_error
        @test adjoint_fastmath_error < T(1e-4)
        @test fastmath_adjoint_error < T(1e-4)

        L_rank = 1
        L_total = L_rank + data.p_oversample
        workspace = HighOrderMRI.RSVDWorkspace(
            times_d, T, data.nSam, data.nVox, L_total, 3,
        )
        u_svd, s_svd, v_svd = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            data.nVox, data.nSam, L_rank, 3, workspace;
            seed=17,
            p_oversample=data.p_oversample,
            rsvd_finalize=:svd,
        )
        u_kernel, s_kernel, v_kernel = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            data.nVox, data.nSam, L_rank, 3, workspace;
            seed=17,
            p_oversample=data.p_oversample,
            rsvd_finalize=:svd,
            rsvd_backend=:kernel,
        )
        E_svd = Array(u_svd * Diagonal(s_svd) * adjoint(v_svd))
        E_kernel = Array(u_kernel * Diagonal(s_kernel) * adjoint(v_kernel))
        integrated_kernel_error = norm(E_kernel - E_svd) / max(norm(E_svd), eps(T))
        @show integrated_kernel_error
        @test integrated_kernel_error < T(1e-3)

        u_fast, s_fast, v_fast = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            data.nVox, data.nSam, L_rank, 3, workspace;
            seed=17,
            p_oversample=data.p_oversample,
            rsvd_finalize=:svd,
            rsvd_backend=:kernel,
            rsvd_fastmath=true,
        )
        E_fast = Array(u_fast * Diagonal(s_fast) * adjoint(v_fast))
        integrated_fastmath_error = norm(E_fast - E_svd) / max(norm(E_svd), eps(T))
        @show integrated_fastmath_error
        @test integrated_fastmath_error < T(1e-3)
    end
end
