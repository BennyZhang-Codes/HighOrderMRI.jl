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
