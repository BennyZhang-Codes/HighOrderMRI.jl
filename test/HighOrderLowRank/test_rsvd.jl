@testset "rSVD" begin
    data = rsvd_test_problem()

    @testset "CPU chunked and Gram finalization" begin
        workspace_full = HighOrderMRI.RSVDWorkspace(
            data.times[:, 1], T, data.nSam, data.nVox, data.L_total, data.nVox,
        )
        @test size(workspace_full.omega) == (data.nVox, data.L_total)
        @test size(workspace_full.W) == (data.nSam, data.L_total)
        @test size(workspace_full.B_adj) == (data.nVox, data.L_total)
        @test size(workspace_full.gram) == (data.L_total, data.L_total)
        @test size(workspace_full.right_vectors) == (data.L_total, data.L_total)

        u_full, s_full, v_full = HighOrderMRI.perform_rsvd(
            data.times[:, 1], data.fieldmap_masked, data.bf_err, data.kspha_err,
            data.nVox, data.nSam, data.L_rank, data.nVox, workspace_full;
            seed=17, p_oversample=data.p_oversample,
        )

        workspace_chunk = HighOrderMRI.RSVDWorkspace(
            data.times[:, 1], T, data.nSam, data.nVox, data.L_total, 3,
        )
        u_chunk, s_chunk, v_chunk = HighOrderMRI.perform_rsvd(
            data.times[:, 1], data.fieldmap_masked, data.bf_err, data.kspha_err,
            data.nVox, data.nSam, data.L_rank, 3, workspace_chunk;
            seed=17, p_oversample=data.p_oversample,
        )

        E_full = u_full * Diagonal(s_full) * adjoint(v_full)
        E_chunk = u_chunk * Diagonal(s_chunk) * adjoint(v_chunk)
        @test norm(E_full - E_chunk) / norm(E_full) < T(1e-4)

        v_scaled = zeros(Complex{T}, data.nVox, data.L_rank)
        u_gram, energy_gram = HighOrderMRI.perform_rsvd(
            data.times[:, 1], data.fieldmap_masked, data.bf_err, data.kspha_err,
            data.nVox, data.nSam, data.L_rank, 3, workspace_chunk;
            seed=17,
            p_oversample=data.p_oversample,
            rsvd_finalize=:gram,
            v_scaled=v_scaled,
            gram_allow_fallback=false,
        )
        E_gram = u_gram * adjoint(v_scaled)

        @test norm(E_chunk - E_gram) / norm(E_chunk) < T(1e-3)
        @test energy_gram ≈ sum(abs2, s_chunk) rtol=T(1e-3)
    end

    if CUDA.functional()
        @testset "CuArray chunked and Gram finalization" begin
            times_d = CuArray(data.times[:, 1])
            fieldmap_d = CuArray(data.fieldmap_masked)
            bf_err_d = CuArray(data.bf_err)
            kspha_err_d = CuArray(data.kspha_err)

            workspace_full = HighOrderMRI.RSVDWorkspace(
                times_d, T, data.nSam, data.nVox, data.L_total, data.nVox,
            )
            @test size(workspace_full.omega) == (data.nVox, data.L_total)
            @test size(workspace_full.W) == (data.nSam, data.L_total)
            @test size(workspace_full.B_adj) == (data.nVox, data.L_total)
            @test size(workspace_full.gram) == (data.L_total, data.L_total)
            @test size(workspace_full.right_vectors) == (data.L_total, data.L_total)

            u_full, s_full, v_full = HighOrderMRI.perform_rsvd(
                times_d, fieldmap_d, bf_err_d, kspha_err_d,
                data.nVox, data.nSam, data.L_rank, data.nVox, workspace_full;
                seed=17, p_oversample=data.p_oversample,
            )

            workspace_chunk = HighOrderMRI.RSVDWorkspace(
                times_d, T, data.nSam, data.nVox, data.L_total, 3,
            )
            u_chunk, s_chunk, v_chunk = HighOrderMRI.perform_rsvd(
                times_d, fieldmap_d, bf_err_d, kspha_err_d,
                data.nVox, data.nSam, data.L_rank, 3, workspace_chunk;
                seed=17, p_oversample=data.p_oversample,
            )

            E_full = Array(u_full * Diagonal(s_full) * adjoint(v_full))
            E_chunk = Array(u_chunk * Diagonal(s_chunk) * adjoint(v_chunk))
            relerr_chunk = norm(E_full - E_chunk) / norm(E_full)
            @test size(u_chunk) == (data.nSam, data.L_rank)
            @test size(v_chunk) == (data.nVox, data.L_rank)
            @test length(s_chunk) == data.L_rank
            @test all(isfinite, Array(s_chunk))
            @test relerr_chunk < T(1e-4)

            L_rank_gram = 1
            L_total_gram = L_rank_gram + data.p_oversample
            workspace_gram = HighOrderMRI.RSVDWorkspace(
                times_d, T, data.nSam, data.nVox, L_total_gram, 3,
            )
            v_scaled = similar(times_d, Complex{T}, data.nVox, L_rank_gram)
            u_gram, energy_gram = HighOrderMRI.perform_rsvd(
                times_d, fieldmap_d, bf_err_d, kspha_err_d,
                data.nVox, data.nSam, L_rank_gram, 3, workspace_gram;
                seed=17,
                p_oversample=data.p_oversample,
                rsvd_finalize=:gram,
                v_scaled=v_scaled,
                gram_allow_fallback=false,
            )
            u_svd, s_svd, v_svd = HighOrderMRI.perform_rsvd(
                times_d, fieldmap_d, bf_err_d, kspha_err_d,
                data.nVox, data.nSam, L_rank_gram, 3, workspace_gram;
                seed=17,
                p_oversample=data.p_oversample,
                rsvd_finalize=:svd,
            )

            E_svd = Array(u_svd * Diagonal(s_svd) * adjoint(v_svd))
            E_gram = Array(u_gram * adjoint(v_scaled))
            gram_error = norm(E_svd - E_gram) / max(norm(E_svd), eps(T))
            @test gram_error < T(1e-3)
            @test energy_gram ≈ sum(abs2, Array(s_svd)) rtol=T(1e-3)
            @test all(isfinite, Array(u_gram))
            @test all(isfinite, Array(v_scaled))
        end
    end
end
