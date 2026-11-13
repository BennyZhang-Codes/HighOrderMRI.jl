@testset "opt-in distributed shared-basis no-add check" begin
    @testset "unsupported constructor paths" begin
        grid, k, t, b0, csm, mask, terms = highorder_lowrank_test_data()
        @test_throws ArgumentError HighOrderLowRankOp(grid, k, t;
            fieldmap=b0, csm, mask, recon_terms=terms, L_rank=3,
            rsvd_finalize=:gram, shared_noadd_check=true)
    end

    if CUDA.functional()
        @testset "FP64 no-add energy guard and original fallback" begin
            rng = MersenneTwister(20261023)
            gpu_ids = collect(0:min(1, length(collect(CUDA.devices())) - 1))
            for FT in (Float32, Float64)
                nVox, nSam, L, r = 131, 33, 6, 3
                tol = FT(0.005)
                S = Complex{FT}.(Matrix(qr(randn(rng, ComplexF64, nVox, r)).Q)[:, 1:r])
                V = S * randn(rng, Complex{FT}, r, L)
                R = randn(rng, Complex{FT}, nVox, L)
                R -= S * (S' * R)
                R *= norm(V) / norm(R)
                w = HighOrderMRI.DistributedRSVDWorkspace(zeros(FT, nVox),
                    zeros(FT, nVox, 1), nSam, 8, L, gpu_ids)
                shared = HighOrderMRI.DistributedSharedSpatialBasis(w, 1, 12, tol)
                try
                    for (scale, basis_scale) in ((0.0, 1.0), (0.002, 1.0),
                            (Float64(tol) / sqrt(1 - Float64(tol)^2), 1.0),
                            (0.01, 1.0), (0.0, 1.001))
                        X = V + FT(scale) * R
                        energy = FT(sum(abs2, ComplexF64.(X)))
                        outputs = Any[]
                        for check in (false, true)
                            shared.rank = r
                            HighOrderMRI.run_on_workers!(Vector{Nothing}(undef, length(gpu_ids)), w.workers) do i
                                rows = w.shards[i].voxels
                                copyto!(view(shared.shards[i].basis, :, 1:r), FT(basis_scale) .* S[rows, :])
                                copyto!(w.shards[i].v_scaled, X[rows, :])
                                CUDA.synchronize()
                                nothing
                            end
                            error, added = HighOrderMRI.update_distributed_shared_basis!(
                                shared, w, 1, energy; no_add_check=check)
                            basis = HighOrderMRI.gather_distributed_shared_basis(shared, w)
                            push!(outputs, (; error, added, basis, coeff=copy(shared.coeff)))
                        end
                        original, candidate = outputs
                        @test candidate.added == original.added
                        @test candidate.coeff == original.coeff
                        @test candidate.basis == original.basis
                        @test abs(candidate.error - original.error) < (FT === Float32 ? 1e-5 : 1e-12)
                    end
                finally
                    try
                        HighOrderMRI.release_distributed_workspaces!(w, shared)
                    finally
                        HighOrderMRI.shutdown_distributed_workers!(w.workers)
                    end
                end
            end
        end
    end

    if CUDA.functional()
        @testset "Gram bypass and guarded fallback" begin
            nVox, nSam, L, r = 17, 5, 2, 1
            gpu_ids = collect(0:min(1, length(collect(CUDA.devices())) - 1))
            S = zeros(ComplexF64, nVox, r)
            S[1, 1] = 1
            V = zeros(ComplexF64, nVox, L)
            V[1, :] .= 1
            V[2, 1] = V[3, 2] = 0.125
            energy = sum(abs2, V)
            residual_energy = 2 * 0.125^2
            # A few ULPs above the exact boundary is inside the FP64 guard.
            # Float32 cannot resolve this narrow interval reliably.
            near_tol = nextfloat(sqrt(residual_energy / energy), 8)
            @test residual_energy < near_tol^2 * energy
            w = HighOrderMRI.DistributedRSVDWorkspace(zeros(nVox),
                zeros(nVox, 1), nSam, 4, L, gpu_ids)
            shared = HighOrderMRI.DistributedSharedSpatialBasis(w, 1, 4, 0.5)
            try
                for (name, tol, bypass, expected_added) in (
                        ("below threshold", 0.5, true, 0),
                        ("inside guard", near_tol, false, 0),
                        ("above threshold", 0.01, false, 2))
                    @testset "$name" begin
                        shared.tol = tol
                        outputs = Any[]
                        for check in (false, true)
                            shared.rank = r
                            HighOrderMRI.run_on_workers!(Vector{Nothing}(undef, length(gpu_ids)), w.workers) do i
                                rows = w.shards[i].voxels
                                CUDA.@sync begin
                                    fill!(shared.shards[i].basis, 0)
                                    copyto!(view(shared.shards[i].basis, :, 1:r), S[rows, :])
                                    copyto!(w.shards[i].v_scaled, V[rows, :])
                                    # The Gram is workspace, not an input. Its
                                    # sentinel survives only when it is skipped.
                                    fill!(shared.shards[i].gram, ComplexF64(NaN, NaN))
                                end
                                nothing
                            end
                            error, added = if check
                                HighOrderMRI.update_distributed_shared_basis!(
                                    shared, w, 1, energy; no_add_check=true)
                            else
                                HighOrderMRI.update_distributed_shared_basis!(shared, w, 1, energy)
                            end
                            grams = HighOrderMRI.run_on_workers!(
                                Vector{Matrix{ComplexF64}}(undef, length(gpu_ids)), w.workers) do i
                                Array(shared.shards[i].gram)
                            end
                            if check && bypass
                                @test all(g -> all(isnan, g), grams)
                            else
                                @test all(g -> all(isfinite, g), grams)
                            end
                            @test added == expected_added
                            basis = HighOrderMRI.gather_distributed_shared_basis(shared, w)
                            push!(outputs, (; error, added, basis, coeff=copy(shared.coeff)))
                        end
                        original, candidate = outputs
                        @test candidate.coeff == original.coeff
                        @test candidate.basis == original.basis
                        @test abs(candidate.error - original.error) < 1e-12
                    end
                end
            finally
                try
                    HighOrderMRI.release_distributed_workspaces!(w, shared)
                finally
                    HighOrderMRI.shutdown_distributed_workers!(w.workers)
                end
            end
        end
    end
end
