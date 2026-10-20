@testset "shared adjoint reuses channel workers" begin
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        @eval import NonuniformFFTs
        previous_backend = AbstractNFFTs.active_backend()
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            for FT in (Float32, Float64)
                @testset "$FT" begin
                    CUDA.device!(first(gpu_ids))
                    _, k, times, fieldmap, _, mask, terms =
                        highorder_lowrank_test_data(nDyn=3, nX=8, nY=6)
                    grid = Grid(8, 6, 1, one(FT), one(FT), one(FT))
                    # Seven distinct complex coil maps exercise uneven channel
                    # shards, conjugation, and a fully zero coil.
                    csm = Complex{FT}[
                        c == 7 ? 0 : (FT(0.5) + FT(v + c) / 100) * cis(FT(v * c) / 17)
                        for v in 1:48, c in 1:7
                    ]
                    op = HighOrderLowRankOp(grid, FT.(k), FT.(times);
                        fieldmap=FT.(fieldmap), csm=reshape(csm, 8, 6, 7), mask,
                        recon_terms=terms, arrayType=CuArray, gpus=gpu_ids,
                        L_rank=2, rsvd_seed=13, rsvd_finalize=:gram,
                        rsvd_backend=:kernel, rsvd_distribution=:single,
                        normal_distribution=:channel)
                    try
                        nPoint = op.nSam * op.nDyn
                        data = CUDA.randn(Complex{FT}, size(op, 1))
                        x = CUDA.randn(Complex{FT}, size(op, 2))
                        serial = adjoint(op) * data
                        forward = op * x
                        weights = CuArray(Complex{FT}.(range(FT(0.4), FT(1.2); length=nPoint)))
                        weighted_serial = adjoint(op) * (data .* repeat(weights, op.nCha))
                        W = WeightingOp(Complex{FT}; weights, rep=op.nCha)
                        E = W ∘ op
                        normal = normalOperator(E)
                        before = normal * x
                        parallel = adjoint(op) * data
                        weighted_parallel = adjoint(E) * data
                        tolerance = FT === Float32 ? FT(2e-5) : FT(1e-11)
                        @test norm(parallel - serial) / norm(serial) < tolerance
                        @test norm(weighted_parallel - weighted_serial) / norm(weighted_serial) < tolerance
                        @test abs(dot(forward, data) - dot(x, parallel)) /
                            max(abs(dot(forward, data)), abs(dot(x, parallel))) < tolerance
                        @test norm(normal * x - before) / norm(before) < tolerance
                        @test Array(parallel)[findall(!, vec(mask))] ==
                            zeros(Complex{FT}, count(!, mask))

                        # Mask-sized outputs retain the standalone fallback.
                        masked = CUDA.zeros(Complex{FT}, op.nVox)
                        HighOrderMRI.ctprod!(masked, op, data)
                        @test norm(Array(masked) - Array(serial)[vec(mask)]) /
                            norm(Array(serial)[vec(mask)]) < tolerance
                        close(op)
                        @test norm(adjoint(op) * data - serial) / norm(serial) < tolerance
                    finally
                        close(op)
                    end
                end
            end
        finally
            AbstractNFFTs.set_active_backend!(previous_backend)
            CUDA.device!(first(gpu_ids))
        end
    end
end

@testset "distributed shared update with zero energy" begin
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        for FT in (Float32, Float64)
            workspace = HighOrderMRI.DistributedRSVDWorkspace(
                zeros(FT, 67), zeros(FT, 67, 1), 3, 3, 3, gpu_ids)
            shared = nothing
            try
                shared = HighOrderMRI.DistributedSharedSpatialBasis(workspace, 3, 3, FT(1e-3))
                V = Complex{FT}[cis(FT(2π * (v - 1) * (r - 1) / 67)) / sqrt(FT(67))
                    for v in 1:67, r in 1:3]
                for dyn in 1:3
                    input = dyn == 2 ? V : zeros(Complex{FT}, size(V))
                    done = Vector{Nothing}(undef, length(workspace.workers))
                    HighOrderMRI.run_on_workers!(done, workspace.workers) do i
                        shard = workspace.shards[i]
                        CUDA.@sync copyto!(shard.v_scaled, input[shard.voxels, :])
                        nothing
                    end
                    error, added = HighOrderMRI.update_distributed_shared_basis!(
                        shared, workspace, dyn, FT(sum(abs2, input)))
                    @test added == (dyn == 2 ? 3 : 0)
                    @test shared.rank == (dyn == 1 ? 0 : 3)
                    @test error <= shared.tol
                    if dyn != 2
                        residuals = Vector{Matrix{Complex{FT}}}(undef, length(workspace.workers))
                        HighOrderMRI.run_on_workers!(residuals, workspace.workers) do i
                            Array(shared.shards[i].residual)
                        end
                        @test all(iszero, Iterators.flatten(residuals))
                    end
                end
            finally
                try
                    HighOrderMRI.release_distributed_workspaces!(workspace, shared)
                finally
                    HighOrderMRI.shutdown_distributed_workers!(workspace.workers)
                end
            end
        end
    end
end
