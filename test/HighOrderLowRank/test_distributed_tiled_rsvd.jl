@testset "shared rSVD tiling and bounded workspace" begin
    @test HighOrderMRI.rsvd_forward_partitions(Float64, 1_129_804, 1232, 11) == 0
    @test HighOrderMRI.rsvd_forward_partitions(Float32, 131, 67, 11) == 0
    parts = HighOrderMRI.rsvd_forward_partitions(Float32, 1_129_804, 1232, 11)
    @test parts > 1
    @test 1232 * 11 * parts * sizeof(ComplexF32) <= 64 * 1024^2
    @test HighOrderMRI.rsvd_forward_partitions(Float32, 1_000_000, 100_000, 100) == 0

    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        nVox, nSam, L, rank = 170_005, 257, 3, 2
        rng = MersenneTwister(904)
        fieldmap = randn(rng, Float32, nVox)
        bf = Float32(0.02) .* randn(rng, Float32, nVox, 2)
        times = Float32(0.02) .* rand(rng, Float32, nSam)
        kspha = randn(rng, Float32, 2, nSam)
        omega = randn(rng, ComplexF32, nVox, L)
        baseline = HighOrderMRI.DistributedRSVDWorkspace(fieldmap, bf, nSam, L, rank, gpu_ids)
        optimized = nothing
        try
            optimized = HighOrderMRI.DistributedRSVDWorkspace(
                fieldmap, bf, nSam, L, rank, gpu_ids; kernel_tiling=true)
            @test baseline.adjoint_lanes == 32
            @test optimized.adjoint_lanes == 2
            @test all(shard -> size(shard.W_partial, 3) > 1, optimized.shards)
            @test all(shard -> sizeof(shard.W_partial) <= 64 * 1024^2, optimized.shards)
            W0 = HighOrderMRI.distributed_rsvd_forward!(baseline, times, kspha; omega)
            W1 = HighOrderMRI.distributed_rsvd_forward!(optimized, times, kspha; omega)
            @test norm(W1 - W0) / norm(W0) < 1f-4
            Q = Matrix(qr(W0).Q * Matrix{ComplexF32}(I, nSam, L))
            G0 = HighOrderMRI.distributed_rsvd_adjoint_gram!(baseline, Q)
            G1 = HighOrderMRI.distributed_rsvd_adjoint_gram!(optimized, Q)
            @test norm(G1 - G0) / norm(G0) < 1f-4
            b0 = Vector{Matrix{ComplexF32}}(undef, length(gpu_ids))
            b1 = similar(b0)
            HighOrderMRI.run_on_workers!(b0, baseline.workers) do i
                Array(baseline.shards[i].B_adj)
            end
            HighOrderMRI.run_on_workers!(b1, optimized.workers) do i
                Array(optimized.shards[i].B_adj)
            end
            @test all(norm(x - y) / norm(x) < 1f-4 for (x, y) in zip(b0, b1))
        finally
            for workspace in (baseline, optimized)
                workspace === nothing && continue
                try
                    HighOrderMRI.release_distributed_workspaces!(workspace)
                finally
                    HighOrderMRI.shutdown_distributed_workers!(workspace.workers)
                end
            end
        end
    end
end
