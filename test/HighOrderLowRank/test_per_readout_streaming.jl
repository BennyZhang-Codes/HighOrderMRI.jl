# Run after test_per_readout.jl, whose dense-reference fixture is reused here.
@testset "per-readout prefetch queue" begin
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        length(gpu_ids) >= 2 || error("At least two healthy GPUs are required")
        @eval import NonuniformFFTs
        previous_backend = AbstractNFFTs.active_backend()
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            CUDA.device!(first(gpu_ids))
            for nZ in (1, 10), nDyn in (1, 7)
                data = per_readout_test_problem(; nDyn, nZ)
                common = (;fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
                    arrayType=CuArray, gpus=gpu_ids, L_rank=2, rsvd_oversample=2,
                    rsvd_seed=2026, rsvd_finalize=:gram, rsvd_backend=:kernel,
                    rsvd_distribution=:voxel, normal_distribution=:channel,
                    basis_layout=:per_readout)
                resident = HighOrderLowRankOp(data.grid, data.kspha, data.times; common...)
                x = CuArray(data.x)
                y = CUDA.randn(ComplexF32, size(resident,1))
                fwd = resident*x
                adj = adjoint(resident)*y
                weights = CuArray(complex.(collect(range(T(0.5),one(T);length=size(data.times,1)*size(data.times,2)))))
                reference = adjoint(resident)*(fwd .* repeat(abs2.(weights),resident.nCha))
                @testset "bounded per-readout staging" begin
                    for capacity in (1,2,5)
                        op = HighOrderLowRankOp(data.grid, data.kspha, data.times;common...,
                            per_readout_storage=:cpu,per_readout_cache_size=capacity)
                        @test op.basis isa Matrix{ComplexF32}
                        @test length(op.readout_cache.basis)==min(capacity,nDyn)
                        @test length(op.readout_cache.staging)==1
                        @test length(op.readout_cache.plans)==1
                        @test isempty(op.readout_plans)
                        for _ in 1:2
                            @test norm(Array(op*x-fwd))/norm(Array(fwd)) < 1f-4
                            @test norm(Array(adjoint(op)*y-adj))/norm(Array(adj)) < 1f-4
                        end
                        W = WeightingOp(ComplexF32;weights,rep=op.nCha)
                        N = normalOperator(W ∘ op)
                        @test norm(Array(N*x-reference))/norm(Array(reference)) < 1f-4
                        backend=op.normal_backend.operator
                        # Once the normal backend exists, adjoints reuse its
                        # GPU workers. Compare with the resident reference
                        # and verify both weighting and adjointness.
                        transfers_before = sum(shard.readout_cache.transfers for shard in backend.state.shards)
                        parallel_adj = adjoint(op) * y
                        transfers_after = sum(shard.readout_cache.transfers for shard in backend.state.shards)
                        @test 0 <= transfers_after-transfers_before <= nDyn
                        @test norm(Array(parallel_adj-adj))/norm(Array(adj)) < 1f-4
                        @test abs(dot(fwd, y)-dot(x, parallel_adj)) /
                            max(abs(dot(fwd, y)), abs(dot(x, parallel_adj))) < 1f-4
                        weighted_reference = adjoint(resident) * (y .* repeat(conj.(weights), resident.nCha))
                        weighted_actual = adjoint(W ∘ op) * y
                        @test norm(Array(weighted_actual-weighted_reference)) /
                            norm(Array(weighted_reference)) < 1f-4
                        @test norm(Array(N*x-reference))/norm(Array(reference)) < 1f-4
                        @test norm(Array(op*x-fwd))/norm(Array(fwd)) < 1f-4
                        @test all(shard.readout_cache.source === op.basis for shard in backend.state.shards)
                        @test all(length(shard.readout_cache.basis)==min(capacity,nDyn) for shard in backend.state.shards)
                        stats=HighOrderMRI.per_readout_cache_stats(op.readout_cache)
                        @test nDyn <= stats.transfers <= 5*nDyn
                        @test stats.transferred_bytes == stats.transfers*size(op.basis,1)*op.L*sizeof(ComplexF32)
                        println("CACHE capacity=",capacity," stats=",stats)
                        close(op)
                        @test all(shard.readout_cache.released for shard in backend.state.shards)
                        @test norm(Array(op*x-fwd))/norm(Array(fwd)) < 1f-4
                        N2=normalOperator(W ∘ op)
                        @test norm(Array(N2*x-reference))/norm(Array(reference)) < 1f-4
                        close(op)
                    end
                end
                close(resident)
            end
            @testset "single-device factor finalization" begin
                data = per_readout_test_problem(; nDyn=3)
                x = CuArray(data.x)
                for finalize in (:gram, :svd)
                    common = (;fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
                        arrayType=CuArray, gpus=[first(gpu_ids)], L_rank=2,
                        rsvd_oversample=2, rsvd_seed=2026, rsvd_finalize=finalize,
                        rsvd_backend=:kernel, rsvd_distribution=:single,
                        basis_layout=:per_readout)
                    resident = HighOrderLowRankOp(data.grid,data.kspha,data.times;common...)
                    staged = HighOrderLowRankOp(data.grid,data.kspha,data.times;common...,
                        per_readout_storage=:cpu,per_readout_cache_size=2)
                    reference = resident*x
                    @test norm(Array(staged*x-reference))/norm(Array(reference)) < 1f-4
                    @test staged.basis isa Matrix{ComplexF32}
                    close(resident)
                    close(staged)
                end
            end
        finally
            AbstractNFFTs.set_active_backend!(previous_backend)
            CUDA.device!(first(gpu_ids))
        end
    end
end
