@testset "Q/Z regenerated spatial factors" begin
    data = per_readout_test_problem()
    @test_throws ArgumentError HighOrderLowRankOp(data.grid, data.kspha, data.times;
        per_readout_storage=:regenerate)
    @test_throws ArgumentError HighOrderLowRankOp(data.grid, data.kspha, data.times;
        per_readout_host_cache_bytes=-1)

    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        previous_backend = AbstractNFFTs.active_backend()
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            CUDA.device!(first(gpu_ids))
            @testset "nZ=$nZ, nDyn=$nDyn" for nZ in (1, 10), nDyn in (1, 7)
                data = per_readout_test_problem(; nZ, nDyn)
                common = (;fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
                    arrayType=CuArray, gpus=gpu_ids, L_rank=2, rsvd_oversample=2,
                    rsvd_seed=2026, rsvd_finalize=:gram, rsvd_backend=:kernel,
                    rsvd_distribution=:voxel, basis_layout=:per_readout)
                reference = HighOrderLowRankOp(data.grid,data.kspha,data.times; common...,
                    per_readout_storage=:cpu)
                x=CuArray(data.x)
                y=CUDA.randn(ComplexF32,size(reference,1))
                forward=reference*x
                backward=adjoint(reference)*y
                weights=CuArray(ComplexF32.(range(0.5,1;length=size(data.times,1)*nDyn)))
                normal_reference=adjoint(reference)*(forward .* repeat(abs2.(weights),reference.nCha))
                block_bytes=size(reference.basis,1)*2*sizeof(ComplexF32)
                for capacity in (1,2,5), host_blocks in (0,2)
                    @testset "capacity=$capacity, host_blocks=$host_blocks" begin
                        budget=host_blocks*block_bytes + block_bytes÷2
                        op=HighOrderLowRankOp(data.grid,data.kspha,data.times;common...,
                            per_readout_storage=:regenerate, per_readout_cache_size=capacity,
                            per_readout_host_cache_bytes=budget, normal_distribution=:readout)
                        try
                            source=op.basis
                            @test source isa HighOrderMRI.RegeneratedSpatialBasis{Float32}
                            @test size(source)==size(reference.basis)
                            @test size(source.Q)==(size(data.times,1),4,nDyn)
                            @test sizeof(source.host_basis)<=budget
                            @test count(!iszero,source.host_slot)==min(host_blocks,nDyn)
                            @test Array(op.q)==Array(reference.q)
                            @test_throws ArgumentError Array(source)
                            cache=op.readout_cache
                            for _ in 1:2, dyn in 1:nDyn
                                HighOrderMRI.prefetch_readout!(cache,dyn)
                                slot=HighOrderMRI.acquire_readout!(cache,dyn)
                                actual=Array(cache.basis[slot])
                                expected=reference.basis[:,((dyn-1)*2+1):(dyn*2)]
                                @test per_readout_relative_error(actual,expected)<2f-6
                                HighOrderMRI.finish_readout!(cache,dyn)
                            end
                            stats=HighOrderMRI.per_readout_cache_stats(cache)
                            if host_blocks==0
                                @test stats.generated_readouts>=nDyn
                                @test stats.transfers==0
                            else
                                @test stats.host_cache_hits>0
                            end
                            @test per_readout_relative_error(Array(op*x),Array(forward))<1f-4
                            @test per_readout_relative_error(Array(adjoint(op)*y),Array(backward))<1f-4
                            N=normalOperator(WeightingOp(ComplexF32;weights,rep=op.nCha) ∘ op)
                            for _ in 1:2
                                @test per_readout_relative_error(Array(N*x),Array(normal_reference))<1f-4
                            end
                            @test per_readout_relative_error(Array(adjoint(op)*y),Array(backward))<1f-4
                            generators=[op.readout_cache.generator;
                                [s.readout_cache.generator for s in op.normal_backend.operator.state.shards]]
                            close(op)
                            @test all(g.released && istaskdone(g.worker.task) for g in generators)
                            @test per_readout_relative_error(Array(op*x),Array(forward))<1f-4
                        finally
                            close(op)
                        end
                    end
                end
                close(reference)
            end
            @testset "single/channel normal and producer error recovery" begin
                data=per_readout_test_problem(;nDyn=7)
                common=(;fieldmap=data.fieldmap,csm=data.csm,mask=data.mask,
                    arrayType=CuArray,gpus=gpu_ids,L_rank=2,rsvd_oversample=2,
                    rsvd_finalize=:gram,rsvd_backend=:kernel,rsvd_distribution=:voxel,
                    basis_layout=:per_readout,per_readout_storage=:regenerate,
                    per_readout_cache_size=2)
                reference=HighOrderLowRankOp(data.grid,data.kspha,data.times;common...)
                x=CuArray(data.x)
                expected=adjoint(reference)*(reference*x)
                for mode in (:single,:channel)
                    op=HighOrderLowRankOp(data.grid,data.kspha,data.times;common...,normal_distribution=mode)
                    try
                        @test per_readout_relative_error(Array(normalOperator(op)*x),Array(expected))<1f-4
                        worker=op.readout_cache.generator.worker
                        @test_throws HighOrderMRI.DistributedGPUWorkerError HighOrderMRI.run_on_workers!(
                            Vector{Nothing}(undef,1),[worker];phase=:injected_generation_error) do _
                            error("injected generation worker failure")
                        end
                        @test per_readout_relative_error(Array(op*x),Array(reference*x))<1f-4
                    finally
                        close(op)
                    end
                end
                close(reference)
            end
        finally
            AbstractNFFTs.set_active_backend!(previous_backend)
            CUDA.device!(first(gpu_ids))
        end
    end
end
