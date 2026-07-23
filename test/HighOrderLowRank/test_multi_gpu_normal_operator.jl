@testset "multi-GPU normal operator" begin
    @test HighOrderMRI.balanced_channel_ranges(8, 3) == [1:3, 4:6, 7:8]
    @test_throws ArgumentError HighOrderMRI.balanced_channel_ranges(2, 3)

    timing = HighOrderMRI.MultiGPUHighOrderNormalTiming(
        [1, 2],
        [1:3, 4:6];
        detailed=true,
    )
    sample = HighOrderMRI.MultiGPUHighOrderNormalWorkerSample(
        1,
        1:3,
        4,
        0.1,
        0.2,
        0.3,
        0.4,
        1.1,
    )
    HighOrderMRI.accumulate_worker_timing!(timing.per_gpu[1], sample)
    @test timing.per_gpu[1].n_calls == 1
    @test timing.per_gpu[1].thread_ids == Set([4])
    @test timing.per_gpu[1].forward_time ≈ 0.2

    timing.n_calls = 1
    timing.total_time_total = 1.5
    HighOrderMRI.reset_multi_gpu_normal_timing!(timing)
    @test timing.n_calls == 0
    @test timing.total_time_total == 0.0
    @test timing.per_gpu[1].n_calls == 0
    @test isempty(timing.per_gpu[1].thread_ids)

    # This integration test is opt-in because it creates persistent workers and
    # one NFFT plan on every selected GPU. It also lets machines with an unhealthy
    # or display-attached GPU exclude that device explicitly.
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(
            Int,
            split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','),
        )
        length(gpu_ids) >= 2 || error("HIGHORDERMRI_TEST_GPUS must list at least two GPUs")
        CUDA.functional() || error("CUDA is not functional")

        nonuniform_path = Base.find_package("NonuniformFFTs")
        nonuniform_path === nothing && error(
            "The opt-in multi-GPU normal test requires NonuniformFFTs",
        )
        @eval import NonuniformFFTs

        previous_backend = AbstractNFFTs.active_backend()
        source_op = nothing

        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            CUDA.device!(first(gpu_ids))

            grid, kspha, times, fieldmap, csm, mask, recon_terms =
                highorder_lowrank_test_data(nDyn=1)
            source_op = HighOrderLowRankOp(
                grid,
                kspha,
                times;
                fieldmap,
                csm,
                mask,
                recon_terms,
                arrayType=CuArray,
                gpus=gpu_ids,
                L_rank=2,
                rsvd_seed=0,
                rsvd_finalize=:gram,
                rsvd_backend=:kernel,
                rsvd_distribution=:single,
                normal_distribution=:channel,
                normal_detailed_timing=true,
            )

            nPoint = source_op.nSam * source_op.nDyn
            weights = Complex{T}.(CuArray(range(T(0.5), T(1.0); length=nPoint)))
            x = CUDA.randn(Complex{T}, prod(source_op.grid_size))

            kspace = source_op * x
            kspace .*= repeat(abs2.(weights), source_op.nCha)
            reference = adjoint(source_op) * kspace

            W = WeightingOp(Complex{T}; weights, rep=source_op.nCha)
            E = ∘(W, source_op)
            normal_op = normalOperator(E)
            result = normal_op * x
            relative_error = norm(result - reference) / max(norm(reference), eps(T))
            @show relative_error HighOrderMRI.multi_gpu_normal_timing(normal_op)

            @test normal_op isa HighOrderLowRankNormalOp
            @test size(normal_op) == (prod(source_op.grid_size), prod(source_op.grid_size))
            @test ishermitian(normal_op)
            @test relative_error < T(1e-4)
            normal_timing = HighOrderMRI.multi_gpu_normal_timing(normal_op)
            @test normal_timing.detailed
            @test normal_timing.n_calls == 1
            @test length(normal_timing.per_gpu) == min(length(gpu_ids), source_op.nCha)
            @test all(worker.n_calls == 1 for worker in normal_timing.per_gpu)
            @test all(worker.forward_time > 0 for worker in normal_timing.per_gpu)
            @test all(worker.adjoint_time > 0 for worker in normal_timing.per_gpu)

            cached_backend = source_op.normal_backend.operator
            second_view = normalOperator(E)
            @test second_view isa HighOrderLowRankNormalOp
            @test source_op.normal_backend.operator === cached_backend
            close(second_view)
            @test !cached_backend.state.released

            reset_multi_gpu_normal_timing!(normal_op)
            @test multi_gpu_normal_timing(normal_op).n_calls == 0
        finally
            if source_op !== nothing
                close(source_op)
                @test source_op.normal_backend.operator === nothing
            end
            AbstractNFFTs.set_active_backend!(previous_backend)
            CUDA.device!(first(gpu_ids))
        end
    end
end
