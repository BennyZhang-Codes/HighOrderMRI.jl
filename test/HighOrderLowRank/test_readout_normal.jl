@testset "readout-distributed normal operator" begin
    data = per_readout_test_problem()
    @test_throws ArgumentError HighOrderLowRankOp(data.grid, data.kspha, data.times;
        fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
        normal_distribution=:readout)

    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        @eval import NonuniformFFTs
        previous_backend = AbstractNFFTs.active_backend()
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            CUDA.device!(first(gpu_ids))
            @testset "nZ=$nZ, nDyn=$nDyn, nCha=$nCha" for nZ in (1, 10), nDyn in (1, 7), nCha in (1, 2)
                data = per_readout_test_problem(; nZ, nDyn)
                # One coil also verifies that readout distribution can use more
                # GPUs than there are coils, unlike channel distribution.
                coil_maps = selectdim(data.csm, ndims(data.csm), 1:nCha)
                common = (; fieldmap=data.fieldmap, csm=coil_maps, mask=data.mask,
                    arrayType=CuArray, gpus=gpu_ids, L_rank=2, rsvd_oversample=2,
                    rsvd_seed=2026, rsvd_finalize=:gram, rsvd_backend=:kernel,
                    rsvd_distribution=:single, basis_layout=:per_readout)
                reference_op = HighOrderLowRankOp(data.grid, data.kspha, data.times; common...)
                x = CuArray(data.x)
                z = CUDA.randn(ComplexF32, length(x))
                y = CUDA.randn(ComplexF32, size(reference_op, 1))
                nPoint = size(data.times, 1) * nDyn
                weights = CuArray(ComplexF32.(range(0.5, 1; length=nPoint)) .*
                    cis.(Float32.(range(-0.2, 0.3; length=nPoint))))
                forward_reference = reference_op * x
                adjoint_reference = adjoint(reference_op) * y
                normal_reference = adjoint(reference_op) * (forward_reference .* repeat(abs2.(weights), nCha))
                for storage in (:cpu, :gpu), capacity in (1, 5)
                    op = HighOrderLowRankOp(data.grid, data.kspha, data.times; common...,
                        normal_distribution=:readout, per_readout_storage=storage,
                        per_readout_cache_size=capacity)
                    W = WeightingOp(ComplexF32; weights, rep=nCha)
                    E = W ∘ op
                    N = normalOperator(E)
                    state = op.normal_backend.operator.state
                    @test state.distribution === :readout
                    @test length(state.workers) == min(nDyn, length(gpu_ids))
                    @test reduce(vcat, collect(s.readouts) for s in state.shards) == collect(1:nDyn)
                    @test all(s.channels == 1:nCha for s in state.shards)
                    @test pointer(first(state.shards).csm) == pointer(op.csm)
                    for _ in 1:2
                        before = storage === :cpu ? sum(s.readout_cache.transfers for s in state.shards) : 0
                        result = N * x
                        after = storage === :cpu ? sum(s.readout_cache.transfers for s in state.shards) : 0
                        @test 0 <= after - before <= nDyn
                        @test per_readout_relative_error(Array(result), Array(normal_reference)) < 1f-4
                    end
                    Nx, Nz = N * x, N * z
                    @test abs(dot(z, Nx) - dot(Nz, x)) /
                        max(abs(dot(z, Nx)), abs(dot(Nz, x)), eps(Float32)) < 1f-4
                    @test real(dot(x, Nx)) > 0
                    @test per_readout_relative_error(Array(adjoint(op) * y), Array(adjoint_reference)) < 1f-4
                    @test per_readout_relative_error(Array(op * x), Array(forward_reference)) < 1f-4
                    # Initial adjoints reuse complete maps in readout mode;
                    # their cleanup must leave the normal worker maps alive.
                    @test per_readout_relative_error(Array(N * x), Array(normal_reference)) < 1f-4
                    cached = op.normal_backend.operator
                    close(op)
                    @test cached.state.released
                    @test per_readout_relative_error(Array(op * x), Array(forward_reference)) < 1f-4
                    rebuilt = normalOperator(E)
                    @test per_readout_relative_error(Array(rebuilt * x), Array(normal_reference)) < 1f-4
                    close(op)
                end
                close(reference_op)
            end
        finally
            AbstractNFFTs.set_active_backend!(previous_backend)
            CUDA.device!(first(gpu_ids))
        end
    end
end
