@testset "multi-GPU normal operator" begin
    @test HighOrderMRI.balanced_channel_ranges(8, 3) == [1:3, 4:6, 7:8]
    @test_throws ArgumentError HighOrderMRI.balanced_channel_ranges(2, 3)

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
                highorder_lowrank_test_data(nDyn=1, nX=8, nY=8)
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
            )

            nPoint = source_op.nSam * source_op.nDyn
            weights = Complex{T}.(CuArray(range(T(0.5), T(1.0); length=nPoint)))
            x = CUDA.randn(Complex{T}, prod(source_op.grid_size))
            source_arrays = map(field -> Array(getproperty(source_op, field)), (:q, :basis, :csm, :mask_idx))
            forward_reference = source_op * x
            adjoint_reference = adjoint(source_op) * forward_reference

            data = CUDA.randn(Complex{T}, source_op.nSam, source_op.nDyn, source_op.nCha)
            weighted_reference = vec(data) .* repeat(weights, source_op.nCha)
            data_pointer = pointer(data)
            data .*= reshape(weights, source_op.nSam, source_op.nDyn, 1)
            @test pointer(data) == data_pointer
            @test Array(vec(data)) == Array(weighted_reference)

            kspace = source_op * x
            kspace .*= repeat(abs2.(weights), source_op.nCha)
            reference = adjoint(source_op) * kspace

            W = WeightingOp(Complex{T}; weights, rep=source_op.nCha)
            E = ∘(W, source_op)
            normal_op = normalOperator(E)
            result = normal_op * x
            relative_error = norm(result - reference) / max(norm(reference), eps(T))
            @test normal_op isa HighOrderMRI.HighOrderLowRankNormalOp
            @test size(normal_op) == (prod(source_op.grid_size), prod(source_op.grid_size))
            @test ishermitian(normal_op)
            @test relative_error < T(1e-4)

            cached_backend = source_op.normal_backend.operator
            primary_shard = first(cached_backend.state.shards)
            # Primary-GPU storage is shared, but each owner must have its own
            # handle so closing the normal backend leaves the parent usable.
            for field in (:q, :basis, :csm, :mask_idx)
                borrowed = getproperty(primary_shard, field)
                owned = getproperty(source_op, field)
                @test borrowed !== owned
                @test pointer(borrowed) == pointer(owned)
            end
            @test cached_backend.state.mask_idx !== source_op.mask_idx
            @test pointer(cached_backend.state.mask_idx) == pointer(source_op.mask_idx)
            second_view = normalOperator(E)
            @test second_view isa HighOrderMRI.HighOrderLowRankNormalOp
            @test source_op.normal_backend.operator === cached_backend
            close(second_view)
            @test !cached_backend.state.released

            close(source_op)
            @test cached_backend.state.released
            @test source_op.normal_backend.operator === nothing

            # Closing the parent is idempotent, including when other normal
            # views still reference the already released state.
            close(source_op)
            @test source_op.normal_backend.operator === nothing

            for (field, original) in zip((:q, :basis, :csm, :mask_idx), source_arrays)
                @test Array(getproperty(source_op, field)) == original
            end
            @test norm(source_op * x - forward_reference) / max(norm(forward_reference), eps(T)) < T(1e-4)
            @test norm(adjoint(source_op) * forward_reference - adjoint_reference) / max(norm(adjoint_reference), eps(T)) < T(1e-4)

            rebuilt_normal = normalOperator(E)
            @test source_op.normal_backend.operator !== cached_backend
            @test norm(rebuilt_normal * x - result) / max(norm(result), eps(T)) < T(1e-4)
            rebuilt_backend = source_op.normal_backend.operator
            close(source_op)
            @test rebuilt_backend.state.released
            close(source_op)
            @test source_op.normal_backend.operator === nothing
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
