function per_readout_test_problem(; nZ::Int=1, nDyn::Int=3)
    nX, nY, nSam, nCha = 12, 13, 6, 2
    grid = Grid(nX, nY, nZ, T(0.8), T(1.3), T(1.1))
    spatial_size = nZ == 1 ? (nX, nY) : (nX, nY, nZ)
    nPixel = prod(spatial_size)
    kspha = zeros(T, 9, nSam, nDyn)
    times = zeros(T, nSam, nDyn)
    for dyn in 1:nDyn
        times[:, dyn] .= range(zero(T), T(1.2e-3 * dyn); length=nSam)
        kspha[1, :, dyn] .= T(0.01 * dyn)
        kspha[2, :, dyn] .= range(T(-0.04), T(0.05); length=nSam)
        kspha[3, :, dyn] .= range(T(0.03), T(-0.02); length=nSam) .* T(dyn)
        kspha[4, :, dyn] .= range(T(-0.02), T(0.025); length=nSam)
        kspha[5, :, dyn] .= range(T(-5e-4 * dyn), T(5e-4 * dyn); length=nSam)
        kspha[6, :, dyn] .= T(2e-4 * dyn)
        kspha[9, :, dyn] .= T(2e-4 * dyn)
    end
    coordinate = reshape(collect(range(-one(T), one(T); length=nPixel)), spatial_size)
    fieldmap = T(20) .* coordinate
    csm = Array{Complex{T}}(undef, spatial_size..., nCha)
    for cha in 1:nCha
        selectdim(csm, ndims(csm), cha) .=
            (one(T) .+ T(0.03 * cha) .* coordinate) .* cis.(T(0.05 * cha) .* coordinate)
    end
    mask = trues(spatial_size)
    mask[1] = mask[end] = false
    x = complex.(T(0.55) .+ T(0.45) .* vec(coordinate), T(0.2) .* vec(coordinate))
    return (; grid, kspha, times, fieldmap, csm, mask, x)
end


function per_readout_dense_encoding(data)
    nSam, nDyn = size(data.times)
    nCha = size(data.csm, ndims(data.csm))
    mask = vec(data.mask)
    nVox = count(mask)
    bf = HighOrderMRI.basisfunc_spha(
        data.grid.x[mask], data.grid.y[mask], data.grid.z[mask], collect(1:9),
    )
    fieldmap = vec(data.fieldmap)[mask]
    csm = reshape(data.csm, :, nCha)[mask, :]
    encoding = zeros(Complex{T}, nSam * nDyn * nCha, length(mask))
    for cha in 1:nCha, dyn in 1:nDyn
        phase = bf * data.kspha[:, :, dyn] .+ fieldmap * transpose(data.times[:, dyn])
        offset = (cha - 1) * nSam * nDyn + (dyn - 1) * nSam
        rows = (offset + 1):(offset + nSam)
        encoding[rows, mask] .= transpose(cis.(T(2π) .* phase)) .*
                                transpose(csm[:, cha]) ./ sqrt(T(nVox))
    end
    return encoding
end


per_readout_relative_error(actual, expected) =
    norm(actual - expected) / max(norm(expected), eps(T))


@testset "per-readout low-rank basis" begin
    @testset "dense encoding and adjoint, nZ=$nZ" for nZ in (1, 10)
        data = per_readout_test_problem(; nZ)
        nSam, nDyn = size(data.times)
        common = (;
            fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
            L_rank=nSam, rsvd_oversample=0, rsvd_seed=2026,
            rsvd_finalize=:svd, rsvd_backend=:chunked,
        )
        op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
            common..., basis_layout=:per_readout)
        reference = per_readout_dense_encoding(data)
        rng = Random.Xoshiro(2026)
        y = randn(rng, Complex{T}, size(op, 1))
        forward = op * data.x
        backward = adjoint(op) * y

        @test op.basis_layout === :per_readout
        @test size(op.q) == (nSam * nDyn, nSam)
        @test size(op.basis) == (count(data.mask), nSam * nDyn)
        @test length(op.readout_plans) == nDyn
        @test all(AbstractNFFTs.size_out(plan) == (nSam,) for plan in op.readout_plans)
        @test length(op.workspace.k_signal) == nSam
        @test length(op.workspace.k_weighted) == nSam
        @test per_readout_relative_error(forward, reference * data.x) < T(2e-3)
        @test per_readout_relative_error(backward, adjoint(reference) * y) < T(2e-3)
        lhs, rhs = dot(forward, y), dot(data.x, backward)
        @test abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps(T)) < T(1e-4)
        @test all(iszero, backward[.!vec(data.mask)])
        plans = copy(op.readout_plans)
        @test per_readout_relative_error(op * data.x, forward) < T(1e-5)
        @test all(op.readout_plans[dyn] === plans[dyn] for dyn in 1:nDyn)
        close(op)
    end

    @testset "same local truncation as shared layout" begin
        data = per_readout_test_problem()
        nSam, nDyn = size(data.times)
        rank = 2
        common = (;
            fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
            L_rank=rank, rsvd_oversample=2, rsvd_seed=2026,
            rsvd_finalize=:svd, rsvd_backend=:chunked,
            shared_rank_max=rank * nDyn, shared_basis_tol=T(1e-6),
        )
        per_readout = HighOrderLowRankOp(data.grid, data.kspha, data.times;
            common..., basis_layout=:per_readout)
        shared = HighOrderLowRankOp(data.grid, data.kspha, data.times; common...)
        @test shared.basis_layout === :shared
        @test isempty(shared.readout_plans)
        @test size(per_readout.q) == (nSam * nDyn, rank)
        @test size(per_readout.basis) == (count(data.mask), rank * nDyn)
        y = randn(Random.Xoshiro(2027), Complex{T}, size(per_readout, 1))
        @test per_readout_relative_error(per_readout * data.x, shared * data.x) < T(2e-4)
        @test per_readout_relative_error(adjoint(per_readout) * y, adjoint(shared) * y) < T(2e-4)
        for dyn in 1:nDyn
            rows = ((dyn - 1) * nSam + 1):(dyn * nSam)
            columns = ((dyn - 1) * rank + 1):(dyn * rank)
            local_factor = per_readout.q[rows, :] * adjoint(per_readout.basis[:, columns])
            shared_factor = shared.q[rows, :] * adjoint(shared.basis)
            @test per_readout_relative_error(local_factor, shared_factor) < T(2e-4)
        end
        # Shared compression controls cannot truncate independent readout blocks.
        ignored_shared_controls = merge(common, (; shared_rank_max=0, shared_basis_tol=T(0.5)))
        uncompressed = HighOrderLowRankOp(data.grid, data.kspha, data.times;
            ignored_shared_controls..., basis_layout=:per_readout)
        @test size(uncompressed.basis) == size(per_readout.basis)
        @test per_readout_relative_error(uncompressed * data.x, per_readout * data.x) < T(1e-6)
        close(uncompressed)
        close(per_readout)
        close(shared)

        @test_throws ArgumentError HighOrderLowRankOp(data.grid, data.kspha, data.times;
            common..., basis_layout=:invalid)
        @test_throws ArgumentError HighOrderLowRankOp(data.grid, data.kspha, data.times;
            common..., basis_layout=:per_readout, shared_basis_method=:joint)
        @test_throws ArgumentError HighOrderLowRankOp(data.grid, data.kspha, data.times;
            common..., basis_layout=:per_readout, global_basis_tol=T(0.1))
    end
end


@testset "per-readout multi-GPU normal and rSVD" begin
    # Like the shared-layout integration test, this requires explicit selection
    # of healthy GPUs because it starts persistent workers and NFFT plans.
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        length(gpu_ids) >= 2 || error("HIGHORDERMRI_TEST_GPUS must list at least two GPUs")
        CUDA.functional() || error("CUDA is not functional")
        Base.find_package("NonuniformFFTs") === nothing && error(
            "The per-readout multi-GPU test requires NonuniformFFTs",
        )
        @eval import NonuniformFFTs
        previous_backend = AbstractNFFTs.active_backend()
        source_op = nothing
        distributed_op = nothing
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            CUDA.device!(first(gpu_ids))
            data = per_readout_test_problem()
            nSam, nDyn = size(data.times)
            common = (;
                fieldmap=data.fieldmap, csm=data.csm, mask=data.mask,
                L_rank=nSam, rsvd_oversample=0, rsvd_seed=2026,
                rsvd_finalize=:gram, rsvd_backend=:kernel,
                basis_layout=:per_readout, arrayType=CuArray, gpus=gpu_ids,
            )
            source_op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                common..., rsvd_distribution=:single, normal_distribution=:channel)
            x = CuArray(data.x)
            forward = source_op * x
            backward = adjoint(source_op) * forward
            reference = per_readout_dense_encoding(data) * data.x
            @test per_readout_relative_error(Array(forward), reference) < T(2e-3)
            weights = CuArray(complex.(
                collect(range(T(0.5), one(T); length=nSam * nDyn)),
                collect(range(T(-0.2), T(0.1); length=nSam * nDyn)),
            ))
            weighted_reference = adjoint(source_op) *
                                 (forward .* repeat(abs2.(weights), source_op.nCha))
            weighted = WeightingOp(Complex{T}; weights, rep=source_op.nCha) ∘ source_op
            normal = normalOperator(weighted)
            result = normal * x
            @test normal isa HighOrderMRI.HighOrderLowRankNormalOp
            @test ishermitian(normal)
            @test per_readout_relative_error(result, weighted_reference) < T(1e-4)

            backend = source_op.normal_backend.operator
            @test all(length(shard.readout_plans) == nDyn for shard in backend.state.shards)
            @test all(length(shard.workspace.k_signal) == nSam for shard in backend.state.shards)
            primary_shard = first(backend.state.shards)
            source_fields = (:q, :basis, :csm, :mask_idx)
            source_arrays = map(field -> Array(getproperty(source_op, field)), source_fields)
            for field in source_fields
                borrowed = getproperty(primary_shard, field)
                owned = getproperty(source_op, field)
                @test borrowed !== owned
                @test pointer(borrowed) == pointer(owned)
            end
            second_view = normalOperator(weighted)
            @test source_op.normal_backend.operator === backend
            close(second_view)
            @test !backend.state.released

            close(source_op)
            @test backend.state.released
            @test source_op.normal_backend.operator === nothing
            close(source_op)
            for (field, original) in zip(source_fields, source_arrays)
                @test Array(getproperty(source_op, field)) == original
            end
            @test per_readout_relative_error(source_op * x, forward) < T(1e-5)
            @test per_readout_relative_error(adjoint(source_op) * forward, backward) < T(1e-5)
            rebuilt_normal = normalOperator(weighted)
            @test source_op.normal_backend.operator !== backend
            @test per_readout_relative_error(rebuilt_normal * x, result) < T(1e-4)
            rebuilt_backend = source_op.normal_backend.operator
            close(source_op)
            @test rebuilt_backend.state.released

            distributed_op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                common..., rsvd_distribution=:voxel)
            @test size(distributed_op.q) == (nSam * nDyn, nSam)
            @test size(distributed_op.basis) == (count(data.mask), nSam * nDyn)
            @test per_readout_relative_error(Array(distributed_op * x), reference) < T(2e-3)
        finally
            source_op === nothing || close(source_op)
            distributed_op === nothing || close(distributed_op)
            AbstractNFFTs.set_active_backend!(previous_backend)
            CUDA.device!(first(gpu_ids))
        end
    end
end
