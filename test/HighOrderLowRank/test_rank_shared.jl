@testset "rank-shared CPU reduction" begin
    rng = MersenneTwister(193)
    for FT in (Float32,Float64), count in (1,5)
        n = 1_000_003
        inputs = [randn(rng,Complex{FT},n+7) for _ in 1:count]
        reference = copy(first(inputs))
        for i in 2:count, p in 1:n
            reference[p] += inputs[i][p]
        end
        result = fill(Complex{FT}(17,-3),n+7)
        HighOrderMRI.rank_shared_sum!(result,inputs,n)
        @test result[1:n] == reference[1:n]
        @test all(==(Complex{FT}(17,-3)),result[n+1:end])
        HighOrderMRI.rank_shared_sum!(first(inputs),inputs,n)
        @test first(inputs) == reference
        HighOrderMRI.rank_shared_sum!(result,inputs,0)
        @test all(==(Complex{FT}(17,-3)),result[n+1:end])
    end
end

@testset "rank-sharded shared factors" begin
    @test HighOrderMRI.rank_shared_ranges(20, 5) == [1:1, 2:6, 7:11, 12:16, 17:20]
    @test HighOrderMRI.rank_shared_ranges(6, 1) == [1:6]
    @test HighOrderMRI.rank_shared_ranges(20, 4; primary_worker=nothing) == [1:5, 6:10, 11:15, 16:20]
    @test HighOrderMRI.rank_shared_ranges(20, 5; primary_worker=3) == [1:5, 6:10, 11:11, 12:16, 17:20]
    @test length.(HighOrderMRI.rank_shared_channel_ranges(32, [1:5,6:10,11:15,16:20],
        31_299_328, 9_272_640, 50_336_000; primary_worker=nothing)) == fill(8,4)
    @test HighOrderMRI.HighOrderNormalBackend(:single, Int[], false, nothing, nothing).primary_gpu == 0
    @test HighOrderMRI.HighOrderNormalBackend(:rank, [1,2], false, nothing, nothing, 37).primary_gpu == 1
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        @eval import NonuniformFFTs
        previous = AbstractNFFTs.active_backend()
        AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
        try
            for FT in (Float32, Float64)
                @testset "$FT" begin
                    CUDA.device!(first(gpu_ids))
                    rng = MersenneTwister(823)
                    shape = (8, 7, 6)
                    grid = Grid(shape..., one(FT), one(FT), one(FT))
                    nSam, nDyn, nCha = 24, 4, 32
                    k = FT(0.03) .* randn(rng, FT, 16, nSam, nDyn)
                    times = repeat(reshape(collect(range(FT(0), FT(.003); length=nSam)), :, 1), 1, nDyn)
                    fieldmap = FT(30) .* randn(rng, FT, shape)
                    csm = randn(rng, Complex{FT}, shape..., nCha)
                    csm[:, :, :, end] .= 0
                    mask = trues(shape)
                    mask[1, 1, 1] = false
                    mask[end, end, end] = false
                    options = (; fieldmap, csm, mask, recon_terms="1111", arrayType=CuArray,
                        gpus=gpu_ids, L_rank=6, rsvd_seed=1234, rsvd_oversample=5,
                        rsvd_finalize=:gram, rsvd_backend=:kernel, rsvd_distribution=:voxel,
                        shared_rank_max=32, shared_basis_tol=FT(.01))
                    serial = HighOrderLowRankOp(grid, k, times; options..., normal_distribution=:single)
                    distributed = HighOrderMRI.rank_shared_operator(
                        Array(serial.q), Array(serial.basis), Array(serial.csm), Array(serial.mask_idx),
                        copy(serial.nfft_traj), shape, 6; gpus=gpu_ids, sample_batch_size=37)
                    tol = FT === Float32 ? FT(4e-5) : FT(2e-11)
                    local x, weights, ref_normal
                    try
                        @test distributed.nfftplan === nothing
                        @test distributed.q isa Matrix
                        @test distributed.basis isa Matrix
                        @test distributed.csm isa Matrix
                        @test isempty(distributed.workspace.grid_flat_prod)
                        x = CuArray(randn(rng, Complex{FT}, prod(shape)))
                        data = randn(rng, Complex{FT}, nSam*nDyn*nCha)
                        weights = randn(rng, Complex{FT}, nSam*nDyn)
                        weights[1] = 0
                        ref_forward = serial*x
                        forward = zeros(Complex{FT}, length(data))
                        mul!(forward, distributed, x)
                        @test norm(forward-Array(ref_forward))/norm(Array(ref_forward)) < tol
                        ref_adjoint = adjoint(serial)*CuArray(data)
                        result = CUDA.zeros(Complex{FT}, prod(shape))
                        mul!(result, adjoint(distributed), data)
                        @test norm(result-ref_adjoint)/norm(ref_adjoint) < tol
                        @test abs(dot(forward, data)-dot(Array(x), Array(result))) /
                            max(abs(dot(forward, data)), abs(dot(Array(x), Array(result)))) < tol
                        normal = normalOperator(distributed, abs2.(weights))
                        weighted = ref_forward .* CuArray(repeat(abs2.(weights), nCha))
                        ref_normal = adjoint(serial)*weighted
                        @test norm(normal*x-ref_normal)/norm(ref_normal) < tol
                        # This oracle contains cross-rank terms and rejects a
                        # sum of independent per-rank normal operators.
                        @test real(dot(x, normal*x)) > 0
                        weighted_data = CuArray(data)
                        w = CuArray(repeat(weights, nCha))
                        weighted_data .*= w
                        weighted_data .*= conj.(w)
                        ref_rhs = adjoint(serial)*weighted_data
                        HighOrderMRI.rank_shared_adjoint!(result, distributed, data; sqrt_weights=weights)
                        @test norm(result-ref_rhs)/norm(ref_rhs) < tol
                        HighOrderMRI.rank_shared_adjoint!(result, distributed, zero.(data))
                        @test all(iszero, Array(result))
                        @test all(iszero, Array(normal*x)[.!vec(mask)])
                        state = distributed.normal_backend.operator.state
                        @test state.csm_host === distributed.csm
                        HighOrderMRI.rank_shared_coil!(state, nCha-1)
                        @test state.host_coil == distributed.csm[:,nCha-1]
                        @test sum(size(s.q, 2) for s in state.shards) == size(serial.q, 2)
                        @test sum(size(s.csm, 2) for s in state.shards) == nCha
                        @test all(size(s.q, 2) < size(serial.q, 2) for s in state.shards)
                        @test all(size(s.csm, 2) < nCha for s in state.shards)
                        @test maximum(length(s.workspace.k_signal) for s in state.shards) == 37
                        close(normal)
                        @test !state.released
                        close(distributed)
                        @test state.released
                        close(distributed)
                        rebuilt = normalOperator(distributed, abs2.(weights))
                        @test norm(rebuilt*x-ref_normal)/norm(ref_normal) < tol
                    finally
                        close(distributed)
                    end
                    # Keep the primary entirely outside the encoding worker list.
                    if length(gpu_ids) >= 2
                        CUDA.device!(first(gpu_ids))
                        dedicated = HighOrderMRI.rank_shared_operator(
                            Array(serial.q), Array(serial.basis), Array(serial.csm), Array(serial.mask_idx),
                            copy(serial.nfft_traj), shape, 6; gpus=gpu_ids[2:end],
                            primary_gpu=first(gpu_ids), sample_batch_size=37)
                        try
                            data = randn(rng, Complex{FT}, nSam*nDyn*nCha)
                            forward = zeros(Complex{FT}, length(data))
                            mul!(forward, dedicated, x)
                            @test norm(forward-Array(serial*x))/norm(forward) < tol
                            result = CUDA.zeros(Complex{FT}, prod(shape))
                            mul!(result, adjoint(dedicated), data)
                            reference = adjoint(serial)*CuArray(data)
                            @test norm(result-reference)/norm(reference) < tol
                            @test abs(dot(forward,data)-dot(Array(x),Array(result))) /
                                max(abs(dot(forward,data)),abs(dot(Array(x),Array(result)))) < tol
                            normal = normalOperator(dedicated, abs2.(weights))
                            @test norm(normal*x-ref_normal)/norm(ref_normal) < tol
                            weighted = CuArray(data) .* CuArray(repeat(abs2.(weights),nCha))
                            reference = adjoint(serial)*weighted
                            HighOrderMRI.rank_shared_adjoint!(result,dedicated,data; sqrt_weights=weights)
                            @test norm(result-reference)/norm(reference) < tol
                            state = dedicated.normal_backend.operator.state
                            @test state.primary_gpu == first(gpu_ids)
                            @test [s.gpu_id for s in state.shards] == gpu_ids[2:end]
                            @test all(s.gpu_id != state.primary_gpu for s in state.shards)
                            @test [s.ranks for s in state.shards] ==
                                HighOrderMRI.balanced_channel_ranges(size(serial.q,2),length(gpu_ids)-1)
                            @test maximum(length(s.channels) for s in state.shards) -
                                minimum(length(s.channels) for s in state.shards) <= 1
                            @test all(CUDA.deviceid(CUDA.device(a)) == first(gpu_ids) for a in
                                (dedicated.Mv,normal.Mv,state.mask_idx,state.primary_input,state.primary_sum))
                            @test state.csm_host === dedicated.csm
                            @test dedicated.nfftplan === nothing
                            @test all(iszero, Array(normal*x)[.!vec(mask)])
                            close(normal)
                            @test !state.released
                            close(dedicated)
                            @test state.released
                            @test norm(normalOperator(dedicated,abs2.(weights))*x-ref_normal)/norm(ref_normal) < tol
                        finally
                            close(dedicated)
                        end
                    end
                    # Exercise the real constructor, phase folding and host
                    # finalization rather than only restoring known factors.
                    built = HighOrderLowRankOp(grid, k, times; options...,
                        normal_distribution=:rank, normal_sample_batch_size=37)
                    try
                        @test norm(built.q-Array(serial.q))/norm(Array(serial.q)) < tol
                        @test norm(built.basis-Array(serial.basis))/norm(Array(serial.basis)) < tol
                        @test norm(normalOperator(built, abs2.(weights))*x-ref_normal)/norm(ref_normal) < tol
                    finally
                        close(built)
                        close(serial)
                    end
                end
            end
        finally
            AbstractNFFTs.set_active_backend!(previous)
            CUDA.device!(first(gpu_ids))
        end
    end
end
