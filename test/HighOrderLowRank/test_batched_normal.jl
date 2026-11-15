@testset "optional batched shared CUDA normal" begin
    if get(ENV,"HIGHORDERMRI_TEST_BATCHED_NORMAL","false") == "true"
        @eval import NonuniformFFTs
        @test Base.get_extension(HighOrderMRI,:HighOrderMRINonuniformFFTsExt) !== nothing
        gpu_ids = parse.(Int,split(get(ENV,"HIGHORDERMRI_TEST_GPUS","0,1"),','))
        previous = AbstractNFFTs.active_backend()
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            CUDA.device!(first(gpu_ids))
            @testset "cooperative gather sample tails" begin
                extension = Base.get_extension(HighOrderMRI, :HighOrderMRINonuniformFFTsExt)
                dims = (8, 8, 8)
                for batch in (2, 4, 8), nt in (1, 2, 3, 4, 15, 16, 17, 31, 32, 33)
                    coils = ComplexF32.(collect(1:batch) ./ 4, fill(-.5f0, batch))
                    grid = CuArray(repeat(reshape(coils, 1, :), prod(dims), 1))
                    q = reshape(ComplexF32.(collect(1:nt) ./ nt, .25f0), :, 1)
                    channel = CUDA.zeros(ComplexF32, nt, batch)
                    CUDA.@cuda threads=128 blocks=cld(nt*8,128) extension.kernel_batched_gather!(
                        channel, grid, CUDA.ones(Int32, nt, 3), CUDA.ones(Float32, nt, 6, 3),
                        CuArray(Int32.(nt:-1:1)), dims, nt, prod(dims), .5f0, CuArray(q), 1, Val(3), Val(batch))
                    # A constant grid and unit windows give a direct 6^3 sum,
                    # independently of the cooperative reduction and permutation.
                    expected = 108f0 .* q .* transpose(coils)
                    @test Array(channel) ≈ expected rtol=2f-6
                end
            end
            for dims in ((9,11,6),(17,19,22)), devices in ([first(gpu_ids)],gpu_ids[1:2])
                @testset "$dims / $(length(devices)) GPUs" begin
                    CUDA.device!(first(devices))
                    rng = MersenneTwister(91237)
                    ns,nd,nc = 67,3,5
                    grid = Grid(dims...,.001f0,.001f0,.001f0)
                    mask = trues(dims); mask[1:13:end] .= false
                    fieldmap = reshape(collect(range(-90f0,90f0;length=prod(dims))),dims)
                    csm = randn(rng,ComplexF32,dims...,nc); csm[:,:,:,nc] .= 0
                    k = .01f0 .* randn(rng,Float32,16,ns,nd)
                    k[2:4,:,:] .*= 60000f0
                    times = reshape(collect(range(.01f0,.014f0;length=ns*nd)),ns,nd)
                    concomitant = 1000f0 .* randn(rng,Float32,4,ns,nd)
                    op = HighOrderLowRankOp(grid,k,times;fieldmap,csm,mask,concomitant,
                        arrayType=CuArray,gpus=devices,L_rank=6,rsvd_oversample=3,
                        rsvd_distribution=:single,rsvd_finalize=:gram,rsvd_seed=13,
                        normal_method=:batched,normal_coil_batch_size=dims==(9,11,6) ? 2 * length(devices) : 8,
                        normal_distribution=length(devices)==1 ? :single : :channel)
                    try
                        x = CuArray(randn(rng,ComplexF32,prod(dims)))
                        data = CuArray(randn(rng,ComplexF32,ns*nd*nc))
                        weights = ComplexF32.(rand(rng,0:16,ns*nd)./16,
                            rand(rng,0:16,ns*nd)./16)
                        weights[1:7:end] .= 0
                        D = abs2.(weights)
                        forward = op*x
                        adjoint_reference = adjoint(op)*data
                        normal_reference = adjoint(op)*(forward.*CuArray(repeat(D,nc)))
                        N = normalOperator(op,CuArray(D))
                        actual = N*x
                        W = WeightingOp(ComplexF32; weights=CuArray(weights),rep=nc)
                        weighted = normalOperator(∘(W,op))
                        @test norm(weighted*x-normal_reference)/norm(normal_reference) < 3f-5
                        @test norm(actual-normal_reference)/norm(normal_reference) < 3f-5
                        @test norm(adjoint(op)*data-adjoint_reference)/norm(adjoint_reference) < 3f-5
                        @test Array(actual)[.!vec(mask)] == zeros(ComplexF32,count(!,mask))
                        @test norm(N*x-actual)/norm(actual) < 3f-5
                        @test abs(dot(forward,data)-dot(x,adjoint(op)*data))/max(abs(dot(forward,data)),abs(dot(x,adjoint(op)*data))) < 3f-5

                        # All workers' forward/adjoint kernels are checked;
                        # RHS inputs here are bare, with no implicit D.
                        backend = op.normal_backend.operator
                        host_x = Array(x)[findall(vec(mask))]
                        host_forward = reshape(Array(forward),ns*nd,nc)
                        checks = Vector{Float64}(undef,length(backend.state.workers))
                        HighOrderMRI.run_on_workers!(checks,backend.state.workers) do i
                            shard = backend.state.shards[i]
                            signal = CUDA.zeros(ComplexF32,ns*nd,length(shard.channels))
                            HighOrderMRI.highorder_batched_forward!(signal,shard.batched_workspace,CuArray(host_x))
                            reference = host_forward[:,shard.channels]
                            error = norm(Array(signal)-reference)/norm(reference)
                            CUDA.unsafe_free!(signal)
                            error
                        end
                        @test maximum(checks) < 3e-5

                        # Independent FP64 Fourier sum of the frozen factors.
                        # Its difference includes the unchanged m3 NUFFT error.
                        indices = findall(vec(mask))
                        coordinates = Float64[i[d]-1-dims[d]÷2 for d in 1:3,i in CartesianIndices(dims)[indices]]
                        carrier = cis.(-2π.*(Float64.(reshape(op.nfft_traj,3,:)')*coordinates))
                        phase = ComplexF64.(Array(op.q))*ComplexF64.(Array(op.basis))'
                        coil_maps = ComplexF64.(Array(op.csm))
                        dense_forward = (carrier.*phase)*(ComplexF64.(host_x).*coil_maps)
                        @test norm(reshape(Array(forward),ns*nd,nc)-dense_forward)/norm(dense_forward) < 2e-3
                        @test_throws ArgumentError normalOperator(op,CuArray(D.+.1f0))
                        close(op)
                        @test backend.state.released
                        @test all(s -> s.batched_workspace === nothing,backend.state.shards)
                        @test norm(op*x-forward)/norm(forward) < 3f-5
                        @test norm(normalOperator(op,CuArray(D))*x-actual)/norm(actual) < 3f-5
                        close(op)
                        @test iszero(norm(normalOperator(op,CUDA.zeros(Float32,ns*nd))*x))
                    finally
                        close(op)
                    end
                end
            end
        finally
            AbstractNFFTs.set_active_backend!(previous)
            CUDA.device!(first(gpu_ids))
        end
    end
end
