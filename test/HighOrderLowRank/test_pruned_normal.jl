@testset "experimental pruned CUDA normal" begin
    if get(ENV,"HIGHORDERMRI_TEST_PRUNED_NORMAL","false") == "true"
        @eval import NonuniformFFTs, Libdl
        extension = Base.get_extension(HighOrderMRI,:HighOrderMRINonuniformFFTsExt)
        gpu_ids = parse.(Int,split(get(ENV,"HIGHORDERMRI_TEST_GPUS","0,1,2"),','))
        options = (fft=:vkfft_pruned,native_ptx=ENV["HIGHORDERMRI_NATIVE_PTX"],
            vkfft_library=ENV["HIGHORDERMRI_VKFFT_LIBRARY"],coefficient_cache_bytes=1<<30)
        previous = AbstractNFFTs.active_backend()
        try
            CUDA.device!(first(gpu_ids))
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            @testset "native index limits before allocation" begin
                plan = (data=(us=(zeros(ComplexF32,1,1,1),),),)
                chunks = zeros(Int32,0,4)
                nt = Int(typemax(Int32))÷18+1
                @test_throws ArgumentError extension.pruned_batched_plan(plan,
                    reshape(1:3nt,nt,3),nothing,Int32[],chunks,Int32[],(1,1,1),options,Val(2))
                entries = Int(typemax(Int32))÷12+1
                budget = merge(options,(coefficient_cache_bytes=typemax(Int),))
                @test_throws ArgumentError extension.pruned_batched_plan(plan,
                    reshape(1:3,1,3),nothing,1:entries,chunks,Int32[],(1,1,1),budget,Val(2))
            end
            @testset "pruned FFT against independent FP64 sum" begin
                for dims in ((11,9,7),(12,10,8))
                    rng = MersenneTwister(3719)
                    batch = 3
                    host = randn(rng,ComplexF32,dims...,batch)
                    left = div.(dims,3);right = left .+ 2
                    keep = map(CartesianIndices(dims)) do I
                        all(d -> !(left[d] <= I[d]-1 < right[d]),1:3)
                    end
                    coordinates = hcat([collect(Tuple(I)).-1 for I in CartesianIndices(dims)]...)
                    E = cis.(-2π.*(transpose(Float64.(coordinates)./collect(dims))*Float64.(coordinates)))
                    buffer = CuArray(host)
                    plan = extension.PrunedFFTPlan(buffer,options.vkfft_library,left,right)
                    try
                        for direction in (-1,1)
                            input = direction==1 ? host.*keep : host
                            copyto!(buffer,input)
                            extension.PrunedFFTTransform(plan,Int32(direction))*buffer
                            expected = (direction==-1 ? E : conj.(E))*reshape(ComplexF64.(input),:,batch)
                            actual = reshape(ComplexF64.(Array(buffer)),:,batch)
                            rows = direction==-1 ? vec(keep) : trues(prod(dims))
                            @test norm(actual[rows,:]-expected[rows,:])/norm(expected[rows,:]) < 3e-6
                        end
                        if length(gpu_ids) > 1
                            CUDA.device!(last(gpu_ids))
                            try
                                @test_throws ArgumentError extension.PrunedFFTTransform(plan,Int32(1))*buffer
                                close(plan)
                                @test CUDA.deviceid(CUDA.device()) == last(gpu_ids)
                            finally
                                CUDA.device!(first(gpu_ids))
                            end
                        end
                    finally
                        close(plan)
                    end
                    close(plan)
                    @test plan.handle == C_NULL
                    @test_throws ErrorException extension.PrunedFFTTransform(plan,Int32(1))*buffer
                end
            end
            cases = (((9,11,6),[first(gpu_ids)],2,0),
                ((9,11,6),[first(gpu_ids)],4,1<<30),
                ((17,19,22),[first(gpu_ids)],8,1<<30),
                ((9,11,6),gpu_ids,8,0),
                ((18,16,14),gpu_ids,4,1<<30),
                ((18,16,14),gpu_ids,2,1<<30))
            for (dims,devices,batch,budget) in cases
                @testset "$dims / $(length(devices)) GPUs / batch $batch / cache $budget" begin
                    CUDA.device!(first(devices))
                    rng = MersenneTwister(91237)
                    ns,nd,nc = 67,3,5
                    grid = Grid(dims...,.001f0,.001f0,.001f0)
                    mask = trues(dims);mask[1:13:end] .= false
                    fieldmap = reshape(collect(range(-90f0,90f0;length=prod(dims))),dims)
                    csm = randn(rng,ComplexF32,dims...,nc);csm[:,:,:,nc] .= 0
                    k = .01f0 .* randn(rng,Float32,16,ns,nd);k[2:4,:,:] .*= 60000f0
                    times = reshape(collect(range(.01f0,.014f0;length=ns*nd)),ns,nd)
                    concomitant = 1000f0 .* randn(rng,Float32,4,ns,nd)
                    selected = merge(options,(coefficient_cache_bytes=budget,))
                    op = HighOrderLowRankOp(grid,k,times;fieldmap,csm,mask,concomitant,
                        arrayType=CuArray,gpus=devices,L_rank=6,rsvd_oversample=3,
                        rsvd_distribution=:single,rsvd_finalize=:gram,rsvd_seed=13,
                        normal_method=:batched,normal_coil_batch_size=batch,normal_options=selected,
                        normal_distribution=length(devices)==1 ? :single : :channel)
                    try
                        x = CuArray(randn(rng,ComplexF32,prod(dims)))
                        signal = CuArray(randn(rng,ComplexF32,ns*nd*nc))
                        w = ComplexF32.(rand(rng,0:16,ns*nd)./16,rand(rng,0:16,ns*nd)./16)
                        w[1:7:end] .= 0
                        D = abs2.(w)
                        forward = op*x
                        reference_adjoint = op'*signal
                        reference = op'*(forward.*CuArray(repeat(D,nc)))
                        N = normalOperator(op,CuArray(D))
                        actual = N*x
                        @test norm(actual-reference)/norm(reference) < 3f-5
                        @test norm(op'*signal-reference_adjoint)/norm(reference_adjoint) < 3f-5
                        W = WeightingOp(ComplexF32;weights=CuArray(w),rep=nc)
                        @test norm(normalOperator(∘(W,op))*x-reference)/norm(reference) < 3f-5
                        @test abs(dot(forward,signal)-dot(x,op'*signal))/max(abs(dot(forward,signal)),abs(dot(x,op'*signal))) < 3f-5
                        @test all(iszero,Array(actual)[.!vec(mask)])
                        backend = op.normal_backend.operator
                        executions = [s.batched_workspace.execution for s in backend.state.shards]
                        @test all(e -> (e.coefficient_cache_bytes>0)==(budget>0),executions)
                        masked = Array(x)[findall(vec(mask))]
                        host_forward = reshape(Array(forward),ns*nd,nc)
                        errors = Vector{Float64}(undef,length(backend.state.workers))
                        HighOrderMRI.run_on_workers!(errors,backend.state.workers) do i
                            shard = backend.state.shards[i]
                            out = CUDA.zeros(ComplexF32,ns*nd,length(shard.channels))
                            HighOrderMRI.highorder_batched_forward!(out,shard.batched_workspace,CuArray(masked))
                            expected = host_forward[:,shard.channels]
                            norm(Array(out)-expected)/max(norm(expected),eps(Float32))
                        end
                        @test maximum(errors) < 3e-5
                        close(op)
                        @test backend.state.released
                        @test all(e -> e.released && e.fft.handle==C_NULL,executions)
                        @test norm(op*x-forward)/norm(forward) < 3f-5
                        @test norm(normalOperator(op,CuArray(D))*x-actual)/norm(actual) < 3f-5
                        close(op)
                        @test iszero(norm(normalOperator(op,CUDA.zeros(Float32,ns*nd))*x))
                    finally
                        close(op)
                    end
                end
            end
            @testset "failed native setup releases resources and preserves parent" begin
                CUDA.device!(first(gpu_ids))
                dims = (9,11,6)
                rng = MersenneTwister(9171)
                grid = Grid(dims...,.001f0,.001f0,.001f0)
                k = .01f0 .* randn(rng,Float32,16,67,3)
                times = reshape(collect(range(.01f0,.014f0;length=67*3)),67,3)
                csm = randn(rng,ComplexF32,dims...,5)
                op = HighOrderLowRankOp(grid,k,times;csm,arrayType=CuArray,gpus=gpu_ids,
                    L_rank=6,rsvd_oversample=3,rsvd_distribution=:single,rsvd_finalize=:gram,rsvd_seed=13,
                    normal_method=:batched,normal_coil_batch_size=8,normal_options=options,
                    normal_distribution=:channel)
                try
                    x = CuArray(randn(rng,ComplexF32,prod(dims)))
                    forward = copy(op*x)
                    D = CUDA.ones(Float32,67*3)
                    expected = copy(op'*forward)
                    mktemp() do path,io
                        write(io,"invalid PTX for setup cleanup test\n")
                        close(io)
                        op.normal_backend.options = merge(options,(native_ptx=path,))
                        @test_throws HighOrderMRI.DistributedGPUWorkerError normalOperator(op,D)
                        @test op.normal_backend.operator === nothing
                    end
                    GC.gc(true)
                    @test !any(p -> occursin(basename(options.vkfft_library),p),Libdl.dllist())
                    @test norm(op*x-forward)/norm(forward) < 3f-5
                    op.normal_backend.options = options
                    @test norm(normalOperator(op,D)*x-expected)/norm(expected) < 3f-5
                finally
                    close(op)
                end
                GC.gc(true)
                @test !any(p -> occursin(basename(options.vkfft_library),p),Libdl.dllist())
            end
        finally
            AbstractNFFTs.set_active_backend!(previous)
            CUDA.device!(first(gpu_ids))
        end
    end
end
