using Test, CUDA, HighOrderMRI, LinearAlgebra, Random
using LinearOperatorCollection

const EXPLICIT_PHASE_CACHE_RESULTS = NamedTuple[]

@testset "Bounded explicit phase cache" begin
    if CUDA.functional()
        CUDA.device!(0)
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        for T in (Float32,Float64), nTerm in (9,16), nCha in (3,33)
            rng=MersenneTwister(551+nTerm+nCha)
            grid=Grid(4,3,2,T(0.03),T(0.02),T(0.01))
            kspha=randn(rng,T,nTerm,11)
            times=T.(range(0.001,0.03;length=11))
            fieldmap=randn(rng,T,4,3,2).*T(31)
            csm=randn(rng,Complex{T},4,3,2,nCha)
            mask=trues(4,3,2);mask[1]=false;mask[13]=false
            x=randn(rng,Complex{T},24);y=randn(rng,Complex{T},11*nCha)
            w=randn(rng,Complex{T},11*nCha)
            A=explicit_reference_matrix(grid,kspha,times,fieldmap,csm,mask)
            D=abs2.(ComplexF64.(w));expected=A'*(D.*(A*x))
            tolerance=T===Float32 ? 3e-5 : 2e-12
            for phase_backend in (:kernel,:gemm), budget_rows in (0,3,4,8,12)
                budget=sizeof(Complex{T})*count(mask)*budget_rows
                op=HighOrderKernelOp(grid,kspha,times;fieldmap,csm,mask,
                    backend=:blocked,phase_backend,block_size=4,phase_cache_bytes=budget)
                @test sizeof(op.gpu_operator.state.phase_cache)<=budget
                @test size(op.gpu_operator.state.phase_cache,1)==min(11,4*(budget_rows÷4))
                before=Array(op.gpu_operator.state.phase_cache)
                xgpu=CuArray(x);ygpu=CuArray(y)
                forward=Array(op*xgpu);backward=Array(op'*ygpu)
                normal=normalOperator(op,T.(D));output=Array(normal*xgpu)
                errors=(;forward_error=norm(forward-A*x)/norm(A*x),
                    adjoint_error=norm(backward-A'*y)/norm(A'*y),
                    normal_error=norm(output-expected)/norm(expected))
                for error in values(errors);@test error<tolerance;end
                @test norm(Array(op'*CuArray(Complex{T}.(D.*y)))-A'*(D.*y))/norm(A'*(D.*y))<tolerance
                @test norm(Array(normal*(2xgpu))-2output)/norm(output)<2tolerance
                @test iszero(norm(Array(normal*zero(xgpu))))
                @test Array(op.gpu_operator.state.phase_cache)==before
                @test all(iszero,output[.!vec(mask)])
                push!(EXPLICIT_PHASE_CACHE_RESULTS,(;precision=string(T),nTerm,nCha,
                    phase_backend=string(phase_backend),budget_rows,cached_rows=size(before,1),errors...))
                close(op)
            end
            if length(collect(CUDA.devices()))>=2
                budget=sizeof(Complex{T})*count(mask)*4
                op=HighOrderKernelOp(grid,kspha,times;gpus=[0,1],fieldmap,csm,mask,
                    backend=:blocked,phase_backend=:gemm,block_size=4,phase_cache_bytes=budget)
                output=Array(normalOperator(op,T.(D))*CuArray(x))
                @test norm(output-expected)/norm(expected)<tolerance
                @test all(shard->sizeof(shard.op.state.phase_cache)<=budget,op.gpu_operator.state.shards)
                close(op)
            end
        end
        grid=Grid(2,2,1,0.01f0,0.01f0,0.01f0);K=zeros(Float32,9,3);times=zeros(Float32,3)
        @test_throws ArgumentError HighOrderKernelOp(grid,K,times;backend=:blocked,phase_cache_bytes=-1)
        @test_throws ArgumentError HighOrderKernelOp(grid,K,times;backend=:direct,phase_cache_bytes=1)
        @test_throws ArgumentError HighOrderKernelOp(grid,K,times;backend=:cached,phase_cache_bytes=1)
    else
        @test_skip false
    end
end
