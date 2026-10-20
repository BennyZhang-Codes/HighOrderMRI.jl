using Test, HighOrderMRI, CUDA, LinearAlgebra, Random
using LinearOperatorCollection, LinearOperators

const EXPLICIT_MULTI_GPU_TEST_RESULTS = NamedTuple[]

@testset "Explicit multi-GPU independent reference" begin
    if CUDA.functional() && length(collect(CUDA.devices())) >= 2
        CUDA.allowscalar(false)
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        ids = collect(0:min(4, length(collect(CUDA.devices()))-1))
        CUDA.device!(first(ids))
        for T in (Float32, Float64), (nt, nc) in ((9, 3), (16, 33))
            rng = MersenneTwister(920+nt+nc)
            grid = Grid(4, 3, 2, T(0.04), T(0.03), T(0.02))
            ns = 13
            kspha = randn(rng, T, nt, ns)
            times = T.(range(0.001, 0.015; length=ns))
            fieldmap = T.(reshape(range(-33, 57; length=24), 4, 3, 2))
            csm = randn(rng, Complex{T}, 4, 3, 2, nc)
            mask = trues(4, 3, 2); mask[1] = false; mask[13] = false
            x = randn(rng, Complex{T}, 24)
            y = randn(rng, Complex{T}, ns*nc)
            w = randn(rng, Complex{T}, ns*nc) .+ T(0.7)
            A = explicit_reference_matrix(grid, kspha, times, fieldmap, csm, mask)
            D = abs2.(ComplexF64.(w))
            expected = A'*(D.*(A*x))
            tolerance = T === Float32 ? 3e-5 : 2e-12
            for (backend, phase_backend) in ((:direct, :kernel), (:blocked, :kernel), (:blocked, :gemm), (:cached, :gemm), (:cached_full, :gemm))
                op = HighOrderExplicitMultiGPUOp(grid, kspha, times; gpus=ids,
                    fieldmap, csm, mask, backend, phase_backend, block_size=4)
                xgpu = CuArray(x); ygpu = CuArray(y)
                forward = Array(op*xgpu)
                backward = Array(adjoint(op)*ygpu)
                normal = normalOperator(WeightingOp(CuArray(w), 1) ∘ op)
                output = Array(normal*xgpu)
                rhs = Array(adjoint(op)*CuArray(Complex{T}.(D.*y)))
                errors = (;forward_error=norm(forward-A*x)/norm(A*x),
                    adjoint_error=norm(backward-A'*y)/norm(A'*y),
                    normal_error=norm(output-expected)/norm(expected),
                    rhs_error=norm(rhs-A'*(D.*y))/norm(A'*(D.*y)),
                    adjoint_identity_error=abs(dot(forward,y)-dot(x,backward))/max(norm(forward)*norm(y),1))
                for value in values(errors); @test value < tolerance; end
                @test all(iszero, output[.!vec(mask)])
                @test norm(Array(normal*(2xgpu))-2output)/norm(output) < 2tolerance
                @test iszero(norm(Array(normal*zero(xgpu))))
                @test norm(Array(normal*xgpu)-output)/norm(output) < tolerance
                @test LinearOperators.storage_type(op) <: CuArray
                @test LinearOperators.storage_type(normal) <: CuArray
                raw_normal = explicit_gpu_normal(op; weights=CuArray(w))
                @test norm(Array(raw_normal*xgpu)-output)/norm(output) < tolerance
                @test_throws ArgumentError normalOperator(op, fill(-one(T), ns))
                @test_throws ArgumentError normalOperator(op, fill(Complex{T}(1,1), ns))
                push!(EXPLICIT_MULTI_GPU_TEST_RESULTS, (;precision=string(T), n_term=nt, n_coil=nc,
                    backend=string(backend), phase_backend=string(phase_backend), devices=length(ids), errors...))
                close(op); close(op)
                @test_throws ArgumentError mul!(similar(xgpu),normal,xgpu)
            end
        end
        grid = Grid(2,2,1,0.01f0,0.01f0,0.01f0)
        K = zeros(Float32,9,3); times=zeros(Float32,3)
        @test_throws ArgumentError HighOrderExplicitMultiGPUOp(grid,K,times;gpus=Int[])
        @test_throws ArgumentError HighOrderExplicitMultiGPUOp(grid,K,times;gpus=[0,0])
        @test_throws ArgumentError HighOrderExplicitMultiGPUOp(grid,K,times;gpus=[999])

        # A caller can select another device between construction, normal
        # setup, application and close. The workspace still belongs to GPU 0.
        CUDA.device!(last(ids))
        op = HighOrderExplicitGPUOp(grid, K, times; gpus=[first(ids)], block_size=2)
        @test CUDA.deviceid(CUDA.device()) == first(ids)
        xgpu = CUDA.ones(ComplexF32, 4)
        output = similar(xgpu)
        CUDA.device!(last(ids))
        normal = normalOperator(op)
        @test CUDA.deviceid(CUDA.device()) == first(ids)
        CUDA.device!(last(ids))
        mul!(output, normal, xgpu)
        @test Array(output) ≈ fill(3f0, 4)
        CUDA.device!(last(ids))
        close(op)
        @test CUDA.deviceid(CUDA.device()) == first(ids)

        # Constructor failures must drain jobs and stop their workers, allowing
        # the next construction to proceed normally.
        @test_throws HighOrderMRI.DistributedGPUWorkerError HighOrderExplicitMultiGPUOp(
            grid, K, times; gpus=ids[1:2], backend=:invalid)
        op = HighOrderExplicitMultiGPUOp(grid, K, times; gpus=ids[1:2], block_size=2)
        @test Array(explicit_gpu_normal(op)*xgpu) ≈ fill(3f0, 4)
        close(op)
    else
        @test_skip false
    end
end
