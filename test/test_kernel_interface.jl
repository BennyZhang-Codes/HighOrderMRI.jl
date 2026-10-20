using Test, CUDA, HighOrderMRI, LinearAlgebra, Random
using LinearOperatorCollection, LinearOperators

@testset "Unified HighOrderKernelOp interface" begin
    @test :HighOrderExplicitGPUOp ∉ names(HighOrderMRI)
    @test :HighOrderExplicitMultiGPUOp ∉ names(HighOrderMRI)
    @test :ExplicitGPUOp ∉ names(HighOrderMRI)
    @test :ExplicitMultiGPUOp ∉ names(HighOrderMRI)
    if CUDA.functional()
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        for T in (Float32, Float64)
            rng = MersenneTwister(731)
            grid = Grid(4, 3, 2, T(0.04), T(0.03), T(0.02))
            kspha = randn(rng, T, 16, 13)
            k_nominal = randn(rng, T, 3, 13)
            times = T.(range(0.001, 0.013; length=13))
            csm = randn(rng, Complex{T}, 4, 3, 2, 3)
            fieldmap = randn(rng, T, 4, 3, 2).*T(25)
            mask = trues(4, 3, 2); mask[1] = false; mask[13] = false
            x = randn(rng, Complex{T}, 24)
            y = randn(rng, Complex{T}, 39)
            weights = randn(rng, Complex{T}, 39)
            D = abs2.(weights)
            # Disable measured zeroth/first-order terms, retaining the nominal
            # first-order trajectory. The oracle never calls prep_kspha.
            effective = copy(kspha)
            effective[1, :] .= 0
            effective[2:4, :] .= k_nominal
            A = explicit_reference_matrix(grid, effective, times, fieldmap, csm, mask)
            kwargs = (; fieldmap, csm, mask, recon_terms="0011", k_nominal)
            CUDA.device!(0)
            legacy = HighOrderKernelOp(grid, kspha, times; kwargs...)
            explicit_legacy = HighOrderKernelOp(grid, kspha, times; kwargs..., backend=:legacy)
            @test legacy isa HighOrderKernelOp
            @test legacy.gpu_operator === nothing
            @test LinearOperators.storage_type(legacy) <: Vector
            @test legacy*x == explicit_legacy*x
            # The historical adjoint launches 512 threads and can exceed the
            # Float64 register limit. Resident Float64 adjoints are tested below.
            if T === Float32
                @test legacy'*y == explicit_legacy'*y
            end
            @test norm(legacy*x - A*x)/norm(A*x) < 3e-5
            close(legacy); close(explicit_legacy)

            choices = length(collect(CUDA.devices())) >= 2 ? ([0], [1], [1,0]) : ([0],)
            for gpus in choices, backend in (:direct, :blocked, :cached_full)
                CUDA.device!(first(gpus))
                op = HighOrderKernelOp(grid, kspha, times; kwargs..., gpus,
                    backend, phase_backend=:gemm, block_size=4)
                @test op isa HighOrderKernelOp
                @test LinearOperators.storage_type(op) <: CuArray
                @test CUDA.deviceid(CUDA.device()) == first(gpus)
                @test (op.gpu_operator isa HighOrderMRI.ExplicitGPUOp) == (length(gpus) == 1)
                @test (op.gpu_operator isa HighOrderMRI.ExplicitMultiGPUOp) == (length(gpus) > 1)
                @test op.prod! === op.gpu_operator.prod!
                @test op.ctprod! === op.gpu_operator.ctprod!
                xgpu, ygpu = CuArray(x), CuArray(y)
                @test Array(op*xgpu) == Array(op.gpu_operator*xgpu)
                @test Array(op'*ygpu) == Array(op.gpu_operator'*ygpu)
                result = Array(normalOperator(op, D)*xgpu)
                @test result == Array(normalOperator(op.gpu_operator, D)*xgpu)
                @test result == Array(explicit_gpu_normal(op; weights)*xgpu)
                tolerance = T === Float32 ? 3e-5 : 2e-12
                @test norm(result - A'*(D.*(A*x)))/norm(A'*(D.*(A*x))) < tolerance
                close(op); close(op)
                @test_throws ArgumentError normalOperator(op)
            end
            CUDA.device!(0)
            @test_throws ArgumentError HighOrderKernelOp(grid, kspha, times; backend=:invalid)
            @test_throws ArgumentError HighOrderKernelOp(grid, kspha, times; backend=:blocked, kspha_dt=kspha)
            @test_throws ArgumentError HighOrderKernelOp(grid, kspha, times; phase_cache_bytes=1)
        end
    else
        @test_skip false
    end
end
