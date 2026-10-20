using Test, CUDA, LinearAlgebra, Random
using HighOrderMRI, LinearOperatorCollection, LinearOperators

const EXPLICIT_GPU_TEST_RESULTS = NamedTuple[]

# Independent FP64 polynomials and scalar summation; do not call the production
# basis/phase helper when constructing the numerical oracle.
function explicit_reference_basis(x, y, z)
    r2 = x*x + y*y + z*z
    return (1.0, x, y, z, x*y, z*y, 3z*z-r2, x*z, x*x-y*y,
        3y*x*x-y*y*y, x*y*z, (5z*z-r2)*y, 5z*z*z-3z*r2,
        (5z*z-r2)*x, (x*x-y*y)*z, x*x*x-3x*y*y)
end

function explicit_reference_matrix(grid, kspha, times, fieldmap, csm, mask)
    ns = length(times)
    nc = size(csm, ndims(csm))
    indices = findall(vec(mask))
    matrix = zeros(ComplexF64, ns*nc, length(mask))
    sensitivities = reshape(csm, :, nc)
    for v in indices
        basis = explicit_reference_basis(Float64(grid.x[v]), Float64(grid.y[v]), Float64(grid.z[v]))
        for s in 1:ns
            ϕ = Float64(times[s])*Float64(fieldmap[v])
            for q in axes(kspha, 1)
                ϕ += Float64(kspha[q,s])*basis[q]
            end
            P = cis(2π*ϕ)/sqrt(length(indices))
            for c in 1:nc
                matrix[s+(c-1)*ns, v] = P*ComplexF64(sensitivities[v,c])
            end
        end
    end
    return matrix
end

function test_explicit_gpu_case(::Type{T}, shape, nt, nc) where {T}
    rng = MersenneTwister(1234+nt+nc)
    grid = Grid(shape..., T(0.07), T(0.08), T(0.09))
    ns = 11
    times = T.(range(0.001, 0.009; length=ns))
    kspha = T.(randn(rng, nt, ns))
    fieldmap = T.(reshape(range(-37, 49; length=prod(shape)), shape))
    csm = randn(rng, Complex{T}, shape..., nc)
    mask = trues(shape); mask[1] = false; mask[end-1] = false
    x = randn(rng, Complex{T}, prod(shape))
    y = randn(rng, Complex{T}, ns*nc)
    w = randn(rng, Complex{T}, ns) .+ T(0.7)
    A = explicit_reference_matrix(grid, kspha, times, fieldmap, csm, mask)
    D = repeat(abs2.(ComplexF64.(w)), nc)
    tol = T===Float32 ? 3e-5 : 2e-12
    xgpu, ygpu = CuArray(x), CuArray(y)
    previous = nothing
    for (backend,phase_backend) in ((:direct,:kernel),(:blocked,:kernel),(:blocked,:gemm),(:cached,:kernel),(:cached,:gemm),(:cached_full,:gemm))
        for B in (4, 7)
            op = HighOrderExplicitGPUOp(grid, kspha, times; fieldmap,csm,mask,
                backend,phase_backend,block_size=B)
            forward = Array(op*xgpu)
            backward = Array(adjoint(op)*ygpu)
            @test norm(forward-A*x)/norm(A*x) < tol
            @test norm(backward-A'*y)/norm(A'*y) < tol
            @test abs(dot(forward,y)-dot(x,backward))/max(norm(forward)*norm(y),1) < tol
            @test all(iszero, backward[.!vec(mask)])
            W = WeightingOp(CuArray(w),nc)
            normal = normalOperator(W ∘ op)
            result = Array(normal*xgpu)
            expected = A'*(D.*(A*x))
            @test norm(result-expected)/norm(expected) < tol
            @test norm(Array(explicit_gpu_normal(op;weights=CuArray(w))*xgpu)-expected)/norm(expected) < tol
            @test norm(Array(normalOperator(op,CuArray(T.(D)))*xgpu)-expected)/norm(expected) < tol
            rhs = Array(adjoint(W ∘ op)*(CuArray(repeat(w,nc)).*ygpu))
            @test norm(rhs-A'*(D.*y))/norm(A'*(D.*y)) < tol
            push!(EXPLICIT_GPU_TEST_RESULTS,(;precision=string(T),shape=collect(shape),n_term=nt,n_coil=nc,
                backend=string(backend),phase_backend=string(phase_backend),block_size=B,
                forward_error=norm(forward-A*x)/norm(A*x),adjoint_error=norm(backward-A'*y)/norm(A'*y),
                normal_error=norm(result-expected)/norm(expected),rhs_error=norm(rhs-A'*(D.*y))/norm(A'*(D.*y)),
                adjoint_identity_error=abs(dot(forward,y)-dot(x,backward))/max(norm(forward)*norm(y),1)))
            @test real(dot(x,result)) >= 0
            @test norm(Array(normal*(2xgpu))-2result)/norm(result) < 2tol
            @test norm(Array(normal*zero(xgpu))) == 0
            @test norm(Array(normal*xgpu)-result)/norm(result) < tol
            @test LinearOperators.storage_type(op) <: CuArray
            @test LinearOperators.storage_type(normal) <: CuArray
            @test_throws ArgumentError normalOperator(op,fill(Complex{T}(1,1),ns))
            @test_throws ArgumentError normalOperator(op,fill(-one(T),ns))
            if previous !== nothing
                @test norm(result-previous)/norm(previous) < 2tol
            end
            previous = result
            close(op); close(op)
            @test_throws ArgumentError normalOperator(op)
            @test_throws ArgumentError mul!(similar(xgpu),normal,xgpu)
        end
    end
    return nothing
end

@testset "Explicit GPU independent FP64 reference" begin
    if CUDA.functional()
        CUDA.allowscalar(false)
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        for T in (Float32, Float64)
            test_explicit_gpu_case(T,(5,4,1),9,3)
            test_explicit_gpu_case(T,(4,3,2),16,33)
            test_explicit_gpu_case(T,(4,5,1),16,5)
            test_explicit_gpu_case(T,(3,4,2),9,33)
        end
        grid = Grid(2, 2, 1, 0.01f0, 0.01f0, 0.01f0)
        kspha = zeros(Float32, 9, 3)
        times = zeros(Float32, 3)
        op = HighOrderExplicitGPUOp(grid, kspha, times; gpus=[0], block_size=2)
        @test Array(op*CUDA.ones(ComplexF32, 4)) ≈ fill(2f0, 3)
        close(op)
        @test_throws ArgumentError HighOrderExplicitGPUOp(grid, kspha, times; gpus=Int[])
        @test_throws ArgumentError HighOrderExplicitGPUOp(grid, kspha, times; gpus=[0, 1])
        @test_throws ArgumentError HighOrderExplicitGPUOp(grid, kspha, times; gpus=[999])
    else
        @test_skip false
    end
end
