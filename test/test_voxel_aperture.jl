using Random, Statistics

@testset "Finite voxel encoding weights" begin
    for T in (Float32, Float64)
        grid = Grid(4, 5, 3, T(0.002), T(0.003), T(0.004))
        k = reshape(T.(range(-100, 120; length=42)), 3, 7, 2)
        saved = copy(k)
        aperture = HighOrderMRI.voxel_aperture(grid, k)
        @test eltype(aperture) === T && size(aperture) == (14,)
        @test aperture ≈ vec(prod(sinc.(k .* reshape(T[0.002,0.003,0.004],3,1,1)); dims=1))
        @test aperture == HighOrderMRI.voxel_aperture(grid, reshape(k, 3, :))
        @test k == saved
        # Independent midpoint integration of a linear phase over one box.
        positions = ((1:256) .- 0.5) ./ 256 .- 0.5
        exact = prod(mean(cis.(2π .* Float64(k[a,1,1]) .* Float64((grid.Δx,grid.Δy,grid.Δz)[a]) .* positions)) for a in 1:3)
        @test aperture[1] ≈ real(exact) rtol=2e-5
        @test HighOrderMRI.voxel_aperture(grid, zeros(T,3,1)) == ones(T,1)
        @test_throws DimensionMismatch HighOrderMRI.voxel_aperture(grid, zeros(T,2,4))
        @test_throws ArgumentError HighOrderMRI.voxel_aperture(grid, fill(T(NaN),3,4))
    end
    function test_encoding_weights(arrayType)
        T = Float32
        rng = Random.Xoshiro(514)
        data = arrayType(randn(rng, ComplexF32, 37, 2))
        weight = arrayType((0.9f0 .+ 0.2f0.*rand(rng,Float32,37)) .* cis.(rand(rng,Float32,37)))
        factor = arrayType((0.8f0 .+ 0.4f0.*rand(rng,Float32,37)) .* cis.(rand(rng,Float32,37)))
        # Use the real package operator so solver allocation, adjoint and normal
        # dispatch are exercised, rather than only testing a matrix identity.
        grid = Grid(3, 3, 1, 1f0, 1f0, 1f0)
        k = randn(rng, Float32, 9, 37) .* 0.01f0
        for j in 1:37
            k[2,j] = Float32(mod(j-1,3)-1)/3
            k[3,j] = Float32(mod((j-1)÷3,3)-1)/3
            k[4,j] = 0
        end
        times = collect(range(0f0, 1f-3; length=37))
        csm = cis.(rand(rng, Float32, 3, 3, 1, 2)) ./ sqrt(2f0)
        op = arrayType === Array ? HighOrderOp(grid, k, times; csm, arrayType) :
            HighOrderKernelOp(grid, k, times; csm, backend=:blocked, gpus=[Int(CUDA.deviceid(CUDA.device()))])
        params = Dict{Symbol,Any}(:solver=>HighOrderMRI.CGNR, :reg=>HighOrderMRI.L2Regularization(0f0),
            :reconSize=>(3,3,1), :iterations=>9, :relTol=>1f-8)
        try
            reference = zeros(ComplexF64, size(op))
            for j in 1:9
                e = zeros(ComplexF32,9);e[j] = 1
                reference[:,j] = Array(op*arrayType(e))
            end
            E = reference .* repeat(ComplexF64.(Array(weight).*Array(factor)),2)
            y = vec(ComplexF64.(Array(data))) .* repeat(ComplexF64.(Array(weight)),2)
            exact = E\y
            actual = recon_HOOp(op, data, weight, params; encoding_weights=factor)
            @test eltype(actual) === ComplexF32
            @test norm(vec(actual)-exact)/norm(exact) < 3e-4
            default = recon_HOOp(op, data, weight, params)
            unchanged = recon_HOOp(op, data, weight, params; encoding_weights=nothing)
            @test default == unchanged
            @test_throws DimensionMismatch recon_HOOp(op, data, weight, params; encoding_weights=factor[1:end-1])
            @test_throws ArgumentError recon_HOOp(op, data, weight, params; encoding_weights=ones(Float64,37))
        finally
            op isa HighOrderKernelOp && close(op)
        end
    end
    test_encoding_weights(Array)
    if CUDA.functional()
        test_encoding_weights(CuArray)
    end
end
