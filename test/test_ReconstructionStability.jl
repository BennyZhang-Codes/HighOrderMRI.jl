using Test, HighOrderMRI, LinearAlgebra, Random
using NFFT

@testset "Sampling density normalization" begin
    HighOrderMRI.AbstractNFFTs.set_active_backend!(NFFT.backend())
    for T in (Float32, Float64)
        angle = T.(range(0, 5π; length=37))
        radius = T.(range(0.01, 0.45; length=37))
        nodes = permutedims(hcat(radius.*cos.(angle), radius.*sin.(angle)))
        saved = copy(nodes)
        raw = samplingDensity(nodes, (8,8))
        historical = samplingDensity(nodes, (8,8); normalize=false)
        normalized = samplingDensity(nodes, (8,8); normalize=true)
        @test raw == historical
        @test nodes == saved
        @test eltype(normalized) === Complex{T}
        @test all(isfinite, normalized)
        @test sum(abs2, normalized)/length(normalized) ≈ one(T) rtol=8eps(T)
        @test normalized ≈ raw/(norm(raw)/sqrt(T(length(raw)))) rtol=8eps(T)
        @test all(>(zero(T)), real.(normalized))
        @test all(iszero, imag.(normalized))
    end
end

@testset "Raw sample weights in reconstruction" begin
    function check_raw_sample_weights(arrayType, ::Type{T}) where T<:AbstractFloat
        rng = Random.Xoshiro(514)
        nSam, nCha = 37, 2
        data_host = randn(rng, Complex{T}, nSam, nCha)
        weight_host = (T(0.5) .+ rand(rng, T, nSam)) .* cis.(rand(rng, T, nSam))
        data, weight = arrayType(data_host), arrayType(weight_host)
        grid = Grid(3, 3, 1, one(T), one(T), one(T))
        kspha = randn(rng, T, 9, nSam) .* T(0.01)
        for j in 1:nSam
            kspha[2, j] = T(mod(j - 1, 3) - 1) / T(3)
            kspha[3, j] = T(mod((j - 1) ÷ 3, 3) - 1) / T(3)
            kspha[4, j] = zero(T)
        end
        times = collect(range(zero(T), T(1e-3); length=nSam))
        csm = cis.(rand(rng, T, 3, 3, 1, nCha)) ./ sqrt(T(nCha))
        op = arrayType === Array ? HighOrderOp(grid, kspha, times; csm, arrayType) :
            HighOrderKernelOp(grid, kspha, times; csm, backend=:blocked,
                gpus=[Int(HighOrderMRI.CUDA.deviceid(HighOrderMRI.CUDA.device()))])
        params = Dict{Symbol,Any}(
            :solver => HighOrderMRI.CGNR,
            :reg => HighOrderMRI.L2Regularization(zero(T)),
            :reconSize => grid.matrixSize, :iterations => 30, :relTol => 1e-10)
        try
            # Materialize the unweighted operator and solve independently by QR.
            reference = zeros(ComplexF64, size(op))
            for v in 1:size(op, 2)
                image = zeros(Complex{T}, size(op, 2))
                image[v] = one(T)
                reference[:, v] = Array(op * arrayType(image))
            end
            repeated_weights = repeat(ComplexF64.(weight_host), nCha)
            expected = (reference .* repeated_weights) \ (vec(ComplexF64.(data_host)) .* repeated_weights)
            actual = recon_HOOp(op, data, weight, params)
            tolerance = T === Float32 ? 3e-4 : 1e-10
            @test norm(vec(actual) - expected) / norm(expected) < tolerance
            @test eltype(actual) === Complex{T}
            @test size(actual) == grid.matrixSize
            @test Array(data) == data_host
            @test Array(weight) == weight_host
        finally
            op isa HighOrderKernelOp && close(op)
        end
    end
    check_raw_sample_weights(Array, Float32)
    check_raw_sample_weights(Array, Float64)
    if HighOrderMRI.CUDA.functional()
        check_raw_sample_weights(HighOrderMRI.CUDA.CuArray, Float32)
    else
        @test_skip "CUDA reconstruction weights require a functional device"
    end
end
