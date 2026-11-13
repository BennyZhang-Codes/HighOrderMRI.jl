using Test, HighOrderMRI, LinearAlgebra
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
