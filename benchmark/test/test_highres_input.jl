include(joinpath(@__DIR__, "..", "common", "highres_input.jl"))

@testset "High-resolution input adapter" begin
    mktempdir() do directory
        source = joinpath(directory, "source.mat")
        output = joinpath(directory, "prepared.mat")
        raw = Dict{String,Any}(
            "matrixSize" => [400, 400, 1], "FOV" => [0.2, 0.2, 0.005],
            "kdata" => fill(1f0 + 2f0im, 8, 32),
            "datatime" => collect(1:8) .* 1e-5,
            "kspha" => zeros(10, 16), "k0_adc" => zeros(1, 8),
            "gre_csm" => fill(1f0 + 2f0im, 200, 200, 32),
            "gre_b0" => fill(12.5f0, 200, 200), "gre_mask" => trues(200, 200),
            "dt" => 2e-6, "start" => 0.0, "tau" => 0.0, "dt_adc" => 1e-5,
        )
        matwrite(source, raw)
        source_hash = open(sha256, source)
        @test prepare_highres_input(source, output) == output
        prepared = matread(output)
        @test size(prepared["gre_csm"]) == (400, 400, 32)
        @test eltype(prepared["gre_csm"]) == ComplexF32
        @test all(value -> value ≈ 1f0 + 2f0im, prepared["gre_csm"])
        @test size(prepared["gre_b0"]) == (400, 400)
        @test all(==(12.5f0), prepared["gre_b0"])
        @test all(prepared["gre_mask"])
        @test prepared["kdata"] == raw["kdata"]
        @test vec(prepared["datatime"]) == raw["datatime"]
        @test prepared["ksphaMeasured"] == raw["kspha"]
        @test open(sha256, source) == source_hash
        @test_throws ArgumentError prepare_highres_input(source, source)
        @test open(sha256, source) == source_hash
        raw["datatime"] = zeros(8)
        matwrite(source, raw)
        invalid = joinpath(directory, "invalid.mat")
        @test_throws ErrorException prepare_highres_input(source, invalid)
        @test !isfile(invalid)
    end
end
