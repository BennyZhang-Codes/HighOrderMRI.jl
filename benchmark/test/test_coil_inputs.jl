module CoilBenchmark2D
include(joinpath(@__DIR__, "..", "run", "sweep_2d_coil_compression.jl"))
end

@testset "Nested coil transforms and timed host inputs" begin
    rng = MersenneTwister(1234)
    data = randn(rng, ComplexF32, 40, 32)
    csm = randn(rng, ComplexF32, 4, 5, 32)
    original_data, original_csm = copy(data), copy(csm)
    dataset = (input_coils=32, data_host=data, csm=csm)
    transform = CoilBenchmark2D.fit_coil_compression(data; coil_dim=2, n_virtual_coils=32)
    full = CoilBenchmark2D.prepare_virtual_coils(dataset, transform, 32)
    @test full.data === data
    @test full.csm === csm
    @test full.compression_apply_s == 0
    compressed = CoilBenchmark2D.prepare_virtual_coils(dataset, transform, 4)
    matrix = transform.compression_matrix[:, 1:4]
    @test size(compressed.data) == (40, 4)
    @test size(compressed.csm) == (4, 5, 4)
    @test eltype(compressed.data) == ComplexF32
    @test eltype(compressed.csm) == ComplexF32
    @test compressed.transform.compression_matrix == matrix
    @test compressed.data ≈ data * matrix
    @test reshape(compressed.csm, :, 4) ≈ reshape(csm, :, 32) * matrix
    weight = ones(ComplexF32, 40)
    inputs = CoilBenchmark2D.prepare_reconstruction_inputs_2d(:kernel, compressed, weight)
    @test inputs.kdata == compressed.data
    @test inputs.weight == weight
    @test isfinite(inputs.input_prepare_s) && inputs.input_prepare_s >= 0
    @test data == original_data
    @test csm == original_csm
end
