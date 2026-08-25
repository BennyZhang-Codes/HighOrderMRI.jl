@testset "RPS grid and coordinate transforms" begin
    geo = Geometry(
        "SIEMENS",
        3,
        [0.2, 0.1, 0.05],
        [4, 3, 2],
        "HFS",
        [0.01, -0.008, -0.015],
        Matrix{Float64}(I, 3, 3),
        1,
    )

    grid_rps = gen_RPS_grid(geo)
    @test size(grid_rps) == (4, 3, 2, 3)
    @test DCS2RPS(geo, RPS2DCS(geo, grid_rps)) ≈ grid_rps
    @test PCS2RPS(geo, RPS2PCS(geo, grid_rps)) ≈ grid_rps
end
