@testset "Geometry construction" begin
    geo = Geometry(
        "SIEMENS",
        3,
        [0.2, 0.1, 0.05],
        [100, 50, 25],
        "HFS",
        [0.01, -0.008, -0.015],
        Matrix{Float64}(I, 3, 3),
        1,
    )

    @test geo.SystemVendor == "SIEMENS"
    @test geo.Dimension == 3
    @test geo.MatrixSize == [100, 50, 25]
    @test geo.Idx_Slice == 1
    @test size(geo.R_RPS_DCS) == (3, 3)
end
