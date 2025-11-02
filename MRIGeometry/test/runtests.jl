using Test
using MRIGeometry

@testset "Geometry struct and transformations" begin
    geo = Geometry(
        3,                              # Dimension
        [0.2, 0.1, 0.05],               # FOV [m]
        [100,  50,   25],               # MatrixSize
        [0.9781341925, -0.2079088408, 0.005235963831], # Normal
        [10.5377449523, -7.95697284367, -14.5749687108]*1e-3, # Position [m]
        0.0,                            # InPlaneRot [rad]
        "HFS",                          # PatientPosition
        zeros(3, 3)        # RotMatrix
    )
    # Basic structure tests
    @test geo.Dimension == 3
    @test length(geo.FOV) == 3
    @test length(geo.MatrixSize) == 3
    @test typeof(geo.RotMatrix) == Matrix{Float64}
    @test size(geo.RotMatrix) == (3, 3)
    # Check spatial transformation matrices
    rot_plane = get_plane_orientation(geo)
    rot_inplane = get_inplane_rotation(geo)
    rot_prs_to_pcs = prs_to_pcs(geo)
    rot_pcs_to_xyz = pcs_to_xyz(geo)
    rot_prs_to_xyz = prs_to_xyz(geo)
    rot_rps_to_xyz = rps_to_xyz(geo)

    @test size(rot_plane) == (3, 3)
    @test size(rot_inplane) == (3, 3)
    @test size(rot_prs_to_pcs) == (3, 3)
    @test size(rot_pcs_to_xyz) == (3, 3)
    @test size(rot_prs_to_xyz) == (3, 3)
    @test size(rot_rps_to_xyz) == (3, 3)

    # Test spatial point transformation from RPS to XYZ
    input_pts = randn(5, 3)  # 5 points in RPS space
    output_pts = rps_point_to_xyz(geo, input_pts)

    @test size(output_pts) == size(input_pts)
    @test eltype(output_pts) == Float64

end