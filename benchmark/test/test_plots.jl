module CoilPlots2D
include(joinpath(@__DIR__, "..", "analysis", "plot_2d_coil_compression.jl"))
end

module CoilPlots3D
include(joinpath(@__DIR__, "..", "analysis", "plot_3d_coil_compression.jl"))
end

@testset "Coil summary CSV and offline figures" begin
    mktempdir() do directory
        path = joinpath(directory, "summary.csv")
        rows = [(status="ok", operator=operator, virtual_coils=coils,
            raw_complex_nrmse_mean=0.02, raw_complex_nrmse_std=0.001,
            magnitude_nrmse_mean=0.01, magnitude_nrmse_std=0.001,
            magnitude_ssim_mean=0.98, magnitude_ssim_std=0.001,
            setup_median_s=1.0, recon_median_s=4.0, operator_total_median_s=5.0,
            end_to_end_median_s=6.0, cg_speedup_vs_32=1.2,
            end_to_end_speedup_vs_32=1.1, retained_energy=0.99,
            note="quoted, metadata \"preserved\"")
            for operator in ("HighOrderKernelOp", "HighOrderLowRankOp") for coils in (4, 32)]
        write_csv(path, rows)
        restored = CoilPlots2D.finite_summary_rows(path)
        @test length(restored) == 4
        @test all(row -> row.note == "quoted, metadata \"preserved\"", restored)
        paths = CoilPlots2D.plot_2d_coil_compression!(directory)
        @test length(paths) == 2
        @test all(file -> isfile(file) && filesize(file) > 0, paths)
        @test occursin("Fresh setup", read(only(filter(file -> endswith(file, ".svg"), paths)), String))
        write_csv(path, [(status="ok", virtual_coils=4)])
        @test_throws ArgumentError CoilPlots2D.finite_summary_rows(path)
    end
    mktempdir() do directory
        path = joinpath(directory, "coil_compression_comparison.csv")
        rows = [(status="ok", virtual_coils=coils, magnitude_nrmse_mean=0.01,
            magnitude_nrmse_std=0.001, magnitude_ssim_mean=0.98, magnitude_ssim_std=0.001,
            total_median_s=4.0, total_std_s=0.01, retained_energy=0.99,
            note="quoted, metadata") for coils in (4, 32)]
        write_csv(path, rows)
        @test length(CoilPlots3D.finite_3d_coil_summary_rows(path)) == 2
        paths = CoilPlots3D.plot_3d_coil_compression!(directory)
        @test length(paths) == 2
        @test all(file -> isfile(file) && filesize(file) > 0, paths)
        @test occursin("Reused basis", read(only(filter(file -> endswith(file, ".svg"), paths)), String))
    end
end
