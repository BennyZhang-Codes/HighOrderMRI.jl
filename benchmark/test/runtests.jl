using Test
using SHA
using TOML
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "common", "benchmark_utils.jl"))

@testset "CSV checkpoint records" begin
    @test parse_benchmark_csv_record("a,\"2,4,6\",\"quoted \"\"text\"\"\",true,") ==
        ["a", "2,4,6", "quoted \"text\"", "true", ""]
    @test parse_benchmark_csv_record(csv_field("中文,带引号\"路径")) == ["中文,带引号\"路径"]
    @test_throws ArgumentError parse_benchmark_csv_record("a,\"unfinished")
    @test parse_benchmark_csv_value("false") === false
    @test parse_benchmark_csv_value("1234") === 1234
    @test parse_benchmark_csv_value("1e-2") == 0.01
    mktempdir() do directory
        path = joinpath(directory, "checkpoint.csv")
        rows = [(status="ok", gpu_ids="2,4,5,6", error="comma, and \"quotes\"",
            flag=true, rank=16, value=0.003, absent=missing)]
        write_csv(path, rows)
        @test isequal(read_benchmark_csv(path), rows)
        write(path, "a,b\n1,2,3\n")
        @test_throws ArgumentError read_benchmark_csv(path)
        write(path, "")
        @test isempty(read_benchmark_csv(path))
    end
end

@testset "Run provenance and recovery" begin
    mktempdir() do directory
        source = joinpath(directory, "source")
        for subdirectory in ("src", "MRIGeometry/src", "benchmark/common", "benchmark/run")
            mkpath(joinpath(source, subdirectory))
        end
        kernel = joinpath(source, "src", "kernel.jl")
        script = joinpath(source, "benchmark", "run", "run.jl")
        write(kernel, "original source\n")
        write(script, "original driver\n")
        input = joinpath(directory, "input.mat")
        write(input, "original input\n")
        parameters = (iterations=20, regularization=1f-9, gpus=[2, 4, 5, 6],
            backend="NFFT", rank=25, repeats=10, seed=1234)
        contract(params=parameters) = benchmark_run_contract((data=input,), params;
            source_root=source, scripts=[script])
        expected = contract()
        output = joinpath(directory, "run")
        @test isnothing(validate_benchmark_run!(output, expected))
        @test isnothing(validate_benchmark_run!(output, contract()))
        reference = joinpath(output, "reference_32ch.jls")
        write(reference, "frozen reference\n")
        contract_bytes = read(joinpath(output, "run_contract.toml"))
        @test isnothing(validate_benchmark_run!(output, contract()))
        for update in ((iterations=60,), (regularization=2f-9,), (gpus=[2, 4],),
            (backend="another backend",), (rank=15,), (repeats=5,), (seed=1235,))
            @test_throws ArgumentError validate_benchmark_run!(output, contract(merge(parameters, update)))
            @test read(reference, String) == "frozen reference\n"
            @test read(joinpath(output, "run_contract.toml")) == contract_bytes
        end
        write(input, "different input\n")
        @test_throws ArgumentError validate_benchmark_run!(output, contract())
        write(input, "original input\n")
        write(kernel, "changed source\n")
        @test_throws ArgumentError validate_benchmark_run!(output, contract())
        write(kernel, "original source\n")
        write(script, "changed driver\n")
        @test_throws ArgumentError validate_benchmark_run!(output, contract())
        write(script, "original driver\n")
        @test isnothing(validate_benchmark_run!(output, contract()))
        legacy = joinpath(directory, "legacy")
        mkpath(legacy)
        write(joinpath(legacy, "runs_checkpoint.csv"), "old results\n")
        @test_throws ArgumentError validate_benchmark_run!(legacy, expected)
        @test !isfile(joinpath(legacy, "run_contract.toml"))
        @test read(joinpath(legacy, "runs_checkpoint.csv"), String) == "old results\n"
    end
end

@testset "Raw image differences" begin
    reference = ComplexF32[1 2; 3 4]
    half_amplitude = reference ./ 2
    @test norm(half_amplitude - reference) / norm(reference) ≈ 0.5
    @test benchmark_magnitude_difference(half_amplitude, reference) ≈ abs.(reference) ./ 2
    @test benchmark_magnitude_difference(half_amplitude, reference; scale=30) ≈ 15 .* abs.(reference)
    @test iszero(norm(benchmark_magnitude_difference(im .* reference, reference)))
    @test_throws DimensionMismatch benchmark_magnitude_difference(vec(reference), reference)
    @test_throws ArgumentError benchmark_magnitude_difference(reference, reference; scale=0)
    @test_throws ArgumentError benchmark_magnitude_difference(reference, reference; scale=Inf)
end

@testset "Timing stages and failed measurements" begin
    @test benchmark_stage_total(2.0, 3.0, 5.0, 11.0) == 21.0
    reused = benchmark_stage_total(0.5, 11.0)
    @test reused == 11.5
    @test benchmark_stage_total(7.0, 5.0, reused) == 23.5
    @test benchmark_stage_total(7.0, missing) === missing
    @test_throws ArgumentError benchmark_stage_total(-1.0, 2.0)
    @test_throws ArgumentError benchmark_stage_total(NaN, 2.0)
end

@testset "Completed result directory selection" begin
    mktempdir() do directory
        complete = joinpath(directory, "2d_local_lowrank_rank_sweep_spiral_0p5_complete")
        partial = joinpath(directory, "2d_local_lowrank_rank_sweep_spiral_0p5_partial")
        other = joinpath(directory, "2d_local_lowrank_rank_sweep_spiral_1p0_complete")
        for path in (complete, partial, other)
            mkpath(path)
            write(joinpath(path, "runs.csv"), "results\n")
        end
        for path in (complete, other)
            write(joinpath(path, "summary_by_rank.csv"), "summary\n")
        end
        required = ("runs.csv", "summary_by_rank.csv")
        @test newest_benchmark_directory(directory, "2d_local_lowrank_rank_sweep_", required;
            name_filter="_0p5_") == complete
        @test_throws ArgumentError newest_benchmark_directory(directory, "missing_", required)
        @test_throws ArgumentError newest_benchmark_directory(directory, "2d_local_lowrank_rank_sweep_",
            required; accept=(_ -> false))
    end
end

@testset "Benchmark dependencies and syntax" begin
    root = normpath(joinpath(@__DIR__, ".."))
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    @test project["sources"]["HighOrderMRI"]["path"] == ".."
    for name in ("MAT", "CSV", "NIfTI", "NFFT", "CUDA", "HighOrderMRI")
        @test haskey(project["deps"], name)
        @test !isnothing(Base.find_package(name))
    end
    for subdirectory in ("run", "analysis", "common")
        for file in readdir(joinpath(root, subdirectory); join=true)
            endswith(file, ".jl") || continue
            ast = Meta.parseall(read(file, String))
            @test !any(node -> node isa Expr && node.head in (:error, :incomplete), ast.args)
        end
    end
end

include("test_highres_input.jl")
include("test_entrypoints.jl")
include("test_coil_inputs.jl")
include("test_plots.jl")
