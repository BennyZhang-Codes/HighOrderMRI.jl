@testset "Real entry points reject unverified recovery before GPU/data work" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    for script in ("sweep_2d_coil_compression.jl", "sweep_3d_lowrank_parameters.jl",
        "sweep_3d_coil_compression.jl")
        mktempdir() do directory
            input = joinpath(directory, "input.mat")
            write(input, "not a MAT file; the contract must reject before reading it")
            output = joinpath(directory, "run")
            mkpath(output)
            sentinel = joinpath(output, "configuration.csv")
            write(sentinel, "previous configuration\n")
            environment = copy(ENV)
            merge!(environment, Dict(
                "CUDA_VISIBLE_DEVICES" => "", "MPLBACKEND" => "Agg",
                "HIGHORDER_2D_CC_DATA" => input, "HIGHORDER_2D_CC_RUN_DIR" => output,
                "HIGHORDER_3D_DATA" => input, "HIGHORDER_3D_REFERENCE" => input,
                "HIGHORDER_3D_RUN_DIR" => output,
            ))
            command = `$(Base.julia_cmd()) --startup-file=no --history-file=no --project=$(joinpath(root, "benchmark")) $(joinpath(root, "benchmark", "run", script))`
            buffer = IOBuffer()
            process = run(pipeline(ignorestatus(setenv(command, environment)); stdout=buffer, stderr=buffer))
            message = String(take!(buffer))
            @test process.exitcode != 0
            @test occursin("Existing results have no verified run contract", message)
            @test read(sentinel, String) == "previous configuration\n"
            @test readdir(output) == ["configuration.csv"]
            @test read(input, String) == "not a MAT file; the contract must reject before reading it"
        end
    end
end
