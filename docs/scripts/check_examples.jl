using Test, TOML, SHA, LinearAlgebra
using HighOrderMRI

get(ENV, "CUDA_VISIBLE_DEVICES", "unset") == "" || error("Hide GPU devices for CPU documentation checks")
site = get(ENV, "DOCS_SITE_PATH", "")
isdir(joinpath(site, "src")) || error("Set DOCS_SITE_PATH to the website docs directory")
Base.find_package("Documenter") === nothing || error("Run documentation checks in an environment without Documenter")
println("DOC_EXAMPLE_JULIA=", VERSION)
println("DOC_EXAMPLE_CODE=", pathof(HighOrderMRI))
println("DOC_EXAMPLE_THREADS=", Threads.nthreads(), "; BLAS=", BLAS.get_num_threads())
flush(stdout)
results = Dict{String,Any}[]

function run_blocks(page; block_count=1)
    source = read(joinpath(site, "src", page), String)
    blocks = collect(eachmatch(r"(?ms)^```julia\s*\n(.*?)^```\s*$", source))
    isempty(blocks) && error("No runnable Julia example in $page")
    length(blocks) >= block_count || error("Not enough selected examples in $page")
    selected = blocks[1:block_count]
    mod = Module(gensym(:DocumentationExample))
    for (i, block) in enumerate(selected)
        code = block.captures[1]
        occursin("using CUDA", code) && error("GPU code was selected for a CPU check: $page")
        println("RUN_EXAMPLE=", page, " block=", i); flush(stdout)
        Base.include_string(mod, code, "$page (Julia block $i)")
        println("PASS_EXAMPLE=", page, " block=", i); flush(stdout)
        push!(results, Dict("page"=>page, "block"=>i, "passed"=>true,
                           "sha256"=>bytes2hex(sha256(code))))
    end
    return mod
end

if ARGS == ["--plot-only"]
    @testset "Plotting documentation in an isolated CPU process" begin
        @test run_blocks("reference/plotting.md") isa Module
    end
    TOML.print(stdout, Dict("julia_version"=>string(VERSION), "examples"=>results))
    exit(0)
end
isempty(ARGS) || error("Unknown documentation check arguments")

@testset "Getting started: consecutive CPU blocks" begin
    mod = run_blocks("getting-started.md"; block_count=3)
    @test size(mod.reconstructed) == (8, 8)
    @test all(isfinite, mod.reconstructed)
    @test norm(mod.reconstructed) > 0
    @test norm(mod.signal) > 0
end

pages = [
    "reference/grid-basis.md", "reference/highorderop.md",
    "reference/highorderlowrankop.md", "reference/coil-compression-transform.md",
    "reference/estimate-noise-covariance.md", "reference/noise-prewhitening-scale.md",
    "reference/fit-coil-compression.md", "reference/apply-coil-compression.md",
    "reference/compress-coils.md", "reference/sampling-density.md",
    "reference/coil-combine-sos.md", "reference/recon-hoop.md",
    "guide/coil-compression.md", "reference/image-metrics.md",
    "reference/utilities.md",
    "reference/girf-model.md", "reference/apply-girf.md",
    "reference/interp-traj-time.md", "reference/geometry.md",
]
@testset "Complete CPU examples from the website" begin
    for page in pages
        @testset "$page" begin
            mod = run_blocks(page)
            @test mod isa Module
            if page == "reference/highorderlowrankop.md"
                @test size(mod.signal) == (64, 2, 2)
                @test all(isfinite, mod.signal) && all(isfinite, mod.backprojection)
                expected = similar(mod.signal)
                for i in 1:mod.nDyn
                    op = HighOrderOp(mod.grid, mod.kspha[:, :, i], mod.times[:, i];
                        fieldmap=mod.fieldmap, csm=mod.csm, mask=mod.mask,
                        arrayType=Array, nBlock=4)
                    expected[:, i, :] = reshape(op * vec(mod.image), mod.nSam, mod.nCha)
                end
                error = norm(mod.signal - expected) / norm(expected)
                @test error <= 2e-3
                forward = dot(vec(mod.signal), vec(mod.signal))
                adjoint_value = dot(vec(mod.image), vec(mod.backprojection))
                adjoint_error = abs(forward - adjoint_value) / abs(forward)
                @test adjoint_error <= 5e-5
                println("LOWRANK_EXAMPLE_FORWARD_ERROR=", error)
                println("LOWRANK_EXAMPLE_ADJOINT_ERROR=", adjoint_error)
            end
        end
    end
end
# Native plotting libraries own finalizers. Keep their objects out of the process
# that subsequently compiles and executes threaded numerical examples.
@testset "Plotting child process" begin
    project = dirname(Base.active_project())
    command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project $(@__FILE__) --plot-only`
    @test success(pipeline(command; stdout=stdout, stderr=stderr))
end
TOML.print(stdout, Dict("julia_version"=>string(VERSION), "examples"=>results,
    "scope"=>"CPU usage examples only; FindDelay convergence, GPU and physical/performance validation are not covered"))
