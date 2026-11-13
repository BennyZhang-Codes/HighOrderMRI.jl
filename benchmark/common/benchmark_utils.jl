"""
Dimension-independent utilities for HighOrderMRI reconstruction benchmarks.

Run scripts own dataset loading, image layout, operator configuration, and
reporting fields.  This file deliberately contains no 2D/3D assumptions.
GPU utilities require `CUDA`, `LinearAlgebra`, and `HighOrderMRI` in the caller.
"""

using SHA
using TOML

parse_env_int(name::AbstractString, default::Integer) = parse(Int, get(ENV, name, string(default)))

function parse_env_gpus(name::AbstractString, default::AbstractVector{<:Integer})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int.(default)
    return parse.(Int, strip.(split(raw, ',')))
end

function parse_env_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, string(default))))
    raw in ("true", "1", "yes", "on") && return true
    raw in ("false", "0", "no", "off") && return false
    throw(ArgumentError("$name must be true/false, 1/0, yes/no, or on/off; got $raw"))
end

function synchronize_gpus!(gpu_ids::AbstractVector{<:Integer})
    isempty(gpu_ids) && throw(ArgumentError("gpu_ids must contain at least one device"))
    for gpu_id in gpu_ids
        CUDA.device!(gpu_id)
        CUDA.device_synchronize(; blocking=true)
    end
    CUDA.device!(first(gpu_ids))
    return nothing
end

"""Measure wall time while including all GPU work submitted by `f`."""
function elapsed_seconds(f::F, gpu_ids::AbstractVector{<:Integer}) where {F}
    synchronize_gpus!(gpu_ids)
    t0 = time_ns()
    value = f()
    synchronize_gpus!(gpu_ids)
    return value, (time_ns() - t0) * 1e-9
end

"""Release only the cached LowRank normal backend; never close an operator generically."""
function release_benchmark_backend!(op)
    op isa HighOrderLowRankOp && release_highorder_normal_backend!(op)
    return nothing
end

"""Fail at the reconstruction boundary rather than in a later metric."""
function require_reconstruction(x, method::AbstractString)
    x === nothing && error("$method returned `nothing`; execute the complete benchmark entry point.")
    return x
end

function gpu_free_memory_mib(gpu_ids::AbstractVector{<:Integer})
    snapshot = String[]
    for gpu_id in gpu_ids
        CUDA.device!(gpu_id)
        push!(snapshot, "$(gpu_id):$(round(CUDA.free_memory() / 2.0^20; digits=1))")
    end
    return join(snapshot, ";")
end

"""Restore the CUDA device owning persistent arrays before applying `op`."""
function activate_operator_device!(op)
    storage = if op isa HighOrderLowRankOp
        op.q
    elseif op isa HighOrderKernelOp
        op.Mv
    else
        nothing
    end
    storage isa CuArray && CUDA.device!(CUDA.deviceid(CUDA.device(storage)))
    return nothing
end

function weighted_residual(op, x, data, weight)
    activate_operator_device!(op)

    # This is a diagnostic outside the timed region.  Operators may return a
    # host vector (Kernel) or a CuArray (LowRank), while benchmark data lives
    # on the primary GPU.  Keep the operator application storage-consistent,
    # then perform the scalar residual calculation entirely on the host.
    x_input = vec(x)
    storage = op isa HighOrderLowRankOp ? op.q :
              op isa HighOrderKernelOp ? op.Mv : nothing
    if storage isa CuArray && !(x_input isa CuArray)
        x_input = CuArray(x_input)
    end

    predicted = Array(op * x_input)
    measured = Array(vec(data))
    sample_weight = Array(vec(weight))
    ncha = div(length(measured), length(sample_weight))
    weights = repeat(sample_weight, ncha)
    numerator = norm((predicted .- measured) .* weights)
    denominator = max(norm(measured .* weights), eps(Float64))
    return numerator / denominator
end

function csv_field(value)
    text = replace(string(value), '"' => "\"\"")
    return occursin(',', text) || occursin('"', text) ? "\"$(text)\"" : text
end

function write_csv(path::AbstractString, rows::AbstractVector{<:NamedTuple})
    isempty(rows) && return nothing
    headers = collect(keys(first(rows)))
    open(path, "w") do io
        println(io, join(string.(headers), ','))
        for row in rows
            println(io, join((csv_field(getproperty(row, header)) for header in headers), ','))
        end
    end
    return nothing
end

"""Parse one CSV record written by `write_csv`, including quoted commas."""
function parse_benchmark_csv_record(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    index = firstindex(line)
    final_index = lastindex(line)

    while index <= final_index
        character = line[index]
        next_index = nextind(line, index)
        if character == '"'
            if quoted && next_index <= final_index && line[next_index] == '"'
                write(buffer, '"')
                index = nextind(line, next_index)
                continue
            end
            quoted = !quoted
        elseif character == ',' && !quoted
            push!(fields, String(take!(buffer)))
        else
            write(buffer, character)
        end
        index = next_index
    end
    quoted && throw(ArgumentError("Unterminated quoted CSV field: $line"))
    push!(fields, String(take!(buffer)))
    return fields
end

function parse_benchmark_csv_value(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && return missing
    lowercase_value = lowercase(stripped)
    lowercase_value == "missing" && return missing
    lowercase_value == "true" && return true
    lowercase_value == "false" && return false
    integer = tryparse(Int, stripped)
    isnothing(integer) || return integer
    floating = tryparse(Float64, stripped)
    isnothing(floating) || return floating
    return stripped
end

"""Read a CSV produced by `write_csv` without requiring the external CSV.jl package."""
function read_benchmark_csv(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSV file was not found: $path"))
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]
    headers = Symbol.(parse_benchmark_csv_record(first(lines)))
    rows = NamedTuple[]
    row_type = NamedTuple{Tuple(headers)}

    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = parse_benchmark_csv_record(line)
        length(fields) == length(headers) || throw(ArgumentError(
            "CSV row has $(length(fields)) fields; expected $(length(headers)): $line",
        ))
        values = Tuple(parse_benchmark_csv_value.(fields))
        push!(rows, row_type(values))
    end
    return rows
end


"""Fingerprint inputs and the actual local source before creating run artifacts."""
function benchmark_run_contract(inputs::NamedTuple, parameters::NamedTuple;
    source_root::AbstractString, scripts::AbstractVector{<:AbstractString})
    if isdefined(@__MODULE__, :HighOrderMRI)
        loaded_root = dirname(dirname(pathof(HighOrderMRI)))
        realpath(loaded_root) == realpath(source_root) || throw(ArgumentError(
            "HighOrderMRI must be loaded from this benchmark checkout; got $loaded_root",
        ))
    end
    source_files = String[]
    for directory in ("src", joinpath("MRIGeometry", "src"),
        joinpath("benchmark", "common"), joinpath("benchmark", "run"))
        for (root, _, files) in walkdir(joinpath(source_root, directory))
            append!(source_files, [joinpath(root, file) for file in files if endswith(file, ".jl")])
        end
    end
    for file in ("Project.toml", "Manifest.toml", joinpath("MRIGeometry", "Project.toml"),
        joinpath("benchmark", "Project.toml"), joinpath("benchmark", "Manifest.toml"))
        path = joinpath(source_root, file)
        isfile(path) && push!(source_files, path)
    end
    append!(source_files, scripts)
    push!(source_files, @__FILE__)
    fingerprint(path) = open(path, "r") do io
        Dict("path" => realpath(path), "sha256" => bytes2hex(sha256(io)))
    end
    contract = Dict(
        "schema_version" => 1,
        "inputs" => Dict(string(name) => fingerprint(path) for (name, path) in pairs(inputs)),
        "parameters" => Dict(string(name) => value for (name, value) in pairs(parameters)),
        "sources" => [fingerprint(path) for path in sort!(unique(source_files))],
        "julia_version" => string(VERSION),
    )
    # Compare the serialized representation, including Float32 parameters,
    # with exactly the types that TOML will restore on a subsequent process.
    return TOML.parse(sprint(TOML.print, contract))
end

"""Sum named timing stages, retaining missing/failed measurements."""
function benchmark_stage_total(stages::Real...)
    all(value -> isfinite(value) && value >= 0, stages) ||
        throw(ArgumentError("Timing stages must be finite and nonnegative"))
    return sum(stages)
end

benchmark_stage_total(stages::Union{Real,Missing}...) = missing

function validate_benchmark_run!(run_dir::AbstractString, contract::AbstractDict)
    mkpath(run_dir)
    path = joinpath(run_dir, "run_contract.toml")
    if isfile(path)
        isequal(TOML.parsefile(path), contract) || throw(ArgumentError(
            "Run inputs, source, or configuration changed; use a new result directory: $run_dir",
        ))
    else
        isempty(readdir(run_dir)) || throw(ArgumentError(
            "Existing results have no verified run contract; use a new result directory: $run_dir",
        ))
        temporary = path * ".tmp"
        open(temporary, "w") do io
            TOML.print(io, contract; sorted=true)
        end
        mv(temporary, path)
    end
    return nothing
end

"""Raw magnitude difference; never fit an amplitude or phase alignment."""
function benchmark_magnitude_difference(x, reference; scale::Real=1)
    size(x) == size(reference) || throw(DimensionMismatch("Image sizes must match"))
    isfinite(scale) && scale > 0 || throw(ArgumentError("Difference scale must be finite and positive"))
    return scale .* abs.(abs.(x) .- abs.(reference))
end

"""Find the newest matching run with all required output files present."""
function newest_benchmark_directory(root::AbstractString, prefix::AbstractString,
    required_files; name_filter::AbstractString="", accept=(_ -> true))
    isdir(root) || throw(ArgumentError("Results root does not exist: $root"))
    candidates = filter(readdir(root; join=true)) do path
        isdir(path) && startswith(basename(path), prefix) &&
            occursin(name_filter, basename(path)) &&
            all(file -> isfile(joinpath(path, file)), required_files) && accept(path)
    end
    isempty(candidates) && throw(ArgumentError("No completed $prefix run found under $root"))
    return candidates[argmax(mtime.(candidates))]
end
