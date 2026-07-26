"""
Dimension-independent utilities for HighOrderMRI reconstruction benchmarks.

Run scripts own dataset loading, image layout, operator configuration, and
reporting fields.  This file deliberately contains no 2D/3D assumptions.
Include it after importing `CUDA`, `LinearAlgebra`, and `HighOrderMRI`.
"""

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

"""Least-squares complex scale mapping `x` onto `reference`."""
function alignment_scale(x, reference)
    values = vec(Array(x))
    target = vec(Array(reference))
    energy = max(real(dot(values, values)), eps(real(eltype(values))))
    return dot(values, target) / energy
end

function aligned_relative_error(x, reference; scale=alignment_scale(x, reference))
    values = vec(Array(x))
    target = vec(Array(reference))
    return norm(scale .* values .- target) / max(norm(target), eps(real(eltype(target))))
end

function magnitude_nrmse(x, reference; scale=alignment_scale(x, reference))
    values = abs.(scale .* vec(Array(x)))
    target = abs.(vec(Array(reference)))
    return norm(values .- target) / max(norm(target), eps(eltype(target)))
end

"""Restore the CUDA device owning persistent arrays before applying `op`."""
function activate_operator_device!(op)
    storage = if op isa HighOrderLowRankOp
        op.q
    elseif op isa HighOrderOp_Kernel
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
              op isa HighOrderOp_Kernel ? op.Mv : nothing
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
