
struct ExplicitGPUShard{T,O<:ExplicitGPUOp{T}}
    gpu_id::Int
    rows::UnitRange{Int}
    op::O
    input::CuVector{Complex{T}}
    output::CuVector{Complex{T}}
    signal::CuVector{Complex{T}}
    host_image::Vector{Complex{T}}
    host_signal::Matrix{Complex{T}}
end

function Base.close(shard::ExplicitGPUShard)
    shard.op.state.released && return nothing
    close(shard.op)
    CUDA.unsafe_free!(shard.input)
    CUDA.unsafe_free!(shard.output)
    CUDA.unsafe_free!(shard.signal)
    return nothing
end

mutable struct ExplicitMultiGPUState{T,S<:ExplicitGPUShard{T}}
    primary_gpu::Int
    workers::Vector{DistributedGPUWorker}
    shards::Vector{S}
    completed::Vector{Nothing}
    host_input::Vector{Complex{T}}
    host_output::Vector{Complex{T}}
    host_signal::Matrix{Complex{T}}
    released::Bool
end

"""
    ExplicitMultiGPUOp(grid, kspha, times; gpus, kwargs...)

Internal implementation of `HighOrderKernelOp` with multiple GPUs and a
resident backend. Explicit encoding is split over contiguous sample ranges.
The first device owns CG vectors. Each persistent GPU worker retains the full
masked spatial model and all coils, generates its own phase blocks, and reuses
each block in both products of the weighted normal. Global `1/sqrt(nVox)`
normalization is identical to `ExplicitGPUOp`.

Communication uses reusable host buffers for image broadcast and reduction;
phase matrices never leave their owning GPU. This also works without CUDA peer
access. Forward and adjoint preserve sample-by-coil ordering. Model inputs must
be host arrays. Device IDs refer to the process's CUDA-visible devices.

`backend`, `phase_backend`, `gemm_backend`, `phase_cache_bytes`, and kernel/block settings are passed
to `ExplicitGPUOp`. The operator is not reentrant. Call `close` after
all derived normal operators are no longer needed.

All GPUs retain the full spatial model and all coils. Cached backends must fit
their local sample rows, model, phase-generation workspace, and application
workspace on each GPU individually. The sum of available GPU memory is not a
sufficient feasibility check. Set Julia's thread count before startup; one
thread per worker plus a coordinator is recommended for concurrent submission.

For `backend=:blocked`, `phase_cache_bytes` limits an optional complex phase
prefix on each GPU independently. Remaining sample blocks are regenerated
during applications, preserving the full encoding. The default is zero;
include initialization in one-shot reconstruction timing.
"""
mutable struct ExplicitMultiGPUOp{T,F,G,S<:ExplicitMultiGPUState{T}} <: HOOp{Complex{T}}
    nrow::Int
    ncol::Int
    symmetric::Bool
    hermitian::Bool
    prod!::F
    tprod!::Nothing
    ctprod!::G
    nprod::Int
    ntprod::Int
    nctprod::Int
    Mv::CuVector{Complex{T}}
    Mtu::CuVector{Complex{T}}
    state::S
end

LinearOperators.storage_type(op::ExplicitMultiGPUOp) = typeof(op.Mv)

function explicit_multi_check!(state::ExplicitMultiGPUState, out, input)
    state.released && throw(ArgumentError("operator is closed"))
    out isa CuArray && input isa CuArray || throw(ArgumentError("multi-GPU inputs and outputs must be CuArrays"))
    CUDA.deviceid(CUDA.device(out)) == state.primary_gpu &&
        CUDA.deviceid(CUDA.device(input)) == state.primary_gpu ||
        throw(ArgumentError("inputs and outputs must be on the primary GPU"))
    CUDA.device!(state.primary_gpu)
    return nothing
end

function explicit_multi_forward!(out, state::ExplicitMultiGPUState{T}, input) where {T}
    explicit_multi_check!(state, out, input)
    copyto!(state.host_input, input)
    run_on_workers!(state.completed, state.workers; phase=:explicit_forward) do i
        shard = state.shards[i]
        copyto!(shard.input, state.host_input)
        mul!(shard.signal, shard.op, shard.input)
        copyto!(vec(shard.host_signal), shard.signal)
        return nothing
    end
    for shard in state.shards
        @views state.host_signal[shard.rows, :] .= shard.host_signal
    end
    copyto!(out, vec(state.host_signal))
    return out
end

function explicit_multi_reduce!(out, state::ExplicitMultiGPUState)
    fill!(state.host_output, zero(eltype(state.host_output)))
    for shard in state.shards
        state.host_output .+= shard.host_image
    end
    copyto!(out, state.host_output)
    return out
end

function explicit_multi_adjoint!(out, state::ExplicitMultiGPUState, input)
    explicit_multi_check!(state, out, input)
    copyto!(vec(state.host_signal), input)
    for shard in state.shards
        @views shard.host_signal .= state.host_signal[shard.rows, :]
    end
    run_on_workers!(state.completed, state.workers; phase=:explicit_adjoint) do i
        shard = state.shards[i]
        copyto!(shard.signal, vec(shard.host_signal))
        mul!(shard.output, adjoint(shard.op), shard.signal)
        copyto!(shard.host_image, shard.output)
        return nothing
    end
    return explicit_multi_reduce!(out, state)
end

function explicit_multi_normal!(out, state::ExplicitMultiGPUState, normals, input)
    explicit_multi_check!(state, out, input)
    copyto!(state.host_input, input)
    run_on_workers!(state.completed, state.workers; phase=:explicit_normal) do i
        shard = state.shards[i]
        copyto!(shard.input, state.host_input)
        mul!(shard.output, normals[i], shard.input)
        copyto!(shard.host_image, shard.output)
        return nothing
    end
    return explicit_multi_reduce!(out, state)
end

function ExplicitMultiGPUOp(grid::Grid{T}, kspha::AbstractMatrix{T}, times::AbstractVector{T};
    gpus, fieldmap=zeros(T, grid.matrixSize...), csm=ones(Complex{T}, grid.matrixSize..., 1),
    mask=trues(grid.matrixSize...), k_nominal=kspha[2:4, :], kwargs...,
) where {T<:AbstractFloat}
    ids = Int.(collect(gpus))
    isempty(ids) && throw(ArgumentError("gpus must not be empty"))
    allunique(ids) || throw(ArgumentError("gpus must be unique"))
    all(id -> 0 <= id < length(collect(CUDA.devices())), ids) || throw(ArgumentError("invalid visible GPU ID"))
    nSam = size(kspha, 2)
    length(ids) <= nSam || throw(ArgumentError("each GPU requires at least one sample"))
    length(times) == nSam || throw(DimensionMismatch("times and kspha"))
    size(k_nominal, 2) == nSam || throw(DimensionMismatch("k_nominal and kspha"))
    nGrid = prod(grid.matrixSize)
    nCha = size(csm, ndims(csm))
    ranges = [fld((i-1)*nSam, length(ids))+1:fld(i*nSam, length(ids)) for i in eachindex(ids)]
    workers = DistributedGPUWorker[]
    shards = Vector{Any}(undef, length(ids))
    try
        for id in ids
            push!(workers, DistributedGPUWorker(id))
        end
        warn_if_insufficient_gpu_worker_threads(length(ids); operation=:explicit_encoding)
        run_on_workers!(shards, workers; phase=:explicit_setup) do i
            rows = ranges[i]
            op = ExplicitGPUOp(grid, kspha[:, rows], times[rows];
                fieldmap, csm, mask, k_nominal=k_nominal[:, rows], kwargs...)
            try
                return ExplicitGPUShard(ids[i], rows, op,
                    CUDA.zeros(Complex{T}, nGrid), CUDA.zeros(Complex{T}, nGrid),
                    CUDA.zeros(Complex{T}, length(rows)*nCha), zeros(Complex{T}, nGrid),
                    zeros(Complex{T}, length(rows), nCha))
            catch
                close(op)
                rethrow()
            end
        end
    catch
        # Successful shards can own a large cached P even if another worker
        # fails. Release them before shutting down the persistent workers.
        try
            completed = Vector{Nothing}(undef, length(workers))
            run_on_workers!(completed, workers; phase=:explicit_setup_cleanup) do i
                isassigned(shards, i) && close(shards[i])
                return nothing
            end
        finally
            shutdown_distributed_workers!(workers)
        end
        rethrow()
    end
    typed_shards = [shard::typeof(first(shards)) for shard in shards]
    state = ExplicitMultiGPUState(first(ids), workers, typed_shards, Vector{Nothing}(undef, length(ids)),
        zeros(Complex{T}, nGrid), zeros(Complex{T}, nGrid), zeros(Complex{T}, nSam, nCha), false)
    CUDA.device!(first(ids))
    forward = (out, input) -> explicit_multi_forward!(out, state, input)
    backward = (out, input) -> explicit_multi_adjoint!(out, state, input)
    return ExplicitMultiGPUOp{T,typeof(forward),typeof(backward),typeof(state)}(
        nSam*nCha, nGrid, false, false, forward, nothing, backward, 0, 0, 0,
        CUDA.zeros(Complex{T}, 0), CUDA.zeros(Complex{T}, 0), state)
end

function Base.adjoint(op::ExplicitMultiGPUOp{T}) where {T}
    return LinearOperator{Complex{T}}(op.ncol, op.nrow, false, false,
        op.ctprod!, nothing, op.prod!; S=CuVector{Complex{T}})
end

function LinearOperatorCollection.normalOperator(op::ExplicitMultiGPUOp{T}, weights=nothing; kwargs...) where {T}
    state = op.state
    state.released && throw(ArgumentError("operator is closed"))
    raw = weights isa LinearOperatorCollection.WeightingOp ? weights.weights : weights
    nSam, nCha = size(state.host_signal)
    D = raw === nothing ? ones(T, nSam*nCha) : vec(Array(raw))
    length(D) == nSam && (D = repeat(D, nCha))
    length(D) == op.nrow || throw(DimensionMismatch("normal weights"))
    all(isfinite, D) || throw(ArgumentError("normal weights must be finite"))
    maximum(abs, imag.(D)) <= 10eps(T)*max(one(T), maximum(abs,D)) ||
        throw(ArgumentError("normal weights must already be real nonnegative WᴴW"))
    minimum(real.(D)) >= zero(T) || throw(ArgumentError("normal weights must be nonnegative"))
    diagonal = reshape(D, nSam, nCha)
    results = Vector{Any}(undef, length(state.shards))
    run_on_workers!(results, state.workers; phase=:explicit_normal_setup) do i
        shard = state.shards[i]
        return LinearOperatorCollection.normalOperator(shard.op, diagonal[shard.rows, :])
    end
    normals = [result::typeof(first(results)) for result in results]
    CUDA.device!(state.primary_gpu)
    normal = (out, input) -> explicit_multi_normal!(out, state, normals, input)
    return LinearOperator{Complex{T}}(op.ncol, op.ncol, false, true,
        normal, nothing, normal; S=CuVector{Complex{T}})
end

function explicit_gpu_normal(op::ExplicitMultiGPUOp; weights=nothing)
    return LinearOperatorCollection.normalOperator(op, weights===nothing ? nothing : abs2.(weights))
end

function Base.close(op::ExplicitMultiGPUOp)
    state = op.state
    state.released && return nothing
    try
        run_on_workers!(state.completed, state.workers; phase=:explicit_close) do i
            close(state.shards[i])
            return nothing
        end
    finally
        shutdown_distributed_workers!(state.workers)
        state.released = true
        CUDA.device!(state.primary_gpu)
    end
    return nothing
end
