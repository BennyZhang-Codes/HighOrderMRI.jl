"""
A bounded GPU queue for independent spatial factors stored in host RAM.

Each slot owns a GPU factor block. One NFFT plan is updated on the compute stream,
while factors arrive on a separate CUDA stream through one pinned host buffer.
Ready/consumed events protect factor use and slot reuse. GPU pointers are obtained
on the compute stream to avoid implicit cross-stream array synchronization.
"""
mutable struct PerReadoutCache{T<:AbstractFloat,P<:AbstractNFFTPlan{T}}
    source          :: Union{Nothing,Matrix{Complex{T}},RegeneratedSpatialBasis{T}}
    generator       :: Union{Nothing,SpatialFactorGenerator{T}}
    nodes           :: Array{T,3}
    basis           :: Vector{CuMatrix{Complex{T}}}
    staging         :: Vector{Matrix{Complex{T}}}
    plans           :: Vector{P}
    copy_stream     :: CuStream
    ready           :: Vector{CuEvent}
    consumed        :: Vector{CuEvent}
    copy_start      :: Vector{CuEvent}
    copy_stop       :: Vector{CuEvent}
    resident        :: Vector{Int}
    used            :: Vector{Bool}
    uncounted       :: Vector{Bool}
    generated_slot  :: Vector{Bool}
    wait_start      :: Vector{CuEvent}
    wait_stop       :: Vector{CuEvent}
    wait_uncounted  :: Vector{Bool}
    last_copy_slot  :: Int
    rank            :: Int
    gpu_id          :: Int
    transfers       :: Int
    transferred_bytes :: Int
    host_staging_s  :: Float64
    device_copy_s   :: Float64
    node_update_s   :: Float64
    generated_readouts :: Int
    host_cache_hits  :: Int
    generation_device_s :: Float64
    ready_wait_s    :: Float64
    released        :: Bool
end

function PerReadoutCache(
    source::Union{Matrix{Complex{T}},RegeneratedSpatialBasis{T}}, nodes::Array{T,3}, grid_size;
    capacity::Int, first_plan,
) where {T<:AbstractFloat}
    nDyn = size(nodes, 3)
    nDyn > 0 || throw(ArgumentError("per-readout cache requires at least one readout"))
    capacity > 0 || throw(ArgumentError("per-readout cache capacity must be positive"))
    rank, remainder = divrem(size(source, 2), nDyn)
    remainder == 0 && rank > 0 || throw(DimensionMismatch("spatial columns must contain one rank block per readout"))
    nSlot = min(capacity, nDyn)
    plans = typeof(first_plan)[first_plan]
    basis = [CuMatrix{Complex{T}}(undef, size(source, 1), rank) for _ in 1:nSlot]
    # Stream-ordered allocations must exist before the transfer stream uses raw
    # pointers. Subsequent slot reuse is protected by consumed events.
    CUDA.synchronize()
    staging = [Matrix{Complex{T}}(undef, size(source, 1), rank)]
    foreach(CUDA.pin, staging)
    copy_stream = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    generator = source isa RegeneratedSpatialBasis ? SpatialFactorGenerator(source, basis, copy_stream) : nothing
    return PerReadoutCache{T,typeof(first_plan)}(
        source, generator, nodes, basis, staging, plans, copy_stream,
        [CuEvent() for _ in 1:nSlot], [CuEvent() for _ in 1:nSlot],
        [CuEvent() for _ in 1:nSlot], [CuEvent() for _ in 1:nSlot],
        zeros(Int, nSlot), fill(false, nSlot), fill(false, nSlot), fill(false, nSlot),
        [CuEvent() for _ in 1:nSlot], [CuEvent() for _ in 1:nSlot], fill(false, nSlot), 0,
        rank, CUDA.deviceid(CUDA.device()), 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, false,
    )
end

function collect_per_readout_copy_time!(cache::PerReadoutCache, slot::Int)
    if cache.uncounted[slot]
        CUDA.synchronize(cache.copy_stop[slot])
        elapsed = CUDA.elapsed(cache.copy_start[slot], cache.copy_stop[slot])
        if cache.generated_slot[slot]
            cache.generation_device_s += elapsed
        else
            cache.device_copy_s += elapsed
        end
        cache.uncounted[slot] = false
    end
    if cache.wait_uncounted[slot]
        CUDA.synchronize(cache.wait_stop[slot])
        cache.ready_wait_s += CUDA.elapsed(cache.wait_start[slot], cache.wait_stop[slot])
        cache.wait_uncounted[slot] = false
    end
    return nothing
end

function prefetch_readout!(cache::PerReadoutCache, dyn::Int)
    cache.released && error("Per-readout cache was released")
    1 <= dyn <= size(cache.nodes, 3) || throw(BoundsError(cache.nodes, dyn))
    slot = mod1(dyn, length(cache.basis))
    cache.resident[slot] == dyn && return slot
    # Other slots can continue computing while this slot waits for its last use.
    cache.used[slot] && CUDA.synchronize(cache.consumed[slot])
    collect_per_readout_copy_time!(cache, slot)
    source = something(cache.source)
    host_slot = source isa RegeneratedSpatialBasis ? source.host_slot[dyn] : dyn
    if host_slot == 0
        submit_generated_factor!(something(cache.generator), slot, dyn, cache.copy_stream,
            cache.consumed[slot], cache.used[slot], cache.copy_start[slot],
            cache.copy_stop[slot], cache.ready[slot])
        cache.resident[slot] = dyn
        cache.used[slot] = false
        cache.uncounted[slot] = true
        cache.generated_slot[slot] = true
        cache.generated_readouts += 1
        return slot
    end
    # The shared staging array must remain untouched until its previous DMA
    # completes. This waits for the copy stream, while compute can keep running.
    if cache.last_copy_slot != 0
        CUDA.synchronize(cache.copy_stop[cache.last_copy_slot])
    end
    host = first(cache.staging)
    device = cache.basis[slot]
    host_source = source isa RegeneratedSpatialBasis ? source.host_basis : source
    count = length(host)
    started = time_ns()
    copyto!(host, 1, host_source, (host_slot - 1) * count + 1, count)
    cache.host_staging_s += (time_ns() - started) * 1e-9
    # Asking for this pointer inside stream! would move CUDA.jl's managed-array
    # ownership to the copy stream and synchronize the whole compute stream.
    # Explicit events already order access to this slot, so retain its ownership
    # on the compute stream and pass the raw pointer to the asynchronous copy.
    device_pointer = pointer(device)
    CUDA.stream!(cache.copy_stream) do
        CUDA.record(cache.copy_start[slot], cache.copy_stream)
        GC.@preserve host device begin
            unsafe_copyto!(device_pointer, pointer(host), count;
                async=true, stream=cache.copy_stream)
        end
        CUDA.record(cache.copy_stop[slot], cache.copy_stream)
        CUDA.record(cache.ready[slot], cache.copy_stream)
    end
    cache.resident[slot] = dyn
    cache.used[slot] = false
    cache.uncounted[slot] = true
    cache.generated_slot[slot] = false
    cache.last_copy_slot = slot
    cache.transfers += 1
    cache.transferred_bytes += sizeof(host)
    source isa RegeneratedSpatialBasis && (cache.host_cache_hits += 1)
    return slot
end

function prime_readout_cache!(cache::PerReadoutCache)
    cache.released && error("Per-readout cache was released")
    for dyn in eachindex(cache.basis)
        prefetch_readout!(cache, dyn)
    end
    return nothing
end

function acquire_readout!(cache::PerReadoutCache, dyn::Int,
    readouts::UnitRange{Int}=1:size(cache.nodes, 3))
    slot = mod1(dyn, length(cache.basis))
    cache.resident[slot] == dyn || error("Readout $dyn was not prefetched into slot $slot")
    started = time_ns()
    update_nfft_nodes!(first(cache.plans), cache.nodes[:, :, dyn])
    cache.node_update_s += (time_ns() - started) * 1e-9
    if cache.generator !== nothing
        CUDA.record(cache.wait_start[slot], CUDA.stream())
    end
    CUDA.wait(cache.ready[slot], CUDA.stream())
    if cache.generator !== nothing
        CUDA.record(cache.wait_stop[slot], CUDA.stream())
        cache.wait_uncounted[slot] = true
        # Start producing into the previous consumed slot BEFORE submitting
        # this readout's coils. Some NUFFT operations wait on their stream,
        # so launching at finish_readout! would leave little work to overlap.
        previous = dyn - 1
        next_readout = previous + length(cache.basis)
        if length(cache.basis) > 1 && previous >= first(readouts) && next_readout <= last(readouts)
            prefetch_readout!(cache, next_readout)
        end
    end
    return slot
end

function finish_readout!(cache::PerReadoutCache, dyn::Int)
    return finish_readout!(cache, dyn, 1:size(cache.nodes, 3))
end

function finish_readout!(cache::PerReadoutCache, dyn::Int, readouts::UnitRange{Int})
    slot = mod1(dyn, length(cache.basis))
    CUDA.record(cache.consumed[slot], CUDA.stream())
    cache.used[slot] = true
    # Submit the current readout's kernels first. Then refill the PREVIOUS
    # completed slot, allowing the copy to overlap the current computation.
    nSlot = length(cache.basis)
    cache.generator !== nothing && nSlot > 1 && return nothing
    previous = nSlot == 1 ? dyn : dyn - 1
    next_readout = previous + nSlot
    if previous >= first(readouts) && next_readout <= last(readouts)
        prefetch_readout!(cache, next_readout)
    end
    return nothing
end

function drain_readout_copies!(cache::PerReadoutCache)
    CUDA.synchronize(cache.copy_stream)
    for slot in eachindex(cache.basis)
        collect_per_readout_copy_time!(cache, slot)
    end
    return nothing
end

function per_readout_cache_stats(cache::PerReadoutCache)
    drain_readout_copies!(cache)
    return (capacity=length(cache.basis), transfers=cache.transfers,
        transferred_bytes=cache.transferred_bytes, host_staging_s=cache.host_staging_s,
        device_copy_s=cache.device_copy_s, node_update_s=cache.node_update_s,
        generated_readouts=cache.generated_readouts, host_cache_hits=cache.host_cache_hits,
        generation_device_s=cache.generation_device_s, ready_wait_s=cache.ready_wait_s)
end

function release_per_readout_cache!(cache::PerReadoutCache)
    cache.released && return nothing
    CUDA.device!(cache.gpu_id)
    CUDA.synchronize(cache.copy_stream)
    for slot in eachindex(cache.basis)
        cache.used[slot] && CUDA.synchronize(cache.consumed[slot])
    end
    cache.generator === nothing || close(cache.generator)
    for slot in eachindex(cache.basis)
        CUDA.unsafe_free!(cache.basis[slot])
    end
    empty!(cache.basis)
    empty!(cache.staging)
    empty!(cache.plans)
    cache.source = nothing
    cache.released = true
    return nothing
end
