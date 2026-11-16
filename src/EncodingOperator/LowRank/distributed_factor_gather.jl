struct DistributedFactorDownload{T<:AbstractFloat}
    host_parts :: Vector{Matrix{Complex{T}}}
    stream     :: CuStream
    ready      :: Vector{CuEvent}
    copy_start :: Vector{CuEvent}
    copy_done  :: Vector{CuEvent}
end


"""
A bounded two-slot pipeline for downloading voxel-sharded spatial factors and
assembling them in host RAM. Compute workers submit downloads; separate workers
wait for them and write disjoint voxel ranges into the final factor bank.
"""
mutable struct DistributedFactorGather{T<:AbstractFloat}
    destination     :: Matrix{Complex{T}}
    workspace       :: DistributedRSVDWorkspace{T}
    collectors      :: Vector{DistributedGPUWorker}
    downloads       :: Vector{DistributedFactorDownload{T}}
    v_read_finished :: Vector{Union{Nothing,CuEvent}}
    submitted       :: Int
    pending         :: Int
    host_write_s    :: Float64
    download_wait_s :: Float64
    device_copy_s   :: Float64
    released        :: Bool
end


function DistributedFactorGather(
    destination::Matrix{Complex{T}},
    workspace::DistributedRSVDWorkspace{T},
) where {T}
    size(destination, 1) == workspace.nVox || throw(DimensionMismatch("factor bank has the wrong voxel count"))
    size(destination, 2) % workspace.L_rank == 0 || throw(DimensionMismatch("factor bank must contain full readout blocks"))
    collectors = DistributedGPUWorker[]
    downloads = Vector{DistributedFactorDownload{T}}(undef, length(workspace.workers))
    try
        for worker in workspace.workers
            push!(collectors, DistributedGPUWorker(worker.gpu_id))
        end
        run_on_workers!(downloads, workspace.workers; phase=:factor_gather_setup) do i
            shape = size(workspace.shards[i].v_scaled)
            host_parts = [Matrix{Complex{T}}(undef, shape) for _ in 1:2]
            foreach(CUDA.pin, host_parts)
            DistributedFactorDownload(host_parts,
                CuStream(; flags=CUDA.STREAM_NON_BLOCKING),
                [CuEvent() for _ in 1:2], [CuEvent() for _ in 1:2], [CuEvent() for _ in 1:2])
        end
    catch
        shutdown_distributed_workers!(collectors)
        rethrow()
    end
    return DistributedFactorGather(destination, workspace, collectors, downloads,
        Union{Nothing,CuEvent}[nothing for _ in downloads], 0, 0, 0.0, 0.0, 0.0, false)
end


function collect_factor_gather_result!(gather::DistributedFactorGather)
    gather.pending > 0 || return nothing
    failure = nothing
    # Always drain every worker, including when one collector failed.
    for (i, worker) in enumerate(gather.collectors)
        status, payload = take!(worker.results)
        if status === :ok
            gather.download_wait_s += payload.download_wait_s
            gather.host_write_s += payload.host_write_s
            gather.device_copy_s += payload.device_copy_s
        elseif failure === nothing
            failure = DistributedGPUWorkerError(worker.gpu_id, i, :factor_host_write, payload)
        end
    end
    gather.pending -= 1
    failure === nothing || throw(failure)
    return nothing
end


function write_downloaded_factor!(gather::DistributedFactorGather, i::Int, slot::Int, dyn::Int)
    download = gather.downloads[i]
    started = time_ns()
    CUDA.synchronize(download.copy_done[slot])
    download_wait_s = (time_ns() - started) * 1e-9
    device_copy_s = CUDA.elapsed(download.copy_start[slot], download.copy_done[slot])
    started = time_ns()
    columns = ((dyn - 1) * gather.workspace.L_rank + 1):(dyn * gather.workspace.L_rank)
    @views gather.destination[gather.workspace.shards[i].voxels, columns] .= download.host_parts[slot]
    return (; download_wait_s, device_copy_s, host_write_s=(time_ns() - started) * 1e-9)
end


function submit_factor_gather!(gather::DistributedFactorGather, dyn::Int)
    gather.released && error("factor gather pipeline was released")
    1 <= dyn <= size(gather.destination, 2) ÷ gather.workspace.L_rank ||
        throw(BoundsError(gather.destination, dyn))
    # A slot is reusable only after BOTH its download and CPU write finish.
    gather.pending == 2 && collect_factor_gather_result!(gather)
    slot = mod1(gather.submitted + 1, 2)
    completed = Vector{Nothing}(undef, length(gather.workspace.workers))
    run_on_workers!(completed, gather.workspace.workers; phase=:factor_download_submit) do i
        source = gather.workspace.shards[i].v_scaled
        download = gather.downloads[i]
        host = download.host_parts[slot]
        # Obtain the pointer on the compute stream. Explicit events handle
        # cross-stream ownership without synchronizing all subsequent compute.
        source_pointer = pointer(source)
        CUDA.record(download.ready[slot], CUDA.stream())
        CUDA.stream!(download.stream) do
            CUDA.wait(download.ready[slot], download.stream)
            CUDA.record(download.copy_start[slot], download.stream)
            GC.@preserve host source begin
                unsafe_copyto!(pointer(host), source_pointer, length(host);
                    async=true, stream=download.stream)
            end
            CUDA.record(download.copy_done[slot], download.stream)
        end
        gather.v_read_finished[i] = download.copy_done[slot]
        nothing
    end
    for (i, worker) in enumerate(gather.collectors)
        put!(worker.jobs, () -> write_downloaded_factor!(gather, i, slot, dyn))
    end
    gather.submitted += 1
    gather.pending += 1
    return nothing
end


function drain_factor_gather!(gather::DistributedFactorGather)
    failure = nothing
    while gather.pending > 0
        try
            collect_factor_gather_result!(gather)
        catch err
            failure === nothing && (failure = err)
        end
    end
    failure === nothing || throw(failure)
    return nothing
end


function Base.close(gather::DistributedFactorGather)
    gather.released && return nothing
    try
        drain_factor_gather!(gather)
    finally
        gather.released = true
        try
            shutdown_distributed_workers!(gather.collectors)
        finally
            # Also finish downloads submitted before a submission error,
            # which may not yet have a corresponding collector job.
            completed = Vector{Nothing}(undef, length(gather.workspace.workers))
            run_on_workers!(completed, gather.workspace.workers; phase=:factor_gather_release) do i
                CUDA.synchronize(gather.downloads[i].stream)
                nothing
            end
        end
    end
    return nothing
end
