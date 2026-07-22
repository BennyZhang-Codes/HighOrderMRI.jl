mutable struct DistributedGPUWorker
    gpu_id  :: Int
    jobs    :: Channel{Any}
    results :: Channel{Any}
    task    :: Task
end


struct DistributedGPUWorkerError <: Exception
    gpu_id      :: Int
    worker_index:: Int
    phase       :: Symbol
    captured    :: CapturedException
end


function Base.showerror(io::IO, err::DistributedGPUWorkerError)
    print(
        io,
        "Distributed GPU worker failed: ",
        "gpu=$(err.gpu_id), ",
        "worker=$(err.worker_index), ",
        "phase=$(err.phase)\n",
    )
    showerror(io, err.captured)
end


function DistributedGPUWorker(gpu_id::Int)
    jobs    = Channel{Any}(1)
    results = Channel{Any}(1)
    ready   = Channel{Any}(1)

    task = Threads.@spawn begin
        startup_result = try
            CUDA.device!(gpu_id)
            (:ok, nothing)
        catch err
            (:error, CapturedException(err, catch_backtrace()))
        end

        put!(ready, startup_result)
        startup_result[1] === :error && return nothing
        startup_result = nothing

        for job in jobs
            result = try
                # A persistent worker may predate closures created by later
                # top-level calls (for example, separate test files or REPL
                # invocations), so execute each submitted job in the latest
                # method world.
                (:ok, Base.invokelatest(job))
            catch err
                (:error, CapturedException(err, catch_backtrace()))
            end

            put!(results, result)

            job = nothing
            result = nothing
        end
    end

    status, payload = take!(ready)
    if status === :error
        wait(task)
        throw(DistributedGPUWorkerError(gpu_id, 0, :startup, payload))
    end

    return DistributedGPUWorker(gpu_id, jobs, results, task)
end


function run_on_workers!(
    f       :: F,
    outputs :: AbstractVector,
    workers :: AbstractVector{DistributedGPUWorker};
    phase   :: Symbol = :unspecified,
) where {F}

    @assert length(outputs) == length(workers)

    for i in eachindex(workers)
        put!(workers[i].jobs, () -> f(i))
    end

    failures = DistributedGPUWorkerError[]

    for i in eachindex(workers)
        status, payload = take!(workers[i].results)

        if status === :ok
            outputs[i] = payload
        elseif status === :error
            push!(failures, DistributedGPUWorkerError(workers[i].gpu_id, i, phase, payload))
        else
            error("Invalid distributed worker response: " * "gpu=$(workers[i].gpu_id), status=$status")
        end
    end

    isempty(failures) || throw(first(failures))

    return outputs
end


function shutdown_distributed_workers!(workers::AbstractVector{DistributedGPUWorker})
    for worker in workers
        isopen(worker.jobs) && close(worker.jobs)
    end

    for worker in workers
        wait(worker.task)
    end

    return nothing
end
