using CUDA
using LinearAlgebra
import LinearOperatorCollection

export release_highorder_normal_backend!


mutable struct MultiGPUHighOrderNormalWorkspace{T<:AbstractFloat}
    grid_forward :: CuVector{Complex{T}}
    grid_adjoint :: CuVector{Complex{T}}
    k_signal     :: CuVector{Complex{T}}
    k_channel    :: CuVector{Complex{T}}
    k_weighted   :: CuVector{Complex{T}}
    x_input      :: CuVector{Complex{T}}
    x_output     :: CuVector{Complex{T}}
end


mutable struct MultiGPUHighOrderNormalShard{T<:AbstractFloat, P<:AbstractNFFTPlan{T}}
    gpu_id        :: Int
    channels      :: UnitRange{Int}
    q             :: CuMatrix{Complex{T}}
    basis         :: CuMatrix{Complex{T}}
    csm           :: CuMatrix{Complex{T}}
    weights2      :: CuVector{T}
    mask_idx      :: CuVector{Int32}
    nfftplan      :: Union{Nothing,P}
    grid_size     :: Tuple
    workspace     :: MultiGPUHighOrderNormalWorkspace{T}
    released      :: Bool
end


mutable struct MultiGPUHighOrderNormalState{
    T<:AbstractFloat,
    S<:MultiGPUHighOrderNormalShard{T},
}
    workers       :: Vector{DistributedGPUWorker}
    shards        :: Vector{S}
    primary_gpu   :: Int
    nVox          :: Int
    nGrid         :: Int
    mask_idx      :: CuVector{Int32}
    primary_input :: CuVector{Complex{T}}
    primary_sum   :: CuVector{Complex{T}}
    host_input    :: Vector{Complex{T}}
    host_outputs  :: Vector{Vector{Complex{T}}}
    released      :: Bool
end


mutable struct MultiGPUHighOrderNormalOp{
    T<:AbstractFloat,
    F,
    S<:AbstractVector{Complex{T}},
    State<:MultiGPUHighOrderNormalState{T},
} <: AbstractLinearOperator{Complex{T}}
    nrow      :: Int
    ncol      :: Int
    symmetric :: Bool
    hermitian :: Bool
    prod!     :: F
    tprod!    :: Nothing
    ctprod!   :: Nothing
    nprod     :: Int
    ntprod    :: Int
    nctprod   :: Int
    Mv        :: S
    Mtu       :: S
    state     :: State
end


LinearOperators.storage_type(op::MultiGPUHighOrderNormalOp) = typeof(op.Mv)
Base.eltype(::MultiGPUHighOrderNormalOp{T}) where {T} = Complex{T}

"""
A lightweight normal-operator view whose multi-GPU resources are owned by its
parent `HighOrderLowRankOp`.
"""
mutable struct HighOrderLowRankNormalOp{
    T<:AbstractFloat,
    F,
    S<:AbstractVector{Complex{T}},
    P<:HighOrderLowRankOp{T},
} <: AbstractLinearOperator{Complex{T}}
    nrow      :: Int
    ncol      :: Int
    symmetric :: Bool
    hermitian :: Bool
    prod!     :: F
    tprod!    :: Nothing
    ctprod!   :: Nothing
    nprod     :: Int
    ntprod    :: Int
    nctprod   :: Int
    Mv        :: S
    Mtu       :: S
    parent    :: P
end


LinearOperators.storage_type(op::HighOrderLowRankNormalOp) = typeof(op.Mv)
Base.eltype(::HighOrderLowRankNormalOp{T}) where {T} = Complex{T}


function balanced_channel_ranges(nCha::Int, nPart::Int)
    nCha > 0 || throw(ArgumentError("nCha must be positive"))
    nPart > 0 || throw(ArgumentError("nPart must be positive"))
    nPart <= nCha || throw(ArgumentError("nPart must not exceed nCha"))

    nBase, nExtra = divrem(nCha, nPart)
    ranges = Vector{UnitRange{Int}}(undef, nPart)
    first_channel = 1

    for i = 1:nPart
        nLocal = nBase + (i <= nExtra ? 1 : 0)
        last_channel = first_channel + nLocal - 1
        ranges[i] = first_channel:last_channel
        first_channel = last_channel + 1
    end

    return ranges
end


function kernel_gather_masked_image!(x_masked, x, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds x_masked[v] = x[mask_idx[v]]
    end
    return nothing
end


function kernel_scatter_masked_image!(x, x_masked, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds x[mask_idx[v]] = x_masked[v]
    end
    return nothing
end


function kernel_scatter_basis_csm!(grid_flat, x_masked, basis, r, csm, c, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds begin
            idx = mask_idx[v]
            grid_flat[idx] = x_masked[v] * conj(basis[v, r]) * csm[v, c]
        end
    end
    return nothing
end


function kernel_accumulate_basis_csm!(x_masked, grid_flat, basis, r, csm, c, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds begin
            idx = mask_idx[v]
            x_masked[v] += grid_flat[idx] * conj(csm[v, c]) * basis[v, r]
        end
    end
    return nothing
end


function local_highorder_normal!(shard::MultiGPUHighOrderNormalShard{T}) where {T}
    shard.released && error("Multi-GPU normal-operator shard on GPU $(shard.gpu_id) was released")
    plan = shard.nfftplan
    plan === nothing && error("NFFT plan on GPU $(shard.gpu_id) was released")

    workspace = shard.workspace
    fill!(workspace.x_output, zero(Complex{T}))

    nVox = length(workspace.x_input)
    shared_rank = size(shard.basis, 2)
    nfft_dims = shard.grid_size[end] == 1 ? shard.grid_size[1:2] : shard.grid_size
    nfft_forward = reshape(workspace.grid_forward, nfft_dims)
    nfft_adjoint = reshape(workspace.grid_adjoint, nfft_dims)

    threads = 256
    blocks_vox = cld(nVox, threads)

    # Channel sharding is exact because A^H W^2 A is a sum of independent
    # per-channel normal operators. Keeping one channel's k-space at a time
    # avoids a persistent nPoint-by-nLocalChannel buffer on every GPU.
    for c = 1:length(shard.channels)
        fill!(workspace.k_channel, zero(Complex{T}))

        for r = 1:shared_rank
            @cuda threads=threads blocks=blocks_vox kernel_scatter_basis_csm!(
                workspace.grid_forward, workspace.x_input, shard.basis,
                r, shard.csm, c, shard.mask_idx, nVox)
            
            mul!(workspace.k_signal, plan, nfft_forward)
            @views @. workspace.k_channel += workspace.k_signal * shard.q[:, r]
        end

        # q already contains the forward factor 1/sqrt(nVox); using q and
        # conj(q) in this fused path therefore supplies the normal operator's
        # combined 1/nVox factor without an additional runtime scale.
        @. workspace.k_channel *= shard.weights2

        for r = 1:shared_rank
            @views @. workspace.k_weighted = workspace.k_channel * conj(shard.q[:, r])
            mul!(nfft_adjoint, adjoint(plan), workspace.k_weighted)
            @cuda threads=threads blocks=blocks_vox kernel_accumulate_basis_csm!(
                workspace.x_output, workspace.grid_adjoint, shard.basis,
                r, shard.csm, c, shard.mask_idx, nVox)
        end
    end

    return workspace.x_output
end


function release_multi_gpu_normal_shard!(shard::MultiGPUHighOrderNormalShard)
    shard.released && return nothing
    shard.released = true

    # NFFT and CUDA kernels may still be executing asynchronously when the
    # solver returns. Complete every operation on this device before any
    # plan-owned or workspace allocation can be released.
    CUDA.device!(shard.gpu_id)
    CUDA.device_synchronize(; blocking=true)

    # The plan owns backend-specific GPU buffers. Drop its last reference on
    # the worker that owns the CUDA context. Its internal allocations are
    # collected once, after every worker has stopped.
    shard.nfftplan = nothing

    CUDA.unsafe_free!(shard.q)
    CUDA.unsafe_free!(shard.basis)
    CUDA.unsafe_free!(shard.csm)
    CUDA.unsafe_free!(shard.weights2)
    CUDA.unsafe_free!(shard.mask_idx)

    workspace = shard.workspace
    CUDA.unsafe_free!(workspace.grid_forward)
    CUDA.unsafe_free!(workspace.grid_adjoint)
    CUDA.unsafe_free!(workspace.k_signal)
    CUDA.unsafe_free!(workspace.k_channel)
    CUDA.unsafe_free!(workspace.k_weighted)
    CUDA.unsafe_free!(workspace.x_input)
    CUDA.unsafe_free!(workspace.x_output)

    # Do not call GC.gc or CUDA.reclaim from persistent GPU workers. Julia's
    # GC is process-wide, while reclaim performs a device-wide memory-pool
    # trim; running either operation concurrently on several CUDA contexts can
    # make teardown race with finalizers from another device.
    return nothing
end


function multi_gpu_normal_prod!(
    y     :: AbstractVector{Complex{T}},
    state :: MultiGPUHighOrderNormalState{T},
    x     :: AbstractVector{Complex{T}},
) where {T}
    state.released && error("Multi-GPU normal operator was released")
    length(x) == state.nGrid || throw(DimensionMismatch("input length must be $(state.nGrid)"))
    length(y) == state.nGrid || throw(DimensionMismatch("output length must be $(state.nGrid)"))
    x isa CuArray || throw(ArgumentError("Multi-GPU normal operator requires a CuArray input"))
    y isa CuArray || throw(ArgumentError("Multi-GPU normal operator requires a CuArray output"))

    input_gpu = CUDA.deviceid(CUDA.device(x))
    output_gpu = CUDA.deviceid(CUDA.device(y))
    input_gpu == state.primary_gpu || throw(ArgumentError("input is on GPU $input_gpu, but the operator primary GPU is $(state.primary_gpu)"))
    output_gpu == state.primary_gpu || throw(ArgumentError("output is on GPU $output_gpu, but the operator primary GPU is $(state.primary_gpu)"))

    CUDA.device!(state.primary_gpu)

    threads = 256
    blocks_vox = cld(state.nVox, threads)
    @cuda threads=threads blocks=blocks_vox kernel_gather_masked_image!(state.primary_input, x, state.mask_idx, state.nVox)
    copyto!(state.host_input, state.primary_input)

    completed = Vector{Nothing}(undef, length(state.workers))
    run_on_workers!(completed, state.workers; phase=:normal_operator) do i
        shard = state.shards[i]
        copyto!(shard.workspace.x_input, state.host_input)
        local_highorder_normal!(shard)
        copyto!(state.host_outputs[i], shard.workspace.x_output)
        nothing
    end

    host_sum = state.host_outputs[1]
    for i = 2:length(state.host_outputs)
        host_part = state.host_outputs[i]
        @inbounds @simd for v in eachindex(host_sum)
            host_sum[v] += host_part[v]
        end
    end

    copyto!(state.primary_sum, host_sum)
    fill!(y, zero(Complex{T}))
    @cuda threads=threads blocks=blocks_vox kernel_scatter_masked_image!(y, state.primary_sum, state.mask_idx, state.nVox)
    CUDA.synchronize()

    return y
end


"""
    MultiGPUHighOrderNormalOp(op, weights; gpus, verbose=false)

Build a channel-sharded multi-GPU implementation of
`adjoint(op) * Diagonal(abs2.(weights)) * op`.

`weights` must contain one value per sample/dynamic point, not one repeated copy
per channel. The existing single-GPU `op` stays unchanged and can still be used as
the forward operator passed to the reconstruction solver. The first GPU in `gpus`
must be the GPU that owns `op` and the solver vectors.

Call [`release_multi_gpu_normal_operator!`](@ref), or `close`, after reconstruction
to stop the persistent workers and release all per-GPU NFFT plans and workspaces.
"""
function MultiGPUHighOrderNormalOp(
    op                   :: HighOrderLowRankOp{T},
    weights              :: AbstractArray;
    gpus                 :: Vector{Int},
    verbose              :: Bool = false,
    _weights_are_squared :: Bool = false,
) where {T<:AbstractFloat}
    op.q isa CuArray || throw(ArgumentError("MultiGPUHighOrderNormalOp requires a CuArray HighOrderLowRankOp"))
    isempty(gpus) && throw(ArgumentError("gpus must contain at least one GPU id"))
    length(unique(gpus)) == length(gpus) || throw(ArgumentError("gpus contains duplicate GPU ids: $gpus"))

    primary_gpu = first(gpus)
    source_gpu = CUDA.deviceid(CUDA.device(op.q))
    source_gpu == primary_gpu || throw(ArgumentError("the first GPU ($primary_gpu) must own the source operator; op is on GPU $source_gpu"))

    nWorker = min(length(gpus), op.nCha)
    setup_gpus = gpus[1:nWorker]
    channel_ranges = balanced_channel_ranges(op.nCha, nWorker)
    warn_if_insufficient_gpu_worker_threads(
        nWorker;
        operation=:multi_gpu_normal_operator,
    )
    nPoint = op.nSam * op.nDyn
    nGrid = prod(op.grid_size)

    length(weights) == nPoint || throw(DimensionMismatch("weights must have length nSam*nDyn=$nPoint; got $(length(weights))"))

    CUDA.device!(primary_gpu)
    q_host = Matrix{Complex{T}}(Array(op.q))
    basis_host = Matrix{Complex{T}}(Array(op.basis))
    csm_host = Matrix{Complex{T}}(Array(op.csm))
    mask_idx_host = Vector{Int32}(Array(op.mask_idx))
    weights_host = vec(Array(weights))
    weights2_host = if _weights_are_squared
        maximum(abs, imag.(weights_host); init=zero(T)) <= T(64) * eps(T) * max(
            maximum(abs, real.(weights_host); init=zero(T)),
            one(T),
        ) || throw(ArgumentError("squared normal weights must be real"))
        real_weights = T.(real.(weights_host))
        minimum(real_weights; init=zero(T)) >= -T(64) * eps(T) * max(
            maximum(abs, real_weights; init=zero(T)),
            one(T),
        ) || throw(ArgumentError("squared normal weights must be non-negative"))
        max.(real_weights, zero(T))
    else
        T.(abs2.(weights_host))
    end
    nfft_nodes_host = reshape(op.nfft_traj, size(op.nfft_traj, 1), nPoint)
    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size

    workers = DistributedGPUWorker[]
    shards_any = Vector{Any}(undef, nWorker)

    try
        for gpu_id in setup_gpus
            push!(workers, DistributedGPUWorker(gpu_id))
        end

        run_on_workers!(shards_any, workers; phase=:normal_operator_setup) do i
            gpu_id = setup_gpus[i]
            channels = channel_ranges[i]
            nfftplan = plan_nfft(CuArray, nfft_nodes_host, nfft_dims; m=3, σ=1.25)
            eltype(nfftplan) == Complex{T} || error("NFFT plan precision $(eltype(nfftplan)) does not match Complex{$T}")

            workspace = MultiGPUHighOrderNormalWorkspace{T}(
                CUDA.zeros(Complex{T}, nGrid),
                CUDA.zeros(Complex{T}, nGrid),
                CUDA.zeros(Complex{T}, nPoint),
                CUDA.zeros(Complex{T}, nPoint),
                CUDA.zeros(Complex{T}, nPoint),
                CUDA.zeros(Complex{T}, op.nVox),
                CUDA.zeros(Complex{T}, op.nVox),
            )

            MultiGPUHighOrderNormalShard{T,typeof(nfftplan)}(
                gpu_id,
                channels,
                CuArray(q_host),
                CuArray(basis_host),
                CuArray(@view csm_host[:, channels]),
                CuArray(weights2_host),
                CuArray(mask_idx_host),
                nfftplan,
                op.grid_size,
                workspace,
                false,
            )
        end
    catch
        if !isempty(workers)
            cleanup = Vector{Nothing}(undef, length(workers))
            try
                run_on_workers!(cleanup, workers; phase=:normal_operator_setup_cleanup) do i
                    if isassigned(shards_any, i) && shards_any[i] isa MultiGPUHighOrderNormalShard
                        release_multi_gpu_normal_shard!(shards_any[i])
                    else
                        # A failure can occur after a plan was allocated but
                        # before the shard was returned to the caller. Wait for
                        # any submitted work here; unreachable allocations are
                        # collected after all workers have stopped.
                        CUDA.device_synchronize(; blocking=true)
                    end
                    nothing
                end
            catch cleanup_error
                @warn "Failed to clean up a partially constructed multi-GPU normal operator" exception=(cleanup_error, catch_backtrace())
            end
            shutdown_distributed_workers!(workers)
            GC.gc(false)
        end
        rethrow()
    end

    shard_type = typeof(shards_any[1])
    all(shard -> shard isa shard_type, shards_any) || error("Per-GPU NFFT plan types do not match")
    shards = Vector{shard_type}(shards_any)

    CUDA.device!(primary_gpu)
    mask_idx = CuArray(mask_idx_host)
    primary_input = CUDA.zeros(Complex{T}, op.nVox)
    primary_sum = CUDA.zeros(Complex{T}, op.nVox)

    # Keep transfers topology-independent in the first implementation. These
    # buffers are reused for every CG iteration, so there are no O(nVox*nGPU)
    # host allocations in the solve loop.
    host_input = Vector{Complex{T}}(undef, op.nVox)
    host_outputs = [Vector{Complex{T}}(undef, op.nVox) for _ = 1:nWorker]

    state = MultiGPUHighOrderNormalState(
        workers,
        shards,
        primary_gpu,
        op.nVox,
        nGrid,
        mask_idx,
        primary_input,
        primary_sum,
        host_input,
        host_outputs,
        false,
    )

    product = (y, x) -> multi_gpu_normal_prod!(y, state, x)
    empty_vector = similar(op.q, Complex{T}, 0)

    if verbose
        @info("Multi-GPU normal operator ready",
            gpus=setup_gpus,
            channel_ranges=channel_ranges,
            shared_rank=size(op.basis, 2),
            nPoint,
            nVox=op.nVox,
            transfer=:host_staged,
        )
    end

    return MultiGPUHighOrderNormalOp(nGrid, nGrid, false, true, product, nothing, nothing, 0, 0, 0, empty_vector, empty_vector, state)
end


function release_multi_gpu_normal_operator!(op::MultiGPUHighOrderNormalOp{T}) where {T}
    state = op.state
    state.released && return nothing
    # Do not retry a partially completed multi-device cleanup. This mirrors the
    # distributed-rSVD release path: a CUDA failure can leave a context unusable.
    state.released = true

    completed = Vector{Nothing}(undef, length(state.workers))
    try
        run_on_workers!(completed, state.workers; phase=:normal_operator_release) do i
            release_multi_gpu_normal_shard!(state.shards[i])
            nothing
        end
    finally
        try
            shutdown_distributed_workers!(state.workers)
        finally
            CUDA.device!(state.primary_gpu)
            CUDA.device_synchronize(; blocking=true)
            CUDA.unsafe_free!(state.mask_idx)
            CUDA.unsafe_free!(state.primary_input)
            CUDA.unsafe_free!(state.primary_sum)

            state.host_input = Complex{T}[]
            state.host_outputs = Vector{Vector{Complex{T}}}()
            empty!(state.shards)
            empty!(state.workers)

            # Workers and their job closures no longer retain NFFT plans, so a
            # single coordinator-side collection can now release plan-owned
            # objects safely. Do not call CUDA.reclaim here: unsafe_free!
            # already returns explicit workspaces to CUDA.jl's reusable pool,
            # and trimming several device pools during teardown has caused
            # cuMemFreeAsync/context failures on large multi-GPU workloads.
            GC.gc(false)
            CUDA.device!(state.primary_gpu)
        end
    end

    return nothing
end


Base.close(op::MultiGPUHighOrderNormalOp) = release_multi_gpu_normal_operator!(op)


function normal_weights2(op :: HighOrderLowRankOp{T}, weights) where {T<:AbstractFloat}
    nPoint = op.nSam * op.nDyn
    weights === nothing && return ones(T, nPoint)

    raw = weights isa LinearOperatorCollection.WeightingOp ? weights.weights : weights
    raw isa AbstractVector || throw(ArgumentError("the optimized normal backend requires a WeightingOp, an AbstractVector, or nothing"))

    nWeight = length(raw)
    expected_repeated = nPoint * op.nCha
    nWeight in (nPoint, expected_repeated) || throw(DimensionMismatch("normal weights must have length $nPoint or $expected_repeated; got $nWeight"))

    first_segment = Vector{eltype(raw)}(undef, nPoint)
    copyto!(first_segment, @view raw[1:nPoint])

    if nWeight == expected_repeated
        segment = similar(first_segment)
        rtol = T(8) * sqrt(eps(T))
        atol = T(8) * eps(T)
        for c = 2:op.nCha
            rows = ((c - 1) * nPoint + 1):(c * nPoint)
            copyto!(segment, @view raw[rows])
            isapprox(segment, first_segment; rtol, atol) || throw(ArgumentError("normal_distribution=:channel currently requires identical weights for every channel"))
        end
    end

    imag_scale = maximum(abs, imag.(first_segment); init=zero(T))
    real_scale = maximum(abs, real.(first_segment); init=zero(T))
    tolerance = T(64) * eps(T) * max(real_scale, one(T))
    imag_scale <= tolerance || throw(ArgumentError("weights passed to normalOperator must already represent WᴴW and therefore be real"))

    weights2 = T.(real.(first_segment))
    minimum(weights2; init=zero(T)) >= -tolerance || throw(ArgumentError("weights passed to normalOperator must already represent non-negative WᴴW",))
    return max.(weights2, zero(T))
end


function highorder_normal_view(
    parent   :: HighOrderLowRankOp{T},
    normal_op:: MultiGPUHighOrderNormalOp{T},
) where {T<:AbstractFloat}
    state = normal_op.state
    product = (y, x) -> multi_gpu_normal_prod!(y, state, x)
    empty_vector = similar(parent.q, Complex{T}, 0)
    nGrid = prod(parent.grid_size)

    return HighOrderLowRankNormalOp(
        nGrid,
        nGrid,
        false,
        true,
        product,
        nothing,
        nothing,
        0,
        0,
        0,
        empty_vector,
        empty_vector,
        parent)
end


function ensure_highorder_normal_backend!(
    op       :: HighOrderLowRankOp{T},
    weights2 :: Vector{T},
) where {T<:AbstractFloat}
    backend = op.normal_backend
    backend.distribution === :channel || error("ensure_highorder_normal_backend! requires normal_distribution=:channel")

    cached = backend.operator
    if cached !== nothing
        cached isa MultiGPUHighOrderNormalOp || error("invalid cached normal backend")
        cached.state.released && error("the cached multi-GPU normal backend was released; call release_highorder_normal_backend! before rebuilding")
        isequal(backend.weights2, weights2) || throw(ArgumentError(
            "this HighOrderLowRankOp already owns a normal backend with different weights; " *
            "call release_highorder_normal_backend!(op) before rebuilding it",
        ))
        return cached
    end

    normal_op = MultiGPUHighOrderNormalOp(
        op,
        weights2;
        gpus=backend.gpus,
        verbose=backend.verbose,
        _weights_are_squared=true,
    )
    backend.operator = normal_op
    backend.weights2 = copy(weights2)
    return normal_op
end


function LinearOperatorCollection.normalOperator(
    op      :: HighOrderLowRankOp{T},
    weights = nothing;
    kwargs...,
) where {T<:AbstractFloat}
    backend = op.normal_backend
    if backend.distribution === :single
        return LinearOperatorCollection.NormalOp(eltype(LinearOperators.storage_type(op)); parent=op, weights=weights)
    end

    isempty(kwargs) || @debug("Ignoring normalOperator keyword arguments for the channel-distributed HighOrderLowRankOp", kwargs)
    weights2 = normal_weights2(op, weights)
    normal_op = ensure_highorder_normal_backend!(op, weights2)
    return highorder_normal_view(op, normal_op)
end


function release_highorder_normal_backend!(op::HighOrderLowRankOp)
    backend = op.normal_backend
    normal_op = backend.operator
    normal_op === nothing && return nothing

    try
        release_multi_gpu_normal_operator!(normal_op)
    finally
        backend.operator = nothing
        backend.weights2 = nothing
    end
    return nothing
end


# The parent owns the distributed plans and workspaces. Closing a view must not
# invalidate other views created from the same HighOrderLowRankOp.
Base.close(::HighOrderLowRankNormalOp) = nothing
Base.close(op::HighOrderLowRankOp) = release_highorder_normal_backend!(op)
