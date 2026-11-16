"""GPU-owned rank and coil blocks for the shared high-order representation."""
mutable struct RankSharedShard{T<:AbstractFloat,P<:AbstractNFFTPlan{T}}
    gpu_id       :: Int
    ranks        :: UnitRange{Int}
    channels     :: UnitRange{Int}
    q            :: CuMatrix{Complex{T}}
    basis        :: CuMatrix{Complex{T}}
    csm          :: CuMatrix{Complex{T}}
    coil         :: CuVector{Complex{T}}
    mask_idx     :: CuVector{Int32}
    weights2     :: CuVector{T}
    sqrt_weights :: CuVector{Complex{T}}
    nfftplan     :: Union{Nothing,P}
    grid_size    :: Tuple
    rows         :: UnitRange{Int}
    workspace    :: MultiGPUHighOrderNormalWorkspace{T}
    released     :: Bool
end


mutable struct RankSharedState{T<:AbstractFloat,S<:RankSharedShard{T}}
    workers      :: Vector{DistributedGPUWorker}
    shards       :: Vector{S}
    primary_gpu  :: Int
    nVox         :: Int
    nGrid        :: Int
    nPoint       :: Int
    nCha         :: Int
    batch_size   :: Int
    nodes        :: Matrix{T}
    weights2     :: Vector{T}
    csm_host     :: Matrix{Complex{T}}
    mask_host    :: Vector{Int32}
    mask_idx     :: CuVector{Int32}
    primary_input:: CuVector{Complex{T}}
    primary_sum  :: CuVector{Complex{T}}
    host_input   :: Vector{Complex{T}}
    host_coil    :: Vector{Complex{T}}
    host_signal  :: Vector{Complex{T}}
    host_signals :: Vector{Vector{Complex{T}}}
    host_outputs :: Vector{Vector{Complex{T}}}
    completed    :: Vector{Nothing}
    released     :: Bool
end


mutable struct RankSharedNormalOp{T,F,S<:AbstractVector{Complex{T}},State<:RankSharedState{T}} <: AbstractLinearOperator{Complex{T}}
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

LinearOperators.storage_type(op::RankSharedNormalOp) = typeof(op.Mv)
Base.eltype(::RankSharedNormalOp{T}) where {T} = Complex{T}


function fold_rank_shared_phases!(q::Matrix{Complex{T}}, kspha, nodes, shape, nVox, center) where {T}
    # Use the same GPU pointwise phase/normalization operations as the resident
    # operator, but upload a bounded number of temporal columns at a time.
    zeroth_phase = CuArray(vec(kspha[1:1, :, :]))
    zeroth = exp.(Complex{T}(0, T(2π)) .* zeroth_phase)
    CUDA.unsafe_free!(zeroth_phase)
    phase = zeroth
    if center
        offsets = T[iseven(n) ? -0.5 : 0 for n in shape]
        center_phase = CuArray(vec(sum(nodes .* reshape(offsets, :, 1, 1); dims=1)))
        correction = exp.(Complex{T}(0, T(2π)) .* center_phase)
        phase = zeroth .* correction
        CUDA.unsafe_free!(center_phase)
        CUDA.unsafe_free!(correction)
        CUDA.unsafe_free!(zeroth)
    end
    try
        scale = inv(sqrt(T(nVox)))
        for r in axes(q, 2)
            column = CuArray(@view q[:, r:r])
            try
                column .*= scale .* reshape(phase, :, 1)
                copyto!(q, (r-1)*size(q,1)+1, column, 1, size(q,1))
            finally
                CUDA.unsafe_free!(column)
            end
        end
        CUDA.synchronize()
    finally
        CUDA.unsafe_free!(phase)
    end
    return nothing
end


function rank_shared_ranges(n::Int, nWorker::Int; primary_worker::Union{Nothing,Int}=1)
    primary_worker === nothing && return balanced_channel_ranges(n, nWorker)
    1 <= primary_worker <= nWorker || throw(ArgumentError("invalid primary worker"))
    nWorker == 1 && return [1:n]
    # Reserve CG memory only when the primary also performs encoding.
    counts = [length(r) for r in balanced_channel_ranges(n-1, nWorker-1)]
    insert!(counts, primary_worker, 1)
    ranges = UnitRange{Int}[]
    first_rank = 1
    for count in counts
        push!(ranges, first_rank:(first_rank+count-1))
        first_rank += count
    end
    return ranges
end


function rank_shared_channel_ranges(nCha, ranks, nVox, nPoint, nGrid; primary_worker::Union{Nothing,Int}=1)
    nWorker = length(ranks)
    counts = ones(Int, nWorker)
    coil_bytes = nVox*sizeof(ComplexF32)
    # Relative byte costs suffice here; all factors use the same precision.
    costs = [length(r)*(nVox+nPoint)*sizeof(ComplexF32)+coil_bytes for r in ranks]
    # Four CG state vectors, RHS/output allowance and primary image reduction.
    if primary_worker !== nothing
        costs[primary_worker] += (6nGrid+2nVox)*sizeof(ComplexF32)
    end
    for _ in (nWorker+1):nCha
        worker = argmin(costs)
        counts[worker] += 1
        costs[worker] += coil_bytes
    end
    ranges = UnitRange{Int}[]
    first_channel = 1
    for count in counts
        push!(ranges, first_channel:(first_channel+count-1))
        first_channel += count
    end
    return ranges
end


"""
    rank_shared_operator(q, basis, csm, mask_idx, nodes, grid_size, L; gpus, ...)

Restore an already phased and normalized shared representation from host RAM.
The host arrays are also the checkpoint masters. No full factor or CSM is
uploaded to the primary GPU; `normalOperator` lazily partitions rank columns
and coil columns among its workers. `gpus` selects encoding workers; set
`primary_gpu` outside that list to reserve a separate device for CG and image
transfers. This does not truncate either basis.
"""
function rank_shared_operator(
    q::Matrix{Complex{T}}, basis::Matrix{Complex{T}}, csm::Matrix{Complex{T}},
    mask_idx::Vector{Int32}, nodes::Array{T,3}, grid_size::Tuple, L::Int;
    gpus::Vector{Int}, primary_gpu::Int=first(gpus),
    sample_batch_size::Int=12_000_000, verbose::Bool=false,
) where {T<:AbstractFloat}
    isempty(gpus) && throw(ArgumentError("gpus must not be empty"))
    allunique(gpus) || throw(ArgumentError("GPU ids must be distinct"))
    sample_batch_size > 0 || throw(ArgumentError("sample_batch_size must be positive"))
    nSam, nDyn = size(nodes, 2), size(nodes, 3)
    nVox, nCha = size(csm)
    size(q, 1) == nSam*nDyn || throw(DimensionMismatch("temporal rows and nodes differ"))
    size(basis) == (nVox, size(q, 2)) || throw(DimensionMismatch("shared factor dimensions differ"))
    length(mask_idx) == nVox || throw(DimensionMismatch("mask and factor rows differ"))
    size(q, 2) > 0 || throw(ArgumentError("the shared basis is empty"))
    CUDA.device!(primary_gpu)
    empty_vector = CuVector{Complex{T}}(undef, 0)
    workspace = HighOrderLowRankWorkspace((CuVector{Complex{T}}(undef, 0) for _ in 1:7)...)
    backend = HighOrderNormalBackend(:rank, copy(gpus), verbose, nothing, nothing, sample_batch_size, primary_gpu)
    P = AbstractNFFTPlan{T}
    return HighOrderLowRankOp{T,Nothing,typeof(ctprod!),P,typeof(q),typeof(basis),typeof(csm),typeof(workspace),typeof(empty_vector)}(
        nSam, nVox, nCha, nDyn, L, :shared, :gpu, 1, nothing, q, basis, csm,
        nothing, P[], nodes, mask_idx, grid_size, workspace, backend,
        nSam*nDyn*nCha, prod(grid_size), false, false, prod!, nothing, ctprod!,
        0, 0, 0, empty_vector, empty_vector)
end


function kernel_rank_scatter!(grid, x, basis, r, coil, mask, nVox)
    v = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds grid[mask[v]] = x[v] * conj(basis[v, r]) * coil[v]
    end
    return nothing
end

function kernel_rank_accumulate!(x, grid, basis, r, coil, mask, nVox)
    v = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds x[v] += grid[mask[v]] * conj(coil[v]) * basis[v, r]
    end
    return nothing
end


function release_rank_shared_shard!(shard::RankSharedShard)
    shard.released && return nothing
    shard.released = true
    CUDA.device!(shard.gpu_id)
    CUDA.device_synchronize(; blocking=true)
    shard.nfftplan = nothing
    for field in (:q, :basis, :csm, :coil, :mask_idx, :weights2, :sqrt_weights)
        CUDA.unsafe_free!(getproperty(shard, field))
    end
    for field in fieldnames(typeof(shard.workspace))
        CUDA.unsafe_free!(getproperty(shard.workspace, field))
    end
    return nothing
end


function RankSharedNormalOp(parent::HighOrderLowRankOp{T}) where {T<:AbstractFloat}
    backend = parent.normal_backend
    backend.distribution === :rank || throw(ArgumentError("expected rank distribution"))
    nWorker = min(length(backend.gpus), size(parent.q, 2), parent.nCha)
    gpus = backend.gpus[1:nWorker]
    primary_worker = findfirst(==(backend.primary_gpu), gpus)
    ranks = rank_shared_ranges(size(parent.q, 2), nWorker; primary_worker)
    nPoint, nGrid = parent.nSam*parent.nDyn, prod(parent.grid_size)
    channels = rank_shared_channel_ranges(parent.nCha, ranks, parent.nVox, nPoint, nGrid; primary_worker)
    batch = min(nPoint, backend.sample_batch_size)
    rows = 1:batch
    nodes = reshape(parent.nfft_traj, size(parent.nfft_traj, 1), nPoint)
    dims = parent.grid_size[end] == 1 ? parent.grid_size[1:2] : parent.grid_size
    workers = DistributedGPUWorker[]
    shards_any = Vector{Any}(undef, nWorker)
    warn_if_insufficient_gpu_worker_threads(nWorker; operation=:rank_shared_operator)
    try
        for gpu in gpus
            push!(workers, DistributedGPUWorker(gpu))
        end
        run_on_workers!(shards_any, workers; phase=:rank_shared_setup) do i
            plan = plan_nfft(CuArray, @view(nodes[:, rows]), dims; m=3, σ=1.25)
            workspace = MultiGPUHighOrderNormalWorkspace{T}(
                CUDA.zeros(Complex{T}, nGrid), CUDA.zeros(Complex{T}, nGrid),
                CuVector{Complex{T}}(undef, batch), CuVector{Complex{T}}(undef, batch),
                CuVector{Complex{T}}(undef, batch), CuVector{Complex{T}}(undef, parent.nVox),
                CUDA.zeros(Complex{T}, parent.nVox))
            RankSharedShard(gpus[i], ranks[i], channels[i],
                CuArray(@view(parent.q[:, ranks[i]])), CuArray(@view(parent.basis[:, ranks[i]])),
                CuArray(@view(parent.csm[:, channels[i]])), CuVector{Complex{T}}(undef, parent.nVox),
                CuArray(parent.mask_idx), CuVector{T}(undef, batch), CuVector{Complex{T}}(undef, batch),
                plan, parent.grid_size, rows, workspace, false)
        end
    catch
        if !isempty(workers)
            done = Vector{Nothing}(undef, length(workers))
            try
                run_on_workers!(done, workers; phase=:rank_shared_setup_cleanup) do i
                    if isassigned(shards_any, i)
                        release_rank_shared_shard!(shards_any[i])
                    else
                        CUDA.device_synchronize(; blocking=true)
                    end
                    nothing
                end
            finally
                shutdown_distributed_workers!(workers)
                GC.gc(false)
            end
        end
        rethrow()
    end
    S = typeof(first(shards_any))
    shards = Vector{S}(shards_any)
    CUDA.device!(backend.primary_gpu)
    state = RankSharedState(workers, shards, backend.primary_gpu, parent.nVox, nGrid, nPoint,
        parent.nCha, batch, nodes, ones(T, nPoint), parent.csm, parent.mask_idx,
        CuArray(parent.mask_idx), CuVector{Complex{T}}(undef, parent.nVox),
        CuVector{Complex{T}}(undef, parent.nVox), Vector{Complex{T}}(undef, parent.nVox),
        Vector{Complex{T}}(undef, parent.nVox), Vector{Complex{T}}(undef, batch),
        [Vector{Complex{T}}(undef, batch) for _ in 1:nWorker],
        [Vector{Complex{T}}(undef, parent.nVox) for _ in 1:nWorker],
        Vector{Nothing}(undef, nWorker), false)
    product = (y, x) -> rank_shared_normal!(y, state, x)
    empty_vector = CuVector{Complex{T}}(undef, 0)
    backend.verbose && @info "Rank-sharded shared operator ready" gpus primary_gpu=backend.primary_gpu ranks channels batch nPoint nVox=parent.nVox
    return RankSharedNormalOp(nGrid, nGrid, false, true, product, nothing, nothing,
        0, 0, 0, empty_vector, empty_vector, state)
end


function ensure_rank_shared_backend!(parent::HighOrderLowRankOp{T}, weights2::Union{Nothing,Vector{T}}=nothing) where {T}
    backend = parent.normal_backend
    cached = backend.operator
    if cached === nothing
        cached = RankSharedNormalOp(parent)
        backend.operator = cached
    end
    cached isa RankSharedNormalOp || error("invalid rank-shared backend")
    cached.state.released && error("the rank-shared backend was released")
    if weights2 !== nothing
        if backend.weights2 === nothing
            copyto!(cached.state.weights2, weights2)
            backend.weights2 = copy(weights2)
        else
            isequal(backend.weights2, weights2) || throw(ArgumentError(
                "the owned rank-shared backend already has different weights; close it before rebuilding"))
        end
    end
    return cached
end


function highorder_normal_view(parent::HighOrderLowRankOp{T}, normal::RankSharedNormalOp{T}) where {T}
    CUDA.device!(normal.state.primary_gpu)
    product = (y, x) -> rank_shared_normal!(y, normal.state, x)
    empty_vector = CuVector{Complex{T}}(undef, 0)
    n = prod(parent.grid_size)
    return HighOrderLowRankNormalOp(n, n, false, true, product, nothing, nothing,
        0, 0, 0, empty_vector, empty_vector, parent)
end


function rank_shared_input!(state::RankSharedState{T}, x) where {T}
    state.released && error("rank-shared backend was released")
    length(x) in (state.nVox, state.nGrid) || throw(DimensionMismatch("invalid image length"))
    CUDA.device!(state.primary_gpu)
    if x isa CuArray
        CUDA.deviceid(CUDA.device(x)) == state.primary_gpu || throw(ArgumentError("image must reside on the primary GPU"))
        if length(x) == state.nGrid
            @cuda threads=256 blocks=cld(state.nVox, 256) kernel_gather_masked_image!(
                state.primary_input, x, state.mask_idx, state.nVox)
            copyto!(state.host_input, state.primary_input)
        else
            copyto!(state.host_input, x)
        end
    elseif length(x) == state.nGrid
        for v in eachindex(state.mask_host)
            @inbounds state.host_input[v] = x[state.mask_host[v]]
        end
    else
        copyto!(state.host_input, x)
    end
    return nothing
end


function rank_shared_batch!(state::RankSharedState, rows)
    state.released && error("rank-shared backend was released")
    run_on_workers!(state.completed, state.workers; phase=:rank_shared_batch) do i
        shard = state.shards[i]
        if shard.rows != rows
            update_nfft_nodes!(something(shard.nfftplan), Matrix(@view state.nodes[:, rows]))
            shard.rows = rows
        end
        copyto!(shard.weights2, 1, state.weights2, first(rows), length(rows))
        nothing
    end
    return nothing
end


function rank_shared_coil!(state::RankSharedState, channel)
    # The immutable checkpoint master already has the same whitened CSM.
    # Avoid downloading it from an owner whose previous adjoint is still queued.
    copyto!(state.host_coil, 1, state.csm_host, (channel-1)*state.nVox+1, state.nVox)
    return nothing
end


function rank_shared_coil!(shard::RankSharedShard, host_coil, channel)
    if channel in shard.channels
        # Reuse this device's inventory without another PCIe upload.
        copyto!(shard.coil, @view shard.csm[:, channel-first(shard.channels)+1])
    else
        copyto!(shard.coil, host_coil)
    end
    return nothing
end


function rank_shared_local_forward!(shard::RankSharedShard{T}) where {T}
    w, rows = shard.workspace, shard.rows
    signal = @view w.k_signal[1:length(rows)]
    channel = @view w.k_channel[1:length(rows)]
    fill!(channel, zero(Complex{T}))
    dims = shard.grid_size[end] == 1 ? shard.grid_size[1:2] : shard.grid_size
    image = reshape(w.grid_forward, dims)
    for r in axes(shard.basis, 2)
        @cuda threads=256 blocks=cld(length(w.x_input), 256) kernel_rank_scatter!(
            w.grid_forward, w.x_input, shard.basis, r, shard.coil, shard.mask_idx, length(w.x_input))
        mul!(signal, something(shard.nfftplan), image)
        @views @. channel += signal * shard.q[rows, r]
    end
    return channel
end


function rank_shared_local_adjoint!(shard::RankSharedShard)
    w, rows = shard.workspace, shard.rows
    channel = @view w.k_channel[1:length(rows)]
    weighted = @view w.k_weighted[1:length(rows)]
    dims = shard.grid_size[end] == 1 ? shard.grid_size[1:2] : shard.grid_size
    image = reshape(w.grid_adjoint, dims)
    for r in axes(shard.basis, 2)
        @views @. weighted = channel * conj(shard.q[rows, r])
        mul!(image, adjoint(something(shard.nfftplan)), weighted)
        @cuda threads=256 blocks=cld(length(w.x_output), 256) kernel_rank_accumulate!(
            w.x_output, w.grid_adjoint, shard.basis, r, shard.coil, shard.mask_idx, length(w.x_output))
    end
    return nothing
end


function rank_shared_sum_range!(output, inputs::NTuple{N}, rows) where {N}
    # Each element keeps the original worker order, including when output
    # aliases the first input. Threads own disjoint contiguous element ranges.
    @inbounds @simd for p in rows
        value = inputs[1][p]
        for i in 2:N
            value += inputs[i][p]
        end
        output[p] = value
    end
    return nothing
end


function rank_shared_sum!(output, inputs, n; ntasks=4)
    n == 0 && return nothing
    sources = Tuple(inputs)
    nTask = min(ntasks, Threads.nthreads(:default), cld(n, 262_144))
    if nTask == 1
        rank_shared_sum_range!(output, sources, 1:n)
    else
        @sync for i in 1:nTask
            rows = (fld((i-1)*n,nTask)+1):fld(i*n,nTask)
            Threads.@spawn rank_shared_sum_range!(output, sources, rows)
        end
    end
    return nothing
end


function rank_shared_sum_signal!(state::RankSharedState, n)
    rank_shared_sum!(state.host_signal, state.host_signals, n)
    return nothing
end


function rank_shared_collect_image!(output, state::RankSharedState{T}) where {T}
    run_on_workers!(state.completed, state.workers; phase=:rank_shared_image_download) do i
        copyto!(state.host_outputs[i], state.shards[i].workspace.x_output)
        nothing
    end
    total = state.host_outputs[1]
    rank_shared_sum!(total, state.host_outputs, length(total))
    CUDA.device!(state.primary_gpu)
    if length(output) == state.nVox
        copyto!(output, total)
    elseif length(output) == state.nGrid
        fill!(output, zero(Complex{T}))
        if output isa CuArray
            copyto!(state.primary_sum, total)
            @cuda threads=256 blocks=cld(state.nVox, 256) kernel_scatter_masked_image!(
                output, state.primary_sum, state.mask_idx, state.nVox)
            CUDA.synchronize()
        else
            for v in eachindex(total)
                @inbounds output[state.mask_host[v]] = total[v]
            end
        end
    else
        throw(DimensionMismatch("invalid adjoint image length"))
    end
    return output
end


function rank_shared_normal!(output, state::RankSharedState{T}, x) where {T}
    rank_shared_input!(state, x)
    run_on_workers!(state.completed, state.workers; phase=:rank_shared_normal_input) do i
        w = state.shards[i].workspace
        copyto!(w.x_input, state.host_input)
        fill!(w.x_output, zero(Complex{T}))
        nothing
    end
    for first_point in 1:state.batch_size:state.nPoint
        rows = first_point:min(first_point+state.batch_size-1, state.nPoint)
        n = length(rows)
        rank_shared_batch!(state, rows)
        for c in 1:state.nCha
            rank_shared_coil!(state, c)
            run_on_workers!(state.completed, state.workers; phase=:rank_shared_forward) do i
                shard = state.shards[i]
                rank_shared_coil!(shard, state.host_coil, c)
                signal = rank_shared_local_forward!(shard)
                copyto!(state.host_signals[i], 1, signal, 1, n)
                nothing
            end
            # A=sum_r A_r. Reduce A_r*x BEFORE weighting and A_r' so the
            # normal includes all A_r' W² A_s cross-rank terms.
            rank_shared_sum_signal!(state, n)
            run_on_workers!(state.completed, state.workers; phase=:rank_shared_normal_adjoint) do i
                shard = state.shards[i]
                signal = @view shard.workspace.k_channel[1:n]
                copyto!(signal, 1, state.host_signal, 1, n)
                @views @. signal *= shard.weights2[1:n]
                rank_shared_local_adjoint!(shard)
                nothing
            end
        end
    end
    return rank_shared_collect_image!(output, state)
end


function rank_shared_adjoint!(output, parent::HighOrderLowRankOp{T}, data;
    sqrt_weights=nothing,
) where {T}
    state = ensure_rank_shared_backend!(parent).state
    return rank_shared_adjoint!(output, state, data; sqrt_weights)
end

function rank_shared_adjoint!(output, state::RankSharedState{T}, data;
    sqrt_weights=nothing,
) where {T}
    state.released && error("rank-shared backend was released")
    length(data) == state.nPoint*state.nCha || throw(DimensionMismatch("invalid adjoint data length"))
    sqrt_weights === nothing || length(sqrt_weights) == state.nPoint || throw(DimensionMismatch("invalid sqrt weights"))
    host_data = reshape(data isa CuArray ? Array(data) : data, state.nPoint, state.nCha)
    host_weights = sqrt_weights === nothing ? nothing :
        (sqrt_weights isa Vector{Complex{T}} ? sqrt_weights : Complex{T}.(Array(sqrt_weights)))
    run_on_workers!(state.completed, state.workers; phase=:rank_shared_adjoint_zero) do i
        fill!(state.shards[i].workspace.x_output, zero(Complex{T}))
        nothing
    end
    for first_point in 1:state.batch_size:state.nPoint
        rows = first_point:min(first_point+state.batch_size-1, state.nPoint)
        n = length(rows)
        rank_shared_batch!(state, rows)
        if host_weights !== nothing
            run_on_workers!(state.completed, state.workers; phase=:rank_shared_sqrt_weights) do i
                copyto!(state.shards[i].sqrt_weights, 1, host_weights, first(rows), n)
                nothing
            end
        end
        for c in 1:state.nCha
            rank_shared_coil!(state, c)
            copyto!(@view(state.host_signal[1:n]), @view(host_data[rows, c]))
            run_on_workers!(state.completed, state.workers; phase=:rank_shared_rhs) do i
                shard = state.shards[i]
                rank_shared_coil!(shard, state.host_coil, c)
                signal = @view shard.workspace.k_channel[1:n]
                copyto!(signal, 1, state.host_signal, 1, n)
                if host_weights !== nothing
                    @views @. signal *= shard.sqrt_weights[1:n]
                    @views @. signal *= conj(shard.sqrt_weights[1:n])
                end
                rank_shared_local_adjoint!(shard)
                nothing
            end
        end
    end
    return rank_shared_collect_image!(output, state)
end


function rank_shared_forward!(output, parent::HighOrderLowRankOp{T}, x) where {T}
    state = ensure_rank_shared_backend!(parent).state
    return rank_shared_forward!(output, state, x)
end

function rank_shared_forward!(output, state::RankSharedState{T}, x) where {T}
    length(output) == state.nPoint*state.nCha || throw(DimensionMismatch("invalid forward output length"))
    rank_shared_input!(state, x)
    run_on_workers!(state.completed, state.workers; phase=:rank_shared_forward_input) do i
        copyto!(state.shards[i].workspace.x_input, state.host_input)
        nothing
    end
    result = reshape(output, state.nPoint, state.nCha)
    for first_point in 1:state.batch_size:state.nPoint
        rows = first_point:min(first_point+state.batch_size-1, state.nPoint)
        n = length(rows)
        rank_shared_batch!(state, rows)
        for c in 1:state.nCha
            rank_shared_coil!(state, c)
            run_on_workers!(state.completed, state.workers; phase=:rank_shared_forward) do i
                shard = state.shards[i]
                rank_shared_coil!(shard, state.host_coil, c)
                signal = rank_shared_local_forward!(shard)
                copyto!(state.host_signals[i], 1, signal, 1, n)
                nothing
            end
            rank_shared_sum_signal!(state, n)
            CUDA.device!(state.primary_gpu)
            copyto!(output, (c-1)*state.nPoint+first(rows), state.host_signal, 1, n)
        end
    end
    return output
end


function release_multi_gpu_normal_operator!(normal::RankSharedNormalOp)
    state = normal.state
    state.released && return nothing
    state.released = true
    try
        run_on_workers!(state.completed, state.workers; phase=:rank_shared_release) do i
            release_rank_shared_shard!(state.shards[i])
            nothing
        end
    finally
        shutdown_distributed_workers!(state.workers)
        CUDA.device!(state.primary_gpu)
        CUDA.device_synchronize(; blocking=true)
        for field in (:mask_idx, :primary_input, :primary_sum)
            CUDA.unsafe_free!(getproperty(state, field))
        end
        empty!(state.workers)
        empty!(state.shards)
        GC.gc(false)
    end
    return nothing
end

Base.close(normal::RankSharedNormalOp) = release_multi_gpu_normal_operator!(normal)
