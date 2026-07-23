
function split_voxel_ranges(
    nVox::Int,
    gpus::AbstractVector{Int},
)
    @assert nVox > 0
    @assert !isempty(gpus)

    nPart = min(length(gpus), nVox)
    base_size = nVox ÷ nPart
    remainder = nVox % nPart

    ranges = Vector{UnitRange{Int}}(undef, nPart)

    first_voxel = 1

    for i = 1:nPart
        local_size = base_size + (i <= remainder ? 1 : 0)
        last_voxel = first_voxel + local_size - 1

        ranges[i] = first_voxel:last_voxel
        first_voxel = last_voxel + 1
    end

    return ranges
end


mutable struct DistributedRSVDShard{T<:AbstractFloat}
    gpu_id      :: Int
    voxels      :: UnitRange{Int}

    fieldmap    :: CuVector{T}
    bf          :: CuMatrix{T}
    times       :: CuVector{T}
    kspha_t     :: CuMatrix{T}

    rng         :: CUDA.RNG
    omega       :: CuMatrix{Complex{T}}
    W           :: CuMatrix{Complex{T}}
    Q           :: CuMatrix{Complex{T}}
    B_adj       :: CuMatrix{Complex{T}}
    gram        :: CuMatrix{Complex{T}}
    Z           :: CuMatrix{Complex{T}}
    v_scaled    :: CuMatrix{Complex{T}}

    released    :: Bool
end

struct DistributedRSVDWorkspace{T<:AbstractFloat}
    workers     :: Vector{DistributedGPUWorker}
    shards      :: Vector{DistributedRSVDShard{T}}
    kspha_t_host:: Matrix{T}

    nVox        :: Int
    nSam        :: Int
    L_total     :: Int
    L_rank      :: Int
end


mutable struct DistributedRSVDTiming
    n_calls          :: Int
    forward_time     :: Float64
    qr_time          :: Float64
    adjoint_gram_time:: Float64
    finalize_time    :: Float64
end


DistributedRSVDTiming() = DistributedRSVDTiming(0, 0.0, 0.0, 0.0, 0.0)


function reset_distributed_rsvd_timing!(timing::DistributedRSVDTiming)
    timing.n_calls = 0
    timing.forward_time = 0.0
    timing.qr_time = 0.0
    timing.adjoint_gram_time = 0.0
    timing.finalize_time = 0.0
    return timing
end


function distributed_rsvd_total_time(timing::DistributedRSVDTiming)
    return timing.forward_time +
           timing.qr_time +
           timing.adjoint_gram_time +
           timing.finalize_time
end


function DistributedRSVDWorkspace(
    fieldmap :: Vector{T},
    bf       :: Matrix{T},
    nSam     :: Int,
    L_total  :: Int,
    L_rank   :: Int,
    gpus     :: Vector{Int},
) where {T<:AbstractFloat}

    nVox = length(fieldmap)
    M = size(bf, 2)

    @assert size(bf, 1) == nVox
    @assert L_rank <= L_total

    ranges = split_voxel_ranges(nVox, gpus)
    warn_if_insufficient_gpu_worker_threads(length(gpus); operation=:distributed_rsvd)

    workers = DistributedGPUWorker[]

    try
        for gpu_id in gpus
            push!(workers, DistributedGPUWorker(gpu_id))
        end

        shards = Vector{DistributedRSVDShard{T}}(undef, length(workers))
        run_on_workers!(shards, workers) do i
            gpu_id = gpus[i]
            voxels = ranges[i]
            nLocal = length(voxels)

            DistributedRSVDShard{T}(
                gpu_id,
                voxels,
                CuArray(fieldmap[voxels]),
                CuArray(bf[voxels, :]),
                CUDA.zeros(T, nSam),
                CUDA.zeros(T, nSam, M),
                CUDA.RNG(0),
                CUDA.zeros(Complex{T}, nLocal, L_total),
                CUDA.zeros(Complex{T}, nSam, L_total),
                CUDA.zeros(Complex{T}, nSam, L_total),
                CUDA.zeros(Complex{T}, nLocal, L_total),
                CUDA.zeros(Complex{T}, L_total, L_total),
                CUDA.zeros(Complex{T}, L_total, L_rank),
                CUDA.zeros(Complex{T}, nLocal, L_rank),
                false,
            )
        end

        kspha_t_host = Matrix{T}(undef, nSam, M)
        return DistributedRSVDWorkspace(
            workers,
            shards,
            kspha_t_host,
            nVox,
            nSam,
            L_total,
            L_rank,
        )
    catch
        shutdown_distributed_workers!(workers)
        rethrow()
    end
end


function release_distributed_rsvd_shard!(shard::DistributedRSVDShard)
    shard.released && return nothing

    # Mark first so cleanup remains idempotent even if a CUDA error interrupts
    # one of the following frees. A failed CUDA context must not be reused.
    shard.released = true

    CUDA.unsafe_free!(shard.fieldmap)
    CUDA.unsafe_free!(shard.bf)
    CUDA.unsafe_free!(shard.times)
    CUDA.unsafe_free!(shard.kspha_t)
    CUDA.unsafe_free!(shard.omega)
    CUDA.unsafe_free!(shard.W)
    CUDA.unsafe_free!(shard.Q)
    CUDA.unsafe_free!(shard.B_adj)
    CUDA.unsafe_free!(shard.gram)
    CUDA.unsafe_free!(shard.Z)
    CUDA.unsafe_free!(shard.v_scaled)

    return nothing
end


function distributed_rsvd_forward!(
    workspace :: DistributedRSVDWorkspace{T},
    times     :: Vector{T},
    kspha     :: Matrix{T};
    omega     :: Union{Nothing, AbstractMatrix{Complex{T}}} = nothing,
    seed      :: Int = 0,
) where {T<:AbstractFloat}

    @assert length(times) == workspace.nSam
    @assert size(kspha, 2) == workspace.nSam
    @assert size(kspha, 1) == size(workspace.kspha_t_host, 2)
    if omega !== nothing @assert size(omega) == (workspace.nVox, workspace.L_total) end

    permutedims!(workspace.kspha_t_host, kspha, (2, 1))

    nShard = length(workspace.shards)
    partial_W = Vector{Matrix{Complex{T}}}(undef, nShard)

    run_on_workers!(partial_W, workspace.workers) do i
        shard = workspace.shards[i]

        copyto!(shard.times, times)
        copyto!(shard.kspha_t, workspace.kspha_t_host)

        if omega === nothing
            shard_seed = seed + 1_000_003 * (i - 1)
            Random.seed!(shard.rng, shard_seed)
            randn!(shard.rng, shard.omega)
        else
            omega_local = omega[shard.voxels, :]
            copyto!(shard.omega, omega_local)
        end

        CUDA.@sync run_kernel_rsvd_forward!(
            shard.W,
            shard.omega,
            shard.times,
            shard.fieldmap,
            shard.bf,
            shard.kspha_t;
            threads=128,
            kspha_transposed=true,
        )

        Array(shard.W)
    end

    W = zeros(Complex{T}, workspace.nSam, workspace.L_total)

    for W_local in partial_W
        W .+= W_local
    end

    return W
end

function distributed_rsvd_adjoint_gram!(
    workspace :: DistributedRSVDWorkspace{T},
    Q         :: Matrix{Complex{T}},
) where {T<:AbstractFloat}

    @assert size(Q) == (workspace.nSam, workspace.L_total)

    nShard = length(workspace.shards)
    local_grams = Vector{Matrix{Complex{T}}}(undef, nShard)

    run_on_workers!(local_grams, workspace.workers) do i
        shard = workspace.shards[i]
        copyto!(shard.Q, Q)

        CUDA.@sync begin
            run_kernel_rsvd_adjoint!(
                shard.B_adj,
                shard.Q,
                shard.times,
                shard.fieldmap,
                shard.bf,
                shard.kspha_t;
                threads=256,
                kspha_transposed=true,
            )
            mul!(shard.gram, adjoint(shard.B_adj), shard.B_adj)
        end

        Array(shard.gram)
    end

    gram = zeros(Complex{T}, workspace.L_total, workspace.L_total)

    for local_gram in local_grams
        gram .+= local_gram
    end

    gram = (gram + adjoint(gram)) * T(0.5)
    return gram
end

function finalize_distributed_rsvd_gram!(
    workspace :: DistributedRSVDWorkspace{T},
    Q         :: Matrix{Complex{T}},
    gram      :: Matrix{Complex{T}},
) where {T<:AbstractFloat}

    eig = eigen(Hermitian(gram))

    order = sortperm(real.(eig.values); rev=true)

    values = T.(real.(eig.values[order]))
    vectors = eig.vectors[:, order]

    @assert all(isfinite, values)
    @assert all(isfinite, vectors)

    λmax = maximum(abs, values)

    λmax > zero(T) || error("Degenerate distributed rSVD Gram matrix")

    negative_tol = T(10) * eps(T) * T(workspace.L_total) * λmax

    minimum(values) >= -negative_tol || error("Distributed rSVD Gram matrix has " * "significant negative eigenvalues")

    values .= max.(values, zero(T))

    Z_cpu = Matrix(@view vectors[:, 1:workspace.L_rank])

    # U = QZ
    u_trunc = Q * Z_cpu

    completed = Vector{Nothing}(undef, length(workspace.shards))

    run_on_workers!(completed, workspace.workers) do i
        shard = workspace.shards[i]
        copyto!(shard.Z, Z_cpu)
        CUDA.@sync mul!(shard.v_scaled, shard.B_adj, shard.Z)
        nothing
    end

    total_energy = T(sum(@view values[1:workspace.L_rank]))

    return u_trunc, total_energy
end


function perform_rsvd_multi_gpu!(
    workspace :: DistributedRSVDWorkspace{T},
    times     :: Vector{T},
    kspha     :: Matrix{T};
    seed      :: Int = 0,
    omega             = nothing,
    timing    :: Union{Nothing, DistributedRSVDTiming} = nothing,
) where {T<:AbstractFloat}

    t0 = time_ns()
    W = distributed_rsvd_forward!(workspace, times, kspha; seed, omega)
    forward_time = (time_ns() - t0) * 1e-9

    t0 = time_ns()
    qr_W = qr(W)

    Q_seed = Matrix{Complex{T}}(I, workspace.nSam, workspace.L_total)

    Q = Matrix(qr_W.Q * Q_seed)
    qr_time = (time_ns() - t0) * 1e-9

    t0 = time_ns()
    gram = distributed_rsvd_adjoint_gram!(workspace, Q)
    adjoint_gram_time = (time_ns() - t0) * 1e-9

    t0 = time_ns()
    result = finalize_distributed_rsvd_gram!(workspace, Q, gram)
    finalize_time = (time_ns() - t0) * 1e-9

    if timing !== nothing
        timing.n_calls += 1
        timing.forward_time += forward_time
        timing.qr_time += qr_time
        timing.adjoint_gram_time += adjoint_gram_time
        timing.finalize_time += finalize_time
    end

    return result
end


function gather_distributed_v_scaled(
    workspace::DistributedRSVDWorkspace{T},
) where {T<:AbstractFloat}

    nShard = length(workspace.shards)

    local_parts = Vector{Matrix{Complex{T}}}(undef, nShard)
    run_on_workers!(local_parts, workspace.workers) do i
        Array(workspace.shards[i].v_scaled)
    end

    v_scaled = zeros(Complex{T}, workspace.nVox, workspace.L_rank)

    for (i, shard) in enumerate(workspace.shards)
        v_scaled[shard.voxels, :] .= local_parts[i]
    end

    return v_scaled
end
