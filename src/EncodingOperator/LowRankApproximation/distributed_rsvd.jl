
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
    kspha       :: CuMatrix{T}

    omega       :: CuMatrix{Complex{T}}
    W           :: CuMatrix{Complex{T}}
    Q           :: CuMatrix{Complex{T}}
    B_adj       :: CuMatrix{Complex{T}}
    gram        :: CuMatrix{Complex{T}}
    v_scaled    :: CuMatrix{Complex{T}}
end

struct DistributedRSVDWorkspace{T<:AbstractFloat}
    shards      :: Vector{DistributedRSVDShard{T}}
    nVox        :: Int
    nSam        :: Int
    L_total     :: Int
    L_rank      :: Int
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
    tasks = Task[]

    for (gpu_id, voxels) in zip(gpus, ranges)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(gpu_id)

            nLocal = length(voxels)

            DistributedRSVDShard{T}(
                gpu_id,
                voxels,
                CuArray(fieldmap[voxels]),
                CuArray(bf[voxels, :]),
                CUDA.zeros(T, nSam),
                CUDA.zeros(T, M, nSam),
                CUDA.zeros(Complex{T}, nLocal, L_total),
                CUDA.zeros(Complex{T}, nSam, L_total),
                CUDA.zeros(Complex{T}, nSam, L_total),
                CUDA.zeros(Complex{T}, nLocal, L_total),
                CUDA.zeros(Complex{T}, L_total, L_total),
                CUDA.zeros(Complex{T}, nLocal, L_rank),
            )
        end)
    end

    shards = DistributedRSVDShard{T}[fetch(task) for task in tasks]

    return DistributedRSVDWorkspace(shards, nVox, nSam, L_total, L_rank)
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
    if omega !== nothing @assert size(omega) == (workspace.nVox, workspace.L_total) end

    partial_W = Vector{Matrix{Complex{T}}}(undef, length(workspace.shards))

    tasks = Task[]

    for (i, shard) in enumerate(workspace.shards)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shard.gpu_id)

            copyto!(shard.times, times)
            copyto!(shard.kspha, kspha)

            if omega === nothing
                CUDA.seed!(seed)
                randn!(shard.omega)
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
                shard.kspha;
                threads=128,
            )

            partial_W[i] = Array(shard.W)
        end)
    end

    foreach(fetch, tasks)

    W = zeros(Complex{T}, workspace.nSam, workspace.L_total)

    for W_local in partial_W
        W .+= W_local
    end

    return W
end

function distributed_rsvd_adjoint!(
    workspace :: DistributedRSVDWorkspace{T},
    Q         :: Matrix{Complex{T}},
) where {T<:AbstractFloat}

    @assert size(Q) == (workspace.nSam, workspace.L_total)

    nShard = length(workspace.shards)

    local_B = Vector{Matrix{Complex{T}}}(undef, nShard)
    local_gram = Vector{Matrix{Complex{T}}}(undef, nShard)

    tasks = Task[]

    for (i, shard) in enumerate(workspace.shards)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shard.gpu_id)

            copyto!(shard.Q, Q)

            CUDA.@sync begin
                run_kernel_rsvd_adjoint_warp!(
                    shard.B_adj,
                    shard.Q,
                    shard.times,
                    shard.fieldmap,
                    shard.bf,
                    shard.kspha;
                    threads=256,
                )

                mul!(shard.gram, adjoint(shard.B_adj), shard.B_adj)
            end

            local_B[i] = Array(shard.B_adj)
            local_gram[i] = Array(shard.gram)
        end)
    end

    foreach(fetch, tasks)

    B_adj = zeros(Complex{T}, workspace.nVox, workspace.L_total)

    gram = zeros(Complex{T}, workspace.L_total, workspace.L_total)

    for (i, shard) in enumerate(workspace.shards)
        B_adj[shard.voxels, :] .= local_B[i]
        gram .+= local_gram[i]
    end

    return B_adj, gram
end



function distributed_rsvd_adjoint_gram!(
    workspace :: DistributedRSVDWorkspace{T},
    Q         :: Matrix{Complex{T}},
) where {T<:AbstractFloat}

    @assert size(Q) == (workspace.nSam, workspace.L_total)

    local_grams = Vector{Matrix{Complex{T}}}(undef, length(workspace.shards))

    tasks = Task[]

    for (i, shard) in enumerate(workspace.shards)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shard.gpu_id)

            copyto!(shard.Q, Q)

            CUDA.@sync begin
                run_kernel_rsvd_adjoint_warp!(
                    shard.B_adj,
                    shard.Q,
                    shard.times,
                    shard.fieldmap,
                    shard.bf,
                    shard.kspha;
                    threads=256,
                )

                mul!(shard.gram, adjoint(shard.B_adj), shard.B_adj)
            end

            local_grams[i] = Array(shard.gram)
        end)
    end

    foreach(fetch, tasks)

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

    tasks = Task[]

    for shard in workspace.shards
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shard.gpu_id)

            Z_d = CuArray(Z_cpu)

            # Vscaled_g = B_g Z
            CUDA.@sync mul!(shard.v_scaled, shard.B_adj, Z_d)
        end)
    end

    foreach(fetch, tasks)

    total_energy = T(sum(@view values[1:workspace.L_rank]))

    return u_trunc, total_energy
end


function perform_rsvd_multi_gpu!(
    workspace :: DistributedRSVDWorkspace{T},
    times     :: Vector{T},
    kspha     :: Matrix{T};
    seed      :: Int = 0,
    omega     = nothing,
) where {T<:AbstractFloat}

    W = distributed_rsvd_forward!(workspace,  times, kspha; seed=seed, omega=omega)

    qr_W = qr(W)

    Q_seed = Matrix{Complex{T}}(I, workspace.nSam, workspace.L_total)

    Q = Matrix(qr_W.Q * Q_seed)

    gram = distributed_rsvd_adjoint_gram!(workspace, Q)

    return finalize_distributed_rsvd_gram!(workspace, Q, gram)
end


function gather_distributed_v_scaled(
    workspace::DistributedRSVDWorkspace{T},
) where {T<:AbstractFloat}

    nShard = length(workspace.shards)

    local_parts = Vector{Matrix{Complex{T}}}(undef, nShard)

    tasks = Task[]

    for (i, shard) in enumerate(workspace.shards)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shard.gpu_id)

            local_parts[i] = Array(shard.v_scaled)
        end)
    end

    foreach(fetch, tasks)

    v_scaled = zeros(Complex{T}, workspace.nVox, workspace.L_rank)

    for (i, shard) in enumerate(workspace.shards)
        v_scaled[shard.voxels, :] .= local_parts[i]
    end

    return v_scaled
end