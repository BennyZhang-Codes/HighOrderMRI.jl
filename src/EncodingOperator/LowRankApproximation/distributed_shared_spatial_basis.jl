mutable struct DistributedSharedBasisShard{T<:AbstractFloat}
    gpu_id      :: Int
    voxels      :: UnitRange{Int}

    basis       :: CuMatrix{Complex{T}}
    residual    :: CuMatrix{Complex{T}}

    small       :: CuMatrix{Complex{T}}
    gram        :: CuMatrix{Complex{T}}
    transform   :: CuMatrix{Complex{T}}
end


mutable struct DistributedSharedSpatialBasis{T<:AbstractFloat}
    shards      :: Vector{DistributedSharedBasisShard{T}}

    coeff       :: Array{Complex{T},3}

    rank        :: Int
    max_rank    :: Int
    L_rank      :: Int
    nVox        :: Int
    tol         :: T
    errors      :: Vector{T}
end

function DistributedSharedSpatialBasis(
    rsvd_workspace :: DistributedRSVDWorkspace{T},
    nDyn           :: Int,
    max_rank       :: Int,
    tol            :: T,
) where {T<:AbstractFloat}

    @assert nDyn > 0
    @assert max_rank > 0
    @assert max_rank <= rsvd_workspace.nVox
    @assert tol >= zero(T)

    L_rank = rsvd_workspace.L_rank

    tasks = Task[]

    for rsvd_shard in rsvd_workspace.shards
        push!(tasks, Threads.@spawn begin
            CUDA.device!(rsvd_shard.gpu_id)

            nLocal = length(rsvd_shard.voxels)

            DistributedSharedBasisShard{T}(
                rsvd_shard.gpu_id,
                rsvd_shard.voxels,
                CUDA.zeros(Complex{T}, nLocal, max_rank),
                CUDA.zeros(Complex{T}, nLocal, L_rank),
                CUDA.zeros(Complex{T}, max_rank, L_rank),
                CUDA.zeros(Complex{T}, L_rank, L_rank),
                CUDA.zeros(Complex{T}, L_rank, L_rank),
            )
        end)
    end

    shards = DistributedSharedBasisShard{T}[fetch(task) for task in tasks]

    coeff = zeros(Complex{T}, max_rank, L_rank, nDyn)

    return DistributedSharedSpatialBasis(shards, coeff, 0, max_rank, L_rank, rsvd_workspace.nVox, tol, zeros(T, nDyn))
end

function update_distributed_shared_basis!(
    shared        :: DistributedSharedSpatialBasis{T},
    rsvd_workspace:: DistributedRSVDWorkspace{T},
    dyn           :: Int,
    total_energy  :: T,
) where {T<:AbstractFloat}

    @assert 1 <= dyn <= size(shared.coeff, 3)
    @assert length(shared.shards) ==
        length(rsvd_workspace.shards)

    r = shared.rank
    L = shared.L_rank
    nShard = length(shared.shards)

    fill!(@view(shared.coeff[:, :, dyn]), zero(Complex{T}))

    # ----------------------------------------------------------
    # R_g = Vscaled_g
    # ----------------------------------------------------------
    tasks = Task[]

    for i = 1:nShard
        shared_shard = shared.shards[i]
        rsvd_shard = rsvd_workspace.shards[i]

        @assert shared_shard.gpu_id == rsvd_shard.gpu_id
        @assert shared_shard.voxels == rsvd_shard.voxels

        push!(tasks, Threads.@spawn begin
            CUDA.device!(shared_shard.gpu_id)

            copyto!(shared_shard.residual, rsvd_shard.v_scaled)
        end)
    end

    foreach(fetch, tasks)

    # ----------------------------------------------------------
    # 第一次投影：
    # C = Σ_g B_gᴴ V_g
    # R_g = V_g - B_g C
    # ----------------------------------------------------------
    if r > 0
        local_projection =
            Vector{Matrix{Complex{T}}}(undef, nShard)

        tasks = Task[]

        for i = 1:nShard
            shared_shard = shared.shards[i]
            rsvd_shard = rsvd_workspace.shards[i]

            push!(tasks, Threads.@spawn begin
                CUDA.device!(shared_shard.gpu_id)

                B = @view shared_shard.basis[:, 1:r]
                projection = @view shared_shard.small[1:r, :]

                CUDA.@sync mul!( projection, adjoint(B), rsvd_shard.v_scaled)

                local_projection[i] = Array(projection)
            end)
        end

        foreach(fetch, tasks)

        C = zeros(Complex{T}, r, L)

        for projection in local_projection
            C .+= projection
        end

        @views shared.coeff[1:r, :, dyn] .= C

        # ------------------------------------------------------
        # R_g -= B_g C
        # correction_g = B_gᴴ R_g
        # ------------------------------------------------------
        local_correction = Vector{Matrix{Complex{T}}}(undef, nShard)

        tasks = Task[]

        for i = 1:nShard
            shared_shard = shared.shards[i]

            push!(tasks, Threads.@spawn begin
                CUDA.device!(shared_shard.gpu_id)

                B = @view shared_shard.basis[:, 1:r]
                small = @view shared_shard.small[1:r, :]

                copyto!(small, C)

                CUDA.@sync begin
                    mul!(shared_shard.residual, B, small, -one(Complex{T}), one(Complex{T}))
                    mul!(small, adjoint(B), shared_shard.residual)
                end

                local_correction[i] = Array(small)
            end)
        end

        foreach(fetch, tasks)

        correction = zeros(Complex{T}, r, L)

        for local_value in local_correction
            correction .+= local_value
        end

        @views shared.coeff[1:r, :, dyn] .+= correction

        # ------------------------------------------------------
        # 第二次正交化：
        # R_g -= B_g correction
        # ------------------------------------------------------
        tasks = Task[]

        for shared_shard in shared.shards
            push!(tasks, Threads.@spawn begin
                CUDA.device!(shared_shard.gpu_id)

                B = @view shared_shard.basis[:, 1:r]
                small = @view shared_shard.small[1:r, :]

                copyto!(small, correction)

                CUDA.@sync mul!(shared_shard.residual, B, small, -one(Complex{T}), one(Complex{T}))
            end)
        end

        foreach(fetch, tasks)
    end

    if total_energy <= eps(T)
        shared.errors[dyn] = zero(T)
        return zero(T), 0
    end

    # ----------------------------------------------------------
    # G = Σ_g R_gᴴR_g
    # ----------------------------------------------------------
    local_grams =
        Vector{Matrix{Complex{T}}}(undef, nShard)

    tasks = Task[]

    for (i, shared_shard) in enumerate(shared.shards)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shared_shard.gpu_id)

            CUDA.@sync mul!(shared_shard.gram, adjoint(shared_shard.residual), shared_shard.residual)

            local_grams[i] = Array(shared_shard.gram)
        end)
    end

    foreach(fetch, tasks)

    gram = zeros(Complex{T}, L, L)

    for local_gram in local_grams
        gram .+= local_gram
    end

    gram .= (gram .+ adjoint(gram)) .* T(0.5)

    eig = eigen(Hermitian(gram))

    order = sortperm(real.(eig.values); rev=true)

    values = max.(T.(real.(eig.values[order])), zero(T))

    vectors = eig.vectors[:, order]

    if !isempty(values)
        numerical_threshold = eps(T) * T(L) * max(maximum(values), one(T))
        values[values .< numerical_threshold] .= zero(T)
    end

    allowed_energy = shared.tol^2 * total_energy

    remaining_energy = sum(values)
    n_add = 0

    while n_add < L && remaining_energy > allowed_energy
        n_add += 1
        remaining_energy -= values[n_add]
    end

    relative_error = sqrt(max(remaining_energy, zero(T)) / total_energy)

    if n_add == 0
        shared.errors[dyn] = relative_error
        return relative_error, 0
    end

    if r + n_add > shared.max_rank
        error(
            "Distributed shared basis rank limit exceeded: " *
            "dynamic=$dyn, current_rank=$r, " *
            "required_rank=$(r + n_add), " *
            "max_rank=$(shared.max_rank), " *
            "estimated_relative_error=$relative_error"
        )
    end

    selected_values = values[1:n_add]

    if any(selected_values .<= zero(T))
        error("Cannot normalize distributed shared basis " * "at dynamic=$dyn: non-positive residual eigenvalue")
    end

    # transform = W / sqrt(λ)
    transform_cpu = vectors[:, 1:n_add] .* reshape(inv.(sqrt.(selected_values)), 1, :)
    new_rows = (r + 1):(r + n_add)

    # ----------------------------------------------------------
    # Bnew_g = R_g * transform
    # Cnew = Σ_g Bnew_gᴴ Vscaled_g
    # ----------------------------------------------------------
    local_new_coeff =
        Vector{Matrix{Complex{T}}}(undef, nShard)

    tasks = Task[]

    for i = 1:nShard
        shared_shard = shared.shards[i]
        rsvd_shard = rsvd_workspace.shards[i]

        push!(tasks, Threads.@spawn begin
            CUDA.device!(shared_shard.gpu_id)
            transform = @view shared_shard.transform[:, 1:n_add]
            copyto!(transform, transform_cpu)

            B_new = @view shared_shard.basis[:, new_rows]
            small_new = @view shared_shard.small[1:n_add, :]

            CUDA.@sync begin
                mul!(B_new, shared_shard.residual, transform)
                mul!(small_new, adjoint(B_new), rsvd_shard.v_scaled)
            end

            local_new_coeff[i] = Array(small_new)
        end)
    end

    foreach(fetch, tasks)

    C_new = zeros(Complex{T}, n_add, L)

    for local_value in local_new_coeff
        C_new .+= local_value
    end

    @views shared.coeff[new_rows, :, dyn] .= C_new

    shared.rank += n_add
    shared.errors[dyn] = relative_error

    return relative_error, n_add
end


function gather_distributed_shared_basis(
    shared::DistributedSharedSpatialBasis{T},
) where {T<:AbstractFloat}

    r = shared.rank

    if r == 0
        return zeros(Complex{T}, shared.nVox, 0)
    end

    local_parts =
        Vector{Matrix{Complex{T}}}(undef, length(shared.shards))

    tasks = Task[]

    for (i, shard) in enumerate(shared.shards)
        push!(tasks, Threads.@spawn begin
            CUDA.device!(shard.gpu_id)

            local_parts[i] = Array(@view shard.basis[:, 1:r])
        end)
    end

    foreach(fetch, tasks)

    basis = zeros(Complex{T}, shared.nVox, r)

    for (i, shard) in enumerate(shared.shards)
        basis[shard.voxels, :] .= local_parts[i]
    end

    return basis
end