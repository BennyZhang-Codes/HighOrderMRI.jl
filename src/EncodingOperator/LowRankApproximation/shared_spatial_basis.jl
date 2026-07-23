mutable struct SharedSpatialBasis{T, BM<:AbstractMatrix{Complex{T}}, CA<:AbstractArray{Complex{T},3}}
    basis      :: BM         # [nVox, max_rank]
    coeff      :: CA         # [max_rank, L, nDyn]
    rank       :: Int        # shared rank
    max_rank   :: Int
    tol        :: T
    errors     :: Vector{T}  # additional compression error for each dynamic
end

function SharedSpatialBasis(
    prototype,
    ::Type{T},
    nVox    :: Int,
    L       :: Int,
    nDyn    :: Int,
    max_rank:: Int,
    tol     :: T,
) where T<:AbstractFloat

    @assert max_rank > 0
    @assert max_rank <= nVox
    @assert tol >= zero(T)

    basis = similar(prototype, Complex{T}, nVox, max_rank)
    coeff = similar(prototype, Complex{T}, max_rank, L, nDyn)

    fill!(basis, zero(Complex{T}))
    fill!(coeff, zero(Complex{T}))

    return SharedSpatialBasis(basis, coeff, 0, max_rank, tol, zeros(T, nDyn))
end

mutable struct SharedBasisUpdateWorkspace{CM<:AbstractMatrix}
    v_scaled  :: CM  # [nVox, L]
    residual  :: CM  # [nVox, L]
    correction:: CM  # [max_rank, L]
    gram      :: CM  # [L, L]
    transform :: CM  # [L, L]
end

function SharedBasisUpdateWorkspace(
    prototype,
    ::Type{T},
    nVox   :: Int,
    L      :: Int,
    max_rank:: Int,
) where T<:AbstractFloat

    return SharedBasisUpdateWorkspace(
        similar(prototype, Complex{T}, nVox, L),
        similar(prototype, Complex{T}, nVox, L),
        similar(prototype, Complex{T}, max_rank, L),
        similar(prototype, Complex{T}, L, L),
        similar(prototype, Complex{T}, L, L),
    )
end


function update_shared_basis!(
    shared      :: SharedSpatialBasis{T},
    workspace   :: SharedBasisUpdateWorkspace,
    v_scaled    :: AbstractMatrix{Complex{T}},
    dyn         :: Int,
    total_energy:: T,
) where T<:AbstractFloat

    @assert size(v_scaled, 1) == size(shared.basis, 1)
    @assert size(v_scaled, 2) == size(shared.coeff, 2)
    @assert 1 <= dyn <= size(shared.coeff, 3)

    r = shared.rank
    L = size(v_scaled, 2)

    residual = workspace.residual
    copyto!(residual, v_scaled)

    # 当前dynamic的系数先清零
    fill!(@view(shared.coeff[:, :, dyn]), zero(Complex{T}))

    # ---------------------------------------------------------
    # 第一次投影：C = BᴴV, R = V - BC
    # ---------------------------------------------------------
    if r > 0
        B = @view shared.basis[:, 1:r]
        C = @view shared.coeff[1:r, :, dyn]

        mul!(C, adjoint(B), v_scaled)
        mul!(residual, B, C, -one(Complex{T}), one(Complex{T}))

        # -----------------------------------------------------
        # 第二次正交化，减少累计数值误差
        # -----------------------------------------------------
        correction = @view workspace.correction[1:r, :]

        mul!(correction, adjoint(B), residual)
        C .+= correction

        mul!(residual, B, correction, -one(Complex{T}), one(Complex{T}))
    end

    if total_energy <= eps(T)
        shared.errors[dyn] = zero(T)
        return zero(T), 0
    end

    # ---------------------------------------------------------
    # G = RᴴR，only L×L
    # ---------------------------------------------------------
    gram = workspace.gram
    mul!(gram, adjoint(residual), residual)

    # L is small
    gram_cpu = Array(gram)
    eig = eigen(Hermitian(gram_cpu))

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
        remaining_energy = n_add == L ? zero(T) : sum(@view values[(n_add + 1):L])
    end

    relative_error = sqrt(max(remaining_energy, zero(T)) / total_energy)

    if n_add == 0
        shared.errors[dyn] = relative_error
        return relative_error, 0
    end

    if r + n_add > shared.max_rank
        required_rank = r + n_add

        error(
            "Shared spatial basis rank limit exceeded: " *
            "dynamic=$dyn, current_rank=$r, " *
            "required_rank=$required_rank, " *
            "max_rank=$(shared.max_rank), " *
            "estimated_relative_error=$relative_error"
        )
    end

    # ---------------------------------------------------------
    # Qnew = residual * W * diag(1 / sqrt(λ))
    # ---------------------------------------------------------
    selected_values = values[1:n_add]

    if any(selected_values .<= zero(T))
        error(
            "Cannot normalize shared basis residual at dynamic=$dyn: " *
            "non-positive residual eigenvalue detected"
        )
    end

    transform_cpu = vectors[:, 1:n_add] .* reshape(inv.(sqrt.(selected_values)), 1, :)

    transform = @view workspace.transform[:, 1:n_add]
    copyto!(transform, transform_cpu)

    new_rows = (r + 1):(r + n_add)
    B_new = @view shared.basis[:, new_rows]

    mul!(B_new, residual, transform)

    C_new = @view shared.coeff[new_rows, :, dyn]
    mul!(C_new, adjoint(B_new), v_scaled)

    shared.rank += n_add
    shared.errors[dyn] = relative_error

    return relative_error, n_add
end


function reconstruct_spatial_factors!(
    destination::AbstractMatrix{Complex{T}},
    shared    :: SharedSpatialBasis{T},
    dyn       :: Int,
) where T<:AbstractFloat

    r = shared.rank

    if r == 0
        fill!(destination, zero(Complex{T}))
        return destination
    end

    B = @view shared.basis[:, 1:r]
    C = @view shared.coeff[1:r, :, dyn]

    mul!(destination, B, C)

    return destination
end
