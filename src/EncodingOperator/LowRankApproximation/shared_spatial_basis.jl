mutable struct SharedSpatialBasis{T, BM<:AbstractMatrix{Complex{T}}, CA<:AbstractArray{Complex{T},3}}
    basis      :: BM         # [nVox, max_rank]
    coeff      :: CA         # [max_rank, L, nDyn]
    rank       :: Int        # shared rank
    max_rank   :: Int
    tol        :: T
    errors     :: Vector{T}  # additional compression error for each dynamic
end

"""
    global_recompress_shared_basis(q, basis, tol)

Apply a final, optional compression to an already shared representation
`q * basis'`.  The incremental shared-basis construction maintains orthonormal
columns in `basis`; consequently the eigenvalues of `q' * q` are the squared
singular values of that representation.  Keeping the leading eigenvectors
`Z` gives `q_new = q * Z`, `basis_new = basis * Z`.

The returned `relative_error` is the Frobenius-norm error introduced relative
to the input representation.  It deliberately excludes local rSVD and online
shared-basis errors.
"""
function global_recompress_shared_basis(
    q::AbstractMatrix{Complex{T}},
    basis::AbstractMatrix{Complex{T}},
    tol::T,
) where {T<:AbstractFloat}
    tol >= zero(T) || throw(ArgumentError("global basis tolerance must be non-negative"))
    size(q, 2) == size(basis, 2) || throw(DimensionMismatch("q and basis must have the same rank"))

    pre_rank = size(q, 2)
    pre_rank == 0 && return q, basis, (
        pre_rank=0, rank=0, relative_error=zero(T), basis_orthogonality_error=zero(T),
    )

    # This is intentionally a host-side R×R eigendecomposition.  Form the two
    # Grams on the selected array first, so CUDA setup transfers only R² values
    # rather than the potentially multi-gigabyte q and spatial-basis matrices.
    # R is bounded by shared_rank_max, giving identical rank decisions on CPU
    # and CUDA paths.
    q_gram = similar(q, Complex{T}, pre_rank, pre_rank)
    mul!(q_gram, adjoint(q), q)
    gram = Matrix(q_gram)
    eig = eigen(Hermitian(gram))
    order = sortperm(real.(eig.values); rev=true)
    values = max.(T.(real.(eig.values[order])), zero(T))
    vectors = Matrix{Complex{T}}(eig.vectors[:, order])
    total_energy = sum(values)

    keep_rank = pre_rank
    relative_error = zero(T)
    if total_energy > eps(T)
        allowed_energy = tol^2 * total_energy
        discarded_energy = total_energy
        keep_rank = 0
        while keep_rank < pre_rank && discarded_energy > allowed_energy
            keep_rank += 1
            discarded_energy -= values[keep_rank]
        end
        # A zero rank is not useful to the NFFT operator even for a numerically
        # zero factor, so retain one direction in that degenerate case.
        keep_rank = max(keep_rank, 1)
        relative_error = sqrt(max(discarded_energy, zero(T)) / total_energy)
    end

    basis_gram = similar(basis, Complex{T}, pre_rank, pre_rank)
    mul!(basis_gram, adjoint(basis), basis)
    identity_gram = Matrix{Complex{T}}(I, pre_rank, pre_rank)
    basis_orthogonality_error = T(opnorm(Matrix(basis_gram) - identity_gram, Inf))

    if keep_rank == pre_rank
        return q, basis, (
            pre_rank=pre_rank, rank=keep_rank, relative_error=relative_error,
            basis_orthogonality_error=basis_orthogonality_error,
        )
    end

    z_host = @view vectors[:, 1:keep_rank]
    z = similar(q, Complex{T}, pre_rank, keep_rank)
    # The eigensystem is intentionally computed on the host.  Upload this
    # small rotation explicitly for CUDA arrays: generic copyto! otherwise
    # selects scalar host indexing on CuArray.
    if q isa CuArray
        copyto!(z, CuArray(z_host))
    else
        copyto!(z, z_host)
    end
    q_new = similar(q, Complex{T}, size(q, 1), keep_rank)
    basis_new = similar(basis, Complex{T}, size(basis, 1), keep_rank)
    mul!(q_new, q, z)
    mul!(basis_new, basis, z)

    return q_new, basis_new, (
        pre_rank=pre_rank, rank=keep_rank, relative_error=relative_error,
        basis_orthogonality_error=basis_orthogonality_error,
    )
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
