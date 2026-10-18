joint_release!(::Nothing) = nothing
joint_release!(::Array) = nothing
joint_release!(a::CuArray) = CUDA.unsafe_free!(a)
joint_synchronize(::Type{Array}) = nothing
joint_synchronize(::Type{CuArray}) = CUDA.synchronize()

function joint_snapshot_selection(temporal, spatial, nSam, nDyn, snapshots, seed, chunk_size)
    dimension = size(spatial, 2)
    gram = zeros(Float64, dimension, dimension)
    mean_spatial = zeros(Float64, dimension)
    for first in 1:chunk_size:size(spatial, 1)
        last = min(first + chunk_size - 1, size(spatial, 1))
        block = Float64.(spatial[first:last, :])
        gram += block' * block
        mean_spatial += vec(sum(block; dims=1))
    end
    mean_spatial ./= size(spatial, 1)
    gram ./= size(spatial, 1)
    gram -= mean_spatial * mean_spatial'
    F = eigen(Hermitian(gram))
    transform = F.vectors * Diagonal(sqrt.(max.(F.values, 0)))

    rng = Random.Xoshiro(seed)
    holdout = nDyn >= 4 ? randperm(rng, nDyn)[1:max(1, nDyn ÷ 10)] : Int[]
    training = setdiff(1:nDyn, holdout)
    samples = unique(vcat(collect(1:16:nSam), nSam))
    rows = [s + (d - 1) * nSam for d in training for s in samples]
    features = Float64.(temporal[rows, :]) * transform
    squared = vec(sum(abs2, features; dims=2))
    nearest = fill(Inf, length(rows))
    assignment = zeros(Int, length(rows))
    work = similar(squared)
    chosen = Int[]
    next = argmax(squared)
    for k in 1:min(snapshots, length(rows))
        push!(chosen, next)
        mul!(work, features, @view(features[next, :]))
        @. work = max(squared + squared[next] - 2work, 0)
        for j in eachindex(work)
            if work[j] < nearest[j]
                nearest[j] = work[j]
                assignment[j] = k
            end
        end
        # Equivalent phase geometries need only one representative. Force a
        # center to own itself even when roundoff makes its distance nonzero.
        assignment[next] = k
        nearest[next] = -Inf
        maximum(nearest) <= 0 && break
        next = argmax(nearest)
    end
    weights = [count(==(k), assignment) for k in eachindex(chosen)]
    return rows[chosen], weights, holdout
end

function joint_snapshot_basis(
    temporal::Matrix{T}, spatial::Matrix{T}, indices, weights, rank_max, chunk_size, arrayType,
) where T<:AbstractFloat
    nVox = size(spatial, 1)
    K = length(indices)
    chunk_size = min(chunk_size, nVox)
    centers = root_weights = spatial_device = phase = encoding = gram = nothing
    rotation = basis_gram = correction = scratch = basis = nothing
    succeeded = false
    try
        centers = arrayType(temporal[indices, :])
        root_weights = arrayType(reshape(sqrt.(T.(weights)), 1, :))
        spatial_device = arrayType(spatial)
        phase = arrayType(zeros(T, chunk_size, K))
        encoding = arrayType(zeros(Complex{T}, chunk_size, K))
        gram = arrayType(zeros(Complex{T}, K, K))
        for first in 1:chunk_size:nVox
            last = min(first + chunk_size - 1, nVox)
            n = last - first + 1
            p = @view phase[1:n, :]
            e = @view encoding[1:n, :]
            mul!(p, @view(spatial_device[first:last, :]), transpose(centers))
            @. e = cis(-T(2π) * p) * root_weights
            mul!(gram, adjoint(e), e, one(Complex{T}), one(Complex{T}))
        end
        F = eigen(Hermitian(ComplexF64.(Array(gram))))
        order = sortperm(F.values; rev=true)
        values = F.values[order]
        # The independent original-phase residual decides acceptance. Dropping
        # unresolved Gram directions prevents division by roundoff-sized energy.
        resolved = count(>(eps(T) * values[1]), values)
        R = min(rank_max, resolved)
        R > 0 || error("Joint spatial basis has no numerically resolved direction")
        rotation = arrayType(Complex{T}.(F.vectors[:, order[1:R]] ./ reshape(sqrt.(values[1:R]), 1, :)))
        basis = arrayType(zeros(Complex{T}, nVox, R))
        for first in 1:chunk_size:nVox
            last = min(first + chunk_size - 1, nVox)
            n = last - first + 1
            p = @view phase[1:n, :]
            e = @view encoding[1:n, :]
            mul!(p, @view(spatial_device[first:last, :]), transpose(centers))
            @. e = cis(-T(2π) * p) * root_weights
            mul!(@view(basis[first:last, :]), e, rotation)
        end
        basis_gram = similar(basis, Complex{T}, R, R)
        mul!(basis_gram, adjoint(basis), basis)
        C = cholesky(Hermitian(ComplexF64.(Array(basis_gram)))).U
        correction = arrayType(Complex{T}.(C \ Matrix{ComplexF64}(I, R, R)))
        scratch = similar(basis, Complex{T}, chunk_size, R)
        for first in 1:chunk_size:nVox
            last = min(first + chunk_size - 1, nVox)
            block = @view basis[first:last, :]
            target = @view scratch[1:last-first+1, :]
            mul!(target, block, correction)
            copyto!(block, target)
        end
        mul!(basis_gram, adjoint(basis), basis)
        orthogonality = norm(ComplexF64.(Array(basis_gram)) - I)
        orthogonality <= 100eps(T) * R || error("Joint spatial basis lost orthogonality: $orthogonality")
        succeeded = true
        return basis, orthogonality
    finally
        joint_synchronize(arrayType)
        succeeded || joint_release!(basis)
        for a in (centers, root_weights, spatial_device, phase, encoding, gram,
                  rotation, basis_gram, correction, scratch)
            joint_release!(a)
        end
    end
end

function joint_sample_rows(nSam, nDyn, seed)
    available = setdiff(1:nSam, unique(vcat(collect(1:16:nSam), nSam)))
    # For tiny inputs there may be no unused time sample. Spatial fitting and
    # validation remain separate; the report records the reused time grid.
    disjoint = length(available) >= 2
    isempty(available) && (available = collect(1:nSam))
    n_validation = min(193, max(1, length(available) ÷ 2))
    n_audit = disjoint ? min(257, length(available) - n_validation) : 1
    validation = Matrix{Int}(undef, n_validation, nDyn)
    audit = Matrix{Int}(undef, n_audit, nDyn)
    rng = Random.Xoshiro(seed)
    for d in 1:nDyn
        order = randperm(rng, length(available))
        validation[:, d] = sort(available[order[1:n_validation]]) .+ (d - 1) * nSam
        audit_order = disjoint ? order[n_validation+1:n_validation+n_audit] : order[1:1]
        audit[:, d] = sort(available[audit_order]) .+ (d - 1) * nSam
    end
    return validation, audit, disjoint
end

function joint_fit_points(basis, pool, sample_count, seed, arrayType)
    index = sampled_device = nothing
    try
        index = arrayType(pool)
        sampled_device = basis[index, :]
        sampled = Array(sampled_device)
        anchors = min(size(basis, 2), sample_count)
        pivots = qr(Matrix(sampled'), ColumnNorm()).p[1:anchors]
        rest = setdiff(randperm(Random.Xoshiro(seed), length(pool)), pivots)
        chosen = vcat(pivots, rest[1:sample_count-anchors])
        return pool[chosen], sampled[chosen, :]
    finally
        joint_synchronize(arrayType)
        joint_release!(sampled_device)
        joint_release!(index)
    end
end

function joint_fit_coefficients(
    temporal::Matrix{T}, spatial::Matrix{T}, points, fit_map, chunk_size, arrayType,
) where T<:AbstractFloat
    nPoint = size(temporal, 1)
    J, R = size(fit_map, 2), size(fit_map, 1)
    chunk_size = min(chunk_size, nPoint)
    a = b = transform = phase = encoding = q = nothing
    succeeded = false
    try
        a = arrayType(temporal)
        b = arrayType(spatial[points, :])
        transform = arrayType(Complex{T}.(fit_map'))
        phase = arrayType(zeros(T, chunk_size, J))
        encoding = arrayType(zeros(Complex{T}, chunk_size, J))
        q = arrayType(zeros(Complex{T}, nPoint, R))
        for first in 1:chunk_size:nPoint
            last = min(first + chunk_size - 1, nPoint)
            n = last - first + 1
            p = @view phase[1:n, :]
            e = @view encoding[1:n, :]
            mul!(p, @view(a[first:last, :]), transpose(b))
            @. e = cis(T(2π) * p)
            mul!(@view(q[first:last, :]), e, transform)
        end
        succeeded = true
        return q
    finally
        joint_synchronize(arrayType)
        succeeded || joint_release!(q)
        for buffer in (a, b, transform, phase, encoding)
            joint_release!(buffer)
        end
    end
end

function joint_sample_errors(
    basis::AbstractMatrix{Complex{T}}, temporal, spatial, rows, voxels, n_random,
    points, fit_map, chunk_size, arrayType; q=nothing,
) where T<:AbstractFloat
    n_validation, nDyn = size(rows)
    nPoint = length(rows)
    R = q === nothing ? size(fit_map, 1) : size(q, 2)
    J = length(points)
    chunk_size = min(chunk_size, nPoint)
    a = a_reference = b = b_reference = sampled = transform = nothing
    phase = encoding = phase_reference = reference = approximate = q_block = nothing
    index = gathered = squared = reduced = nothing
    errors = zeros(Float64, nDyn)
    roi_errors = zeros(Float64, nDyn)
    try
        a_host = temporal[vec(rows), :]
        a_reference = arrayType(Float64.(a_host))
        b_reference = arrayType(Float64.(spatial[voxels, :]))
        index = arrayType(voxels)
        sampled = basis[index, 1:R]
        joint_synchronize(arrayType)
        joint_release!(index)
        index = nothing
        if q === nothing
            a = arrayType(a_host)
            b = arrayType(spatial[points, :])
            transform = arrayType(Complex{T}.(fit_map'))
            phase = arrayType(zeros(T, chunk_size, J))
            encoding = arrayType(zeros(Complex{T}, chunk_size, J))
        end
        phase_reference = arrayType(zeros(Float64, chunk_size, length(voxels)))
        reference = arrayType(zeros(ComplexF64, chunk_size, length(voxels)))
        approximate = arrayType(zeros(Complex{T}, chunk_size, length(voxels)))
        q_block = arrayType(zeros(Complex{T}, chunk_size, R))
        for first in 1:chunk_size:nPoint
            last = min(first + chunk_size - 1, nPoint)
            n = last - first + 1
            qb = @view q_block[1:n, :]
            if q === nothing
                p = @view phase[1:n, :]
                e = @view encoding[1:n, :]
                mul!(p, @view(a[first:last, :]), transpose(b))
                @. e = cis(T(2π) * p)
                mul!(qb, e, transform)
            else
                index = arrayType(vec(rows)[first:last])
                gathered = q[index, :]
                copyto!(qb, gathered)
            end
            p_reference = @view phase_reference[1:n, :]
            truth = @view reference[1:n, :]
            prediction = @view approximate[1:n, :]
            mul!(p_reference, @view(a_reference[first:last, :]), transpose(b_reference))
            @. truth = cis(2π * p_reference)
            mul!(prediction, qb, adjoint(sampled))
            squared = abs2.(prediction .- truth)
            for (columns, accumulated) in ((1:n_random, errors), (n_random+1:length(voxels), roi_errors))
                reduced = sum(@view(squared[:, columns]); dims=2)
                host = vec(Array(reduced))
                for j in eachindex(host)
                    accumulated[cld(first + j - 1, n_validation)] += host[j]
                end
                joint_release!(reduced)
                reduced = nothing
            end
            # The host reductions have completed every producer on this stream.
            for buffer in (index, gathered, squared)
                joint_release!(buffer)
            end
            index = gathered = squared = nothing
        end
        return sqrt.(errors ./ (n_validation * n_random)),
            sqrt.(roi_errors ./ (n_validation * (length(voxels) - n_random)))
    finally
        joint_synchronize(arrayType)
        for buffer in (a, a_reference, b, b_reference, sampled, transform, phase, encoding,
                       phase_reference, reference, approximate, q_block, index, gathered, squared, reduced)
            joint_release!(buffer)
        end
    end
end

"""
    joint_spatial_basis(times, fieldmap, bf, kspha_err; kwargs...)

Build `q, basis, report` for the pure-phase encoding
`H[s,v,d] = cis(2π * (times[s,d]*fieldmap[v] + sum(kspha_err[:,s,d].*bf[v,:])))`,
with `q` of size `(nSam*nDyn, R)` and `basis` of size `(nVox, R)`.
The flattened approximation is `H ≈ q * basis'`. Times are in seconds,
fieldmap in Hz, and the field-coefficient products in cycles.
No NFFT, coil, zeroth-order or `1/sqrt(nVox)` correction is included.

Inputs are host arrays; `arrayType=Array` or `CuArray` selects computation and
returned storage. GPU setup uses the current device. Both Float32 and Float64
are supported. The caller owns the returned factors; temporary device arrays
are released on success and failure.

`snapshots` sets the representative phase-row budget, bounded by available
phase geometries. The full budget is used before rank selection.
`sample_count` sets the
initial coefficient-fitting sample count (doubled on failure), and `rank_max`
bounds the final rank. `chunk_size` bounds temporary phase matrices. The search
tests rank 1 and then blocks of eight; it does not prove a minimum rank.

Acceptance requires every profile's independent sampled relative Frobenius
error to meet `tol`, followed by a fresh audit of the completed factors.
Rank selection reserves 10% of the tolerance for sampling variation. This
margin is heuristic; the audit remains mandatory and can still fail.
`roi_tol` optionally also constrains the sampled largest-|B0| voxels. Without
it, large ROI errors are reported and warned about, not certified. These are
sampled matrix errors, not full-matrix or image guarantees. The report records
when tiny inputs require reuse of a validation time. At least four voxels are
required to separate fitting and validation. `report_ref` may be a `Ref` to
capture diagnostics even on a rank-limit or validation error.

All rank/validation failures throw; there is no CPU fallback or returned
partial approximation. The snapshot Gram is computed in the requested
precision; Float64 validation does not recover directions lost by that Gram.
"""
function joint_spatial_basis(
    times       :: AbstractMatrix{T},
    fieldmap    :: AbstractVector{T},
    bf          :: AbstractMatrix{T},
    kspha_err   :: AbstractArray{T,3};
    arrayType   :: Type{<:AbstractArray} = Array,
    rank_max    :: Int = 128,
    tol         :: T = T(1e-2),
    snapshots   :: Int = 256,
    sample_count:: Int = 512,
    chunk_size  :: Int = 32768,
    seed        :: Int = 1234,
    roi_tol     :: Union{Nothing,T} = nothing,
    report_ref  :: Union{Nothing,Ref} = nothing,
    verbose     :: Bool = false,
) where T<:AbstractFloat
    T in (Float32, Float64) || throw(ArgumentError("Joint spatial basis requires Float32 or Float64"))
    arrayType in (Array, CuArray) || throw(ArgumentError("Joint spatial basis supports Array or CuArray"))
    nSam, nDyn = size(times)
    nVox = length(fieldmap)
    size(bf, 1) == nVox || throw(DimensionMismatch("bf rows must match fieldmap length"))
    size(kspha_err) == (size(bf, 2), nSam, nDyn) || throw(DimensionMismatch("kspha_err must have size (nTerm, nSam, nDyn)"))
    nSam > 0 && nDyn > 0 && nVox >= 4 || throw(ArgumentError("Joint spatial basis requires samples, dynamics, and at least four voxels"))
    rank_max > 0 && snapshots > 0 && sample_count > 0 && chunk_size > 0 || throw(ArgumentError("Joint rank, snapshots, sample count and chunk size must be positive"))
    isfinite(tol) && tol > zero(T) || throw(ArgumentError("Joint basis tolerance must be finite and positive"))
    isnothing(roi_tol) || (isfinite(roi_tol) && roi_tol > zero(T)) || throw(ArgumentError("Joint ROI tolerance must be finite and positive"))
    all(isfinite, times) && all(isfinite, fieldmap) && all(isfinite, bf) && all(isfinite, kspha_err) || throw(ArgumentError("Joint phase inputs must be finite"))

    spatial = hcat(Vector{T}(fieldmap), Matrix{T}(bf))
    temporal = hcat(vec(Matrix{T}(times)), permutedims(reshape(Array{T,3}(kspha_err), size(bf, 2), nSam * nDyn)))
    order = randperm(Random.Xoshiro(seed), nVox)
    pool_size = min(8192, nVox ÷ 2)
    pool = sort(order[1:pool_size])
    available = order[pool_size+1:end]
    n_random = min(1024, max(1, length(available) ÷ 2))
    validation_voxels = available[1:n_random]
    audit_order = randperm(Random.Xoshiro(seed + 1), length(available))
    audit_voxels = available[audit_order[1:n_random]]
    roi_candidates = setdiff(sortperm(abs.(fieldmap); rev=true), pool)
    n_roi = min(128, length(available) - n_random)
    validation_roi = setdiff(roi_candidates, validation_voxels)[1:n_roi]
    audit_roi = setdiff(roi_candidates, audit_voxels)[1:n_roi]
    validation_rows, audit_rows, disjoint_times = joint_sample_rows(nSam, nDyn, seed + 2)
    effective_rank_max = min(rank_max, nSam * nDyn, pool_size, 2 * min(sample_count, pool_size))
    basis = q = selected_basis = nothing
    selected_points = Int[]
    selected_indices = Int[]
    selected_holdout = Int[]
    selected_map = zeros(ComplexF64, 0, 0)
    errors = roi_errors = fill(Inf, nDyn)
    audit_errors = audit_roi_errors = Float64[]
    selection_tol = T(0.9) * tol
    selected_R = 0
    orthogonality = Inf
    passed = false
    succeeded = false
    try
        indices, weights, holdout = joint_snapshot_selection(temporal, spatial, nSam, nDyn, snapshots, seed + 3, chunk_size)
        basis, orthogonality = joint_snapshot_basis(temporal, spatial, indices, weights, effective_rank_max, chunk_size, arrayType)
        selected_indices = indices
        selected_holdout = holdout
        for J in unique([min(sample_count, pool_size), min(2 * min(sample_count, pool_size), pool_size)])
            points, sampled = joint_fit_points(basis, pool, J, seed + 4, arrayType)
            rank_limit = min(size(basis, 2), J)
            for R in unique(vcat(1, collect(8:8:rank_limit), rank_limit))
                fit = qr(ComplexF64.(sampled[:, 1:R]))
                fit_map = fit \ Matrix{ComplexF64}(I, J, J)
                errors, roi_errors = joint_sample_errors(basis, temporal, spatial, validation_rows,
                    vcat(validation_voxels, validation_roi), n_random, points, fit_map, chunk_size, arrayType)
                passed = maximum(errors) <= selection_tol && (isnothing(roi_tol) || maximum(roi_errors) <= T(0.9) * roi_tol)
                selected_R, selected_points, selected_map = R, points, fit_map
                if verbose @info("Joint shared-basis rank check", snapshots=length(indices), rank=R,
                    samples=J, max_error=maximum(errors), max_roi_error=maximum(roi_errors), passed) end
                passed && break
            end
            passed && break
        end
        if passed
            selected_basis = copy(@view basis[:, 1:selected_R])
            joint_synchronize(arrayType)
            joint_release!(basis)
            basis = nothing
            q = joint_fit_coefficients(temporal, spatial, selected_points, selected_map, chunk_size, arrayType)
            audit_errors, audit_roi_errors = joint_sample_errors(selected_basis, temporal, spatial, audit_rows,
                vcat(audit_voxels, audit_roi), n_random, selected_points, selected_map, chunk_size, arrayType; q)
        end
        audit_passed = passed && maximum(audit_errors) <= tol && (isnothing(roi_tol) || maximum(audit_roi_errors) <= roi_tol)
        report = (; rank=selected_R, rank_max=effective_rank_max, snapshots=length(selected_indices),
            sample_count=length(selected_points), tolerance=tol, selection_tolerance=selection_tol, roi_tolerance=roi_tol,
            validation_errors=errors, validation_roi_errors=roi_errors, audit_errors, roi_errors=audit_roi_errors,
            orthogonality, passed=audit_passed, stage=passed ? :audit : :rank_limit,
            snapshot_indices=selected_indices, heldout_profiles=selected_holdout,
            fit_indices=selected_points, validation_rows, audit_rows,
            validation_voxels, audit_voxels, validation_roi, audit_roi, disjoint_times, seed)
        isnothing(report_ref) || (report_ref[] = report)
        passed || error("Joint spatial basis rank limit or sample limit reached: rank=$(selected_R), max_error=$(maximum(errors)), max_roi_error=$(maximum(roi_errors))")
        audit_passed || error("Joint spatial basis failed independent audit: rank=$(selected_R), max_error=$(maximum(audit_errors)), max_roi_error=$(maximum(audit_roi_errors))")
        if isnothing(roi_tol) && maximum(audit_roi_errors) > tol
            @warn "Joint spatial basis meets sampled random-voxel tolerance but not the sampled high-B0 region" max_roi_error=maximum(audit_roi_errors) tolerance=tol
        end
        if verbose @info("Joint shared spatial basis complete", rank=selected_R, snapshots=report.snapshots,
            samples=report.sample_count, max_error=maximum(audit_errors), max_roi_error=maximum(audit_roi_errors)) end
        succeeded = true
        return q, selected_basis, report
    finally
        joint_synchronize(arrayType)
        joint_release!(basis)
        if !succeeded
            joint_release!(q)
            joint_release!(selected_basis)
        end
    end
end
