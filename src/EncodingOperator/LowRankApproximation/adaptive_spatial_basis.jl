function adaptive_exact_phase(temporal, spatial, row, voxel)
    cycles = 0.0
    @inbounds for p in axes(temporal, 2)
        cycles += Float64(temporal[row, p]) * Float64(spatial[voxel, p])
    end
    return cis(2π * cycles)
end

function adaptive_uniform_pairs(nTime, nVox, count, seed)
    rng = Random.Xoshiro(seed)
    return (rows=rand(rng, 1:nTime, count), voxels=rand(rng, 1:nVox, count))
end


function adaptive_prefix_errors(temporal, spatial, q, S, pairs; all_prefixes=true)
    n = length(pairs.rows)
    n == length(pairs.voxels) && n >= 2 || throw(DimensionMismatch("paired index count"))
    size(q) == (size(temporal, 1), size(S, 2)) || throw(DimensionMismatch("temporal factor"))
    size(S, 1) == size(spatial, 1) || throw(DimensionMismatch("spatial factor"))
    size(temporal, 2) == size(spatial, 2) || throw(DimensionMismatch("phase features"))
    checkbounds(q, pairs.rows, 1)
    checkbounds(S, pairs.voxels, 1)
    r = size(q, 2)
    nChecks = all_prefixes ? r : 1
    nBlocks = min(n, Threads.nthreads())
    sums = zeros(Float64, nChecks, nBlocks)
    squared_sums = similar(sums)
    fill!(squared_sums, 0)
    Threads.@threads for b in 1:nBlocks
        first, last = fld((b-1) * n, nBlocks) + 1, fld(b * n, nBlocks)
        @inbounds for j in first:last
            t, v = pairs.rows[j], pairs.voxels[j]
            h = adaptive_exact_phase(temporal, spatial, t, v)
            prediction = zero(ComplexF64)
            for k in 1:r
                prediction += ComplexF64(q[t, k]) * conj(ComplexF64(S[v, k]))
                if all_prefixes || k == r
                    i = all_prefixes ? k : 1
                    z = abs2(h - prediction)
                    sums[i, b] += z
                    squared_sums[i, b] += z*z
                end
            end
        end
    end
    total = vec(sum(sums; dims=2))
    e2 = total ./ n
    variance = max.(0, (vec(sum(squared_sums; dims=2)) .- total .* e2) ./ (n-1))
    all(isfinite, e2) && all(isfinite, variance) || error("Nonfinite paired residual")
    return e2, sqrt.(variance ./ n)
end

function adaptive_coefficient_spectrum(q; block_size=32768)
    # A QR of stacked triangular factors preserves the full coefficient spectrum.
    triangular = Matrix{ComplexF64}(undef, 0, size(q, 2))
    for first in 1:block_size:size(q, 1)
        rows = first:min(first + block_size - 1, size(q, 1))
        block = ComplexF64.(@view q[rows, :])
        local_r = Matrix(qr!(block).R)
        triangular = Matrix(qr!(vcat(triangular, local_r)).R)
    end
    return svd(triangular; full=false)
end

function adaptive_fit_pool(nVox, count, seed)
    count == nVox && return collect(1:nVox)
    rng = Random.Xoshiro(seed)
    pool = Set{Int}()
    while length(pool) < count
        push!(pool, rand(rng, 1:nVox))
    end
    return sort!(collect(pool))
end


function adaptive_snapshot_space(temporal, spatial, nSam, nDyn, K, cap, options, arrayType)
    stages = Dict{String,Float64}()
    stages["snapshot_selection_s"] = @elapsed begin
        indices, weights, holdout = joint_snapshot_selection(temporal, spatial,
            nSam, nDyn, K, options.build_seed + 3, options.chunk_size)
    end
    device_basis = nothing
    try
        stages["snapshot_basis_s"] = @elapsed begin
            device_basis, orthogonality, resolved, tail, fallback = joint_snapshot_basis(
                temporal, spatial, indices, weights, cap, options.chunk_size, arrayType)
        end
        stages["basis_transfer_s"] = @elapsed S0 = Array(device_basis)
        return (; S0, indices, weights, holdout, orthogonality, resolved, tail, fallback, stages)
    finally
        joint_synchronize(arrayType)
        joint_release!(device_basis)
    end
end

function adaptive_fit_space(S0, temporal, spatial, J, options, arrayType)
    stages = Dict{String,Float64}()
    stages["fit_map_s"] = @elapsed begin
        pool = adaptive_fit_pool(size(spatial, 1), min(size(spatial, 1), options.fit_pool), options.build_seed)
        points, sampled = joint_fit_points(S0, pool, J, options.build_seed + 4, Array)
        fit = qr(ComplexF64.(sampled))
        values = svdvals(Matrix(fit.R))
        condition = first(values) / last(values)
    end
    if !isfinite(condition) || condition > options.condition_limit
        return (; status=:ill_conditioned, condition, points, pool, stages)
    end
    fit_map = fit \ Matrix{ComplexF64}(I, J, J)
    device_q = q0 = nothing
    try
        stages["all_time_coefficients_s"] = @elapsed device_q = joint_fit_coefficients(
            temporal, spatial, points, fit_map, options.chunk_size, arrayType)
        stages["coefficient_transfer_s"] = @elapsed q0 = Array(device_q)
    finally
        joint_synchronize(arrayType)
        joint_release!(device_q)
    end
    stages["coefficient_svd_s"] = @elapsed spectrum = adaptive_coefficient_spectrum(q0; block_size=options.chunk_size)
    stages["factor_rotation_s"] = @elapsed begin
        rotation = Complex{eltype(temporal)}.(spectrum.V)
        q = q0 * rotation
        S = S0 * rotation
    end
    return (; status=:computed, q, S, condition, points, pool, singular_values=spectrum.S, stages)
end

"""
    adaptive_spatial_basis(times, fieldmap, bf, kspha_err; epsilon_H,
        arrayType=Array, seed=1234, chunk_size=4096, rank_max=128,
        initial_parent=32, snapshots=128, sample_count=512,
        max_snapshots=512, max_samples=2048, fit_pool_size=8192,
        max_attempts=8, max_seconds=1800.0, parent_increment=16,
        condition_limit=1e4, memory_fraction=0.8,
        selection_pairs=100_000, acceptance_pairs=400_000,
        selection_seed=seed+1_000_000, acceptance_seed=seed+2_000_000,
        diagnostic_seed=seed+3_000_000, diagnostic_rows=24)

Build experimental shared factors for the pure residual matrix
`H[(s,d),v] = cis(2π*(times[s,d]*fieldmap[v] + Σ kspha_err[p,s,d]*bf[v,p]))`.
Times are seconds, fieldmap is Hz, and each higher-order product is in cycles.
No first-order NFFT, zeroth-order or center phase, CSM, weights or voxel-count
normalization is included. `q` and `S` are host matrices of `Complex{T}` with
`H ≈ q*S'`, even when construction uses `CuArray`.

Start from a stable snapshot master, fit all time rows by thin QR, and rotate
using a blocked Float64 QR/small SVD of the full-time coefficients. Select R
by residuals against the original H, not by coefficient tail energy alone.
Both selection and acceptance use independent uniform paired entries from
the entire H, including construction entries. The target is global relative
Frobenius error; it is not a worst-readout, ROI or image-error requirement.

`sqrt(mean(loss)+4SE)` is an empirical heuristic, not a probability bound.
The returned `certification_status` and `external_audit_status` are always
`:not_requested`. An independent saved-factor audit remains the caller's
responsibility. On budget failure return `q=S=nothing`, rank zero and the
failure reason; never return partial factors as success.

Only after failure, diagnose sampled rows over all voxels and expand J, the
parent cap or K within the declared budgets. `rank_max` caps the parent, not
the requested final rank. The memory estimate is approximate; `max_seconds`
is checked between attempts, not inside a CUDA call. This setup function
does not create NFFT plans and does not change any production default.
"""
function adaptive_spatial_basis(
    times::AbstractMatrix{T}, fieldmap::AbstractVector{T},
    bf::AbstractMatrix{T}, kspha_err::AbstractArray{T,3};
    epsilon_H::Real,
    arrayType::Type{<:AbstractArray}=Array,
    seed::Int=1234, chunk_size::Int=4096, rank_max::Int=128,
    initial_parent::Int=32, snapshots::Int=128, sample_count::Int=512,
    max_snapshots::Int=512, max_samples::Int=2048, fit_pool_size::Int=8192,
    max_attempts::Int=8, max_seconds::Real=1800.0, parent_increment::Int=16,
    condition_limit::Real=1e4, memory_fraction::Real=0.8,
    selection_pairs::Int=100_000, acceptance_pairs::Int=400_000,
    selection_seed::Int=seed+1_000_000, acceptance_seed::Int=seed+2_000_000,
    diagnostic_seed::Int=seed+3_000_000, diagnostic_rows::Int=24,
) where T<:AbstractFloat
    T in (Float32, Float64) || throw(ArgumentError("Adaptive basis requires Float32 or Float64"))
    arrayType in (Array, CuArray) || throw(ArgumentError("Adaptive basis supports Array or CuArray"))
    isfinite(epsilon_H) && 0 < epsilon_H < 1 || throw(ArgumentError("epsilon_H must lie in (0,1)"))
    nSam, nDyn = size(times)
    nVox = length(fieldmap)
    nSam > 0 && nDyn > 0 && nVox > 0 || throw(ArgumentError("Adaptive basis requires nonempty inputs"))
    size(bf, 1) == nVox || throw(DimensionMismatch("bf rows must match fieldmap length"))
    size(kspha_err) == (size(bf, 2), nSam, nDyn) || throw(DimensionMismatch("residual coefficient layout"))
    all(>(0), (chunk_size, rank_max, initial_parent, snapshots, sample_count,
        max_snapshots, max_samples, fit_pool_size, max_attempts, parent_increment,
        diagnostic_rows)) || throw(ArgumentError("Adaptive construction budgets must be positive"))
    selection_pairs >= 2 && acceptance_pairs >= 2 || throw(ArgumentError("Paired checks require at least two entries"))
    snapshots <= max_snapshots || throw(ArgumentError("snapshots exceeds max_snapshots"))
    sample_count <= max_samples <= fit_pool_size || throw(ArgumentError("Require sample_count <= max_samples <= fit_pool_size"))
    isfinite(max_seconds) && max_seconds > 0 || throw(ArgumentError("max_seconds must be finite and positive"))
    isfinite(condition_limit) && condition_limit >= 1 || throw(ArgumentError("condition_limit must be finite and at least one"))
    isfinite(memory_fraction) && 0 < memory_fraction <= 1 || throw(ArgumentError("memory_fraction must lie in (0,1]"))
    all(isfinite, times) && all(isfinite, fieldmap) && all(isfinite, bf) && all(isfinite, kspha_err) ||
        throw(ArgumentError("Adaptive phase inputs must be finite"))
    spatial = hcat(Vector{T}(fieldmap), Matrix{T}(bf))
    temporal = hcat(vec(Matrix{T}(times)), permutedims(reshape(Array{T,3}(kspha_err), size(bf, 2), length(times))))
    options = (; build_seed=seed, chunk_size, hard_parent_limit=rank_max,
        snapshots, fit_points=sample_count, max_snapshots, max_fit_points=max_samples,
        fit_pool=fit_pool_size, max_attempts, max_seconds, parent_increment,
        condition_limit, memory_fraction, selection_pairs, acceptance_pairs,
        selection_seed, acceptance_seed, diagnostic_seed, diagnostic_rows)
    initial_parent = min(initial_parent, rank_max, nVox, length(times), sample_count)
    return adaptive_phase_factors(temporal, spatial, nSam, nDyn, epsilon_H, options;
        initial_parent, arrayType)
end

function adaptive_internal_ranks(temporal, spatial, q, S, epsilon, options, attempt)
    nTime, nVox = size(temporal, 1), size(spatial, 1)
    selection = adaptive_uniform_pairs(nTime, nVox, options.selection_pairs, options.selection_seed + attempt)
    acceptance = adaptive_uniform_pairs(nTime, nVox, options.acceptance_pairs, options.acceptance_seed + attempt)
    e2, se = adaptive_prefix_errors(temporal, spatial, q, S, selection)
    a2, ase = adaptive_prefix_errors(temporal, spatial, q, S, acceptance)
    rank = findfirst(r -> sqrt(e2[r]+4se[r]) < epsilon && sqrt(a2[r]+4ase[r]) < epsilon, eachindex(e2))
    return (; rank, e2, se, a2, ase, selection, acceptance)
end

function adaptive_projection_diagnostic(temporal, spatial, q, S, options)
    rows = adaptive_fit_pool(size(temporal, 1), min(size(temporal,1), options.diagnostic_rows),
        options.diagnostic_seed)
    a = Float64.(temporal[rows, :])
    r = size(S, 2)
    gram = zeros(ComplexF64, r, r)
    projected = zeros(ComplexF64, length(rows), r)
    chunk = options.chunk_size
    for first in 1:chunk:size(spatial,1)
        voxels = first:min(first+chunk-1, size(spatial,1))
        block = ComplexF64.(S[voxels, :])
        H = cis.(2π .* (a * transpose(Float64.(spatial[voxels, :]))))
        mul!(gram, adjoint(block), block, one(ComplexF64), one(ComplexF64))
        mul!(projected, H, block, one(ComplexF64), one(ComplexF64))
    end
    # S is already close to orthonormal. This small positive-definite solve
    # also corrects its saved Float32 Gram; no total-minus-leading subtraction.
    solve = cholesky(Hermitian(gram))
    oracle = Matrix(adjoint(solve \ adjoint(projected)))
    fitted = ComplexF64.(q[rows, :])
    spatial_loss = zeros(Float64, length(rows))
    fitting_loss = similar(spatial_loss)
    gappy_loss = similar(spatial_loss)
    fill!(fitting_loss, 0)
    fill!(gappy_loss, 0)
    for first in 1:chunk:size(spatial,1)
        voxels = first:min(first+chunk-1, size(spatial,1))
        block = ComplexF64.(S[voxels, :])
        H = cis.(2π .* (a * transpose(Float64.(spatial[voxels, :]))))
        residual = H - oracle * adjoint(block)
        fit_residual = (fitted-oracle) * adjoint(block)
        spatial_loss .+= vec(sum(abs2, residual; dims=2))
        fitting_loss .+= vec(sum(abs2, fit_residual; dims=2))
        gappy_loss .+= vec(sum(abs2, residual-fit_residual; dims=2))
    end
    nVox = size(spatial, 1)
    space2 = mean(spatial_loss) / nVox
    fit2 = mean(fitting_loss) / nVox
    gappy2 = mean(gappy_loss) / nVox
    return (; rows, space_error=sqrt(space2), fit_error=sqrt(fit2), gappy_error=sqrt(gappy2),
        decomposition_defect=gappy2-space2-fit2, gram_condition=cond(gram),
        evidence=:sampled_rows_full_voxels)
end

function adaptive_expansion_reason(diagnostic, epsilon, resolved, width, cap)
    if diagnostic.space_error <= 0.9epsilon || diagnostic.fit_error^2 > diagnostic.space_error^2
        return :fitting
    end
    return resolved > width && width == cap ? :parent : :snapshots
end

function adaptive_parent_resource_limit(temporal, spatial, options, arrayType)
    arrayType === Array && return options.hard_parent_limit
    available = floor(Int, options.memory_fraction * CUDA.total_memory())
    chunk = options.chunk_size
    # The full device basis is freed before q allocation. These are separate
    # peaks; phase features and the largest permitted tile are included.
    basis_fixed = sizeof(eltype(spatial))*length(spatial) + 2sizeof(Complex{eltype(spatial)})*chunk*options.max_snapshots
    fit_fixed = sizeof(eltype(temporal))*length(temporal) + 2sizeof(Complex{eltype(spatial)})*chunk*options.max_fit_points
    basis_limit = fld(max(0, available-basis_fixed), sizeof(Complex{eltype(spatial)})*size(spatial,1))
    fit_limit = fld(max(0, available-fit_fixed), sizeof(Complex{eltype(temporal)})*size(temporal,1))
    return min(options.hard_parent_limit, 8fld(min(basis_limit, fit_limit), 8))
end

function adaptive_phase_factors(temporal, spatial, nSam, nDyn, epsilon_H::Real, options;
    initial_parent, arrayType=CuArray)
    isfinite(epsilon_H) && 0 < epsilon_H < 1 || throw(ArgumentError("epsilon_H must lie in (0,1)"))
    size(temporal,1) == nSam*nDyn || throw(DimensionMismatch("readout layout"))
    size(temporal,2) == size(spatial,2) || throw(DimensionMismatch("phase features"))
    start = time_ns()
    K, J, cap = options.snapshots, min(options.fit_points, size(spatial,1)), initial_parent
    maximum_J = min(options.max_fit_points, options.fit_pool, size(spatial,1))
    limit = adaptive_parent_resource_limit(temporal, spatial, options, arrayType)
    trials = NamedTuple[]
    space = nothing
    reason = :attempt_budget
    for attempt in 1:options.max_attempts
        if (time_ns()-start)*1e-9 > options.max_seconds
            reason = :time_budget
            break
        end
        if cap > limit || cap > J
            reason = :memory_or_sampling_budget
            break
        end
        attempt_start = time_ns()
        rebuild = isnothing(space)
        if rebuild
            space = adaptive_snapshot_space(temporal, spatial, nSam, nDyn, K, cap, options, arrayType)
        end
        fitted = adaptive_fit_space(space.S0, temporal, spatial, J, options, arrayType)
        selection_s = diagnostic_s = 0.0
        diagnostic = nothing
        checks = nothing
        rank = nothing
        if fitted.status === :computed
            selection_s = @elapsed checks = adaptive_internal_ranks(temporal, spatial, fitted.q, fitted.S,
                epsilon_H, options, attempt)
            rank = checks.rank
        end
        if !isnothing(rank)
            q, S = fitted.q[:, 1:rank], fitted.S[:, 1:rank]
            trial = (; attempt, K, J, cap, r0=size(space.S0,2), space.resolved, rebuild,
                action=:accepted, fitted.condition, selection_s, diagnostic_s,
                base_e_H=sqrt(last(checks.a2)), selected_e_H=sqrt(checks.a2[rank]),
                selected_upper=sqrt(checks.a2[rank]+4checks.ase[rank]),
                elapsed_s=(time_ns()-attempt_start)*1e-9,
                snapshot_indices=space.indices, fitting_indices=fitted.points,
                space_stages=rebuild ? space.stages : Dict{String,Float64}(), fit_stages=fitted.stages,
                diagnostic)
            push!(trials, trial)
            return (; construction_status=:computed, empirical_status=:passed,
                error_metric=:global_relative_frobenius, certification_status=:not_requested,
                external_audit_status=:not_requested, epsilon_H, rank,
                q, S, K, J, cap, r0=size(space.S0,2), resource_parent_limit=limit, trials,
                indices=space.indices, snapshot_weights=space.weights, heldout_profiles=space.holdout,
                points=fitted.points, pool=fitted.pool, checks, space.orthogonality, space.fallback,
                singular_values=fitted.singular_values, elapsed_s=(time_ns()-start)*1e-9,
                failure_reason=:none)
        end
        cause = :fitting
        if fitted.status === :computed
            diagnostic_s = @elapsed diagnostic = adaptive_projection_diagnostic(temporal, spatial,
                fitted.q, fitted.S, options)
            cause = adaptive_expansion_reason(diagnostic, epsilon_H, space.resolved, size(space.S0,2), cap)
        end
        new_K, new_J, new_cap = K, J, cap
        if cause === :fitting && J < maximum_J
            new_J = min(2J, maximum_J)
            reason = :expand_J
        elseif cause === :parent && cap < limit
            new_cap = min(cap+options.parent_increment, space.resolved, limit, J)
            reason = new_cap > cap ? :expand_parent : :parent_budget
        elseif cause === :snapshots && K < options.max_snapshots
            new_K = min(2K, options.max_snapshots)
            reason = :expand_K
        else
            reason = Symbol(cause, "_budget")
        end
        trial = (; attempt, K, J, cap, r0=size(space.S0,2), space.resolved, rebuild, action=reason,
            condition=isfinite(fitted.condition) ? fitted.condition : nothing,
            selection_s, diagnostic_s,
            base_e_H=isnothing(checks) ? nothing : sqrt(last(checks.a2)),
            selected_e_H=nothing, selected_upper=nothing, elapsed_s=(time_ns()-attempt_start)*1e-9,
            snapshot_indices=space.indices, fitting_indices=fitted.points,
            space_stages=rebuild ? space.stages : Dict{String,Float64}(), fit_stages=fitted.stages,
            diagnostic)
        push!(trials, trial)
        (new_K, new_J, new_cap) == (K, J, cap) && break
        if attempt == options.max_attempts
            reason = :attempt_budget
            break
        end
        fitted = nothing
        checks = nothing
        (new_K != K || new_cap != cap) && (space = nothing)
        K, J, cap = new_K, new_J, new_cap
    end
    return (; construction_status=:budget_exhausted, empirical_status=:not_passed,
        error_metric=:global_relative_frobenius, certification_status=:not_requested,
        external_audit_status=:not_requested, epsilon_H,
        rank=0, q=nothing, S=nothing, K, J, cap, resource_parent_limit=limit, trials,
        elapsed_s=(time_ns()-start)*1e-9, failure_reason=reason)
end
