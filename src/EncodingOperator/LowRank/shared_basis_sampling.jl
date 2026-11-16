"""Select representative readouts in the sampled residual-phase geometry.

This is an empirical construction budget, not a full-time coverage bound.
The original readout indices determine the local rSVD seeds.
"""
function select_shared_readouts(spatial, temporal, nSam, nDyn, count; seed=1234)
    1 <= count <= nDyn || throw(ArgumentError("shared_readout_count must lie in 1:nDyn"))
    rng = MersenneTwister(seed)
    probes = rand(rng, 1:size(spatial, 1), 1024)
    points = Float64.(spatial[probes, :])
    points .-= mean(points; dims=1)
    F = eigen(Hermitian(points' * points / length(probes)))
    root = F.vectors * Diagonal(sqrt.(max.(F.values, 0))) * F.vectors'
    samples = unique(round.(Int, range(1, nSam; length=7)))
    nField = size(points, 2)
    features = zeros(Float64, nDyn, length(samples) * nField)
    for (j, sample) in enumerate(samples)
        rows = sample:nSam:nSam*nDyn
        features[:, (j-1)*nField+1:j*nField] .= Float64.(temporal[rows, :]) * root
    end
    next = argmax(vec(sum(abs2, features .- mean(features; dims=1); dims=2)))
    selected = Int[]
    distance = fill(Inf, nDyn)
    for _ in 1:count
        push!(selected, next)
        delta = vec(sum(abs2, features .- transpose(features[next, :]); dims=2))
        distance = min.(distance, delta)
        distance[selected] .= -Inf
        next = argmax(distance)
    end
    return sort!(selected)
end


function refit_shared_coefficients(temporal::Matrix{T}, spatial::Matrix{T}, basis;
    arrayType=Array, sample_count=512, chunk_size=4096, seed=1234,
) where {T<:AbstractFloat}
    nVox, rank = size(basis)
    count = min(sample_count, nVox)
    count >= rank || throw(ArgumentError("shared_fit_samples must cover the shared rank $rank"))
    selected = randperm(MersenneTwister(seed), nVox)[1:count]
    sampled = ComplexF64.(Array(basis[selected, :]))
    singular = svdvals(sampled)
    minimum(singular) > maximum(singular) * max(size(sampled)...) * eps(Float64) ||
        throw(ArgumentError("sampled shared spatial basis is numerically rank deficient"))
    fit_map = pinv(sampled)
    q = joint_fit_coefficients(temporal, spatial, selected, fit_map, chunk_size, arrayType)
    return q, (fit_indices=selected, fit_samples=count,
        fit_condition=maximum(singular)/minimum(singular))
end


function shared_basis_phase_check(q, basis, temporal, spatial;
    exclude_voxels=Int[], seed=9876, time_samples=2048, spatial_samples=127,
)
    rng = MersenneTwister(seed)
    rows = sort!(randperm(rng, size(q, 1))[1:min(time_samples, size(q, 1))])
    available = setdiff(1:size(basis, 1), exclude_voxels)
    disjoint = !isempty(available)
    isempty(available) && (available = collect(1:size(basis, 1)))
    voxels = available[randperm(rng, length(available))[1:min(spatial_samples, length(available))]]
    expected = cis.(2π .* (Float64.(temporal[rows, :]) * transpose(Float64.(spatial[voxels, :]))))
    actual = ComplexF64.(Array(q[rows, :])) * adjoint(ComplexF64.(Array(basis[voxels, :])))
    return (relative_error=norm(actual-expected)/norm(expected),
        temporal_samples=length(rows), voxels=length(voxels),
        spatial_training_disjoint=disjoint, seed,
        scope=:sampled_phase_error_not_image_error)
end


function sampled_shared_spatial_basis(times::Matrix{T}, fieldmap::Vector{T},
    bf::Matrix{T}, coefficients::Array{T,3};
    arrayType=Array, gpus=[0], readout_count=nothing, sample_count=512,
    L_rank=15, rank_max=128, tol=T(1e-2), seed=1234, chunk_size=4096,
    oversample=5, finalize=:svd, backend=:chunked, distributed=false,
    no_add_check=false, verbose=false,
) where {T<:AbstractFloat}
    nSam, nDyn = size(times)
    nVox = length(fieldmap)
    spatial = hcat(bf, fieldmap)
    temporal = hcat(permutedims(reshape(coefficients, size(bf, 2), :)), vec(times))
    selected = isnothing(readout_count) ? collect(1:nDyn) :
        select_shared_readouts(spatial, temporal, nSam, nDyn, readout_count; seed)
    nSelected = length(selected)
    basis, errors = if distributed
        workspace = DistributedRSVDWorkspace(fieldmap, bf, nSam, L_rank+oversample,
            L_rank, gpus; rsvd_backend=backend, rsvd_chunk=chunk_size, kernel_tiling=true)
        shared = nothing
        try
            shared = DistributedSharedSpatialBasis(workspace, nSelected, rank_max, tol)
            for (j, readout) in enumerate(selected)
                _, energy = perform_rsvd_multi_gpu!(workspace, times[:, readout],
                    coefficients[:, :, readout]; seed=seed+readout-1)
                update_distributed_shared_basis!(shared, workspace, j, energy;
                    no_add_check=no_add_check)
                if verbose @info("Sampled shared spatial update", readout, rank=shared.rank) end
            end
            gather_distributed_shared_basis(shared, workspace), copy(shared.errors)
        finally
            try
                release_distributed_workspaces!(workspace, shared)
            finally
                shutdown_distributed_workers!(workspace.workers)
                CUDA.device!(first(gpus))
            end
        end
    else
        prototype = arrayType(zeros(Complex{T}, 0))
        shared = SharedSpatialBasis(prototype, T, nVox, L_rank, nSelected, rank_max, tol)
        update = SharedBasisUpdateWorkspace(prototype, T, nVox, L_rank, rank_max)
        workspace = RSVDWorkspace(prototype, T, nSam, nVox, L_rank+oversample, chunk_size)
        field = arrayType(fieldmap)
        fields = arrayType(bf)
        for (j, readout) in enumerate(selected)
            time = arrayType(times[:, readout])
            coefficient = arrayType(coefficients[:, :, readout])
            if finalize === :gram
                _, energy = perform_rsvd(time, field, fields, coefficient,
                    nVox, nSam, L_rank, chunk_size, workspace;
                    seed=seed+readout-1, p_oversample=oversample,
                    rsvd_finalize=:gram, rsvd_backend=backend,
                    v_scaled=update.v_scaled, verbose)
            else
                _, singular, right = perform_rsvd(time, field, fields, coefficient,
                    nVox, nSam, L_rank, chunk_size, workspace;
                    seed=seed+readout-1, p_oversample=oversample,
                    rsvd_finalize=:svd, rsvd_backend=backend, verbose)
                update.v_scaled .= right .* reshape(singular, 1, L_rank)
                energy = T(real(dot(singular, singular)))
            end
            update_shared_basis!(shared, update, update.v_scaled, j, energy)
        end
        copy(@view(shared.basis[:, 1:shared.rank])), copy(shared.errors)
    end
    q, fit = refit_shared_coefficients(temporal, spatial, basis;
        arrayType, sample_count, chunk_size, seed)
    audit = shared_basis_phase_check(q, basis, temporal, spatial; exclude_voxels=fit.fit_indices)
    report = (; selected_readouts=selected, local_errors=errors, fit...,
        phase_check=audit, rank=size(basis, 2), seed,
        readout_selection=:sampled_phase_geometry,
        reconstruction_samples_reduced=false)
    return q, basis, errors, report
end
