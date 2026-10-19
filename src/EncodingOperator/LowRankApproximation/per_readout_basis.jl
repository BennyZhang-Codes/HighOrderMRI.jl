"""
Build independent `U` and `VΣ` factors for each readout, without shared compression.

Rows of `q` contain each readout's temporal factor; consecutive `L_rank` columns
of `basis` contain its spatial factor. The phase matrix is `U * (VΣ)'`.
"""
function per_readout_basis(
    times_host::Matrix{T},
    fieldmap_host::Vector{T},
    bf_err_host::Matrix{T},
    kspha_err_host::Array{T,3};
    arrayType::Type{<:AbstractArray},
    gpus::Vector{Int},
    L_rank::Int,
    rsvd_seed::Int,
    rsvd_chunk::Int,
    rsvd_oversample::Int,
    rsvd_finalize::Symbol,
    rsvd_backend::Symbol,
    use_distributed_rsvd::Bool,
    verbose::Bool,
    host_storage::Bool=false,
    gather_mode::Symbol=:pipeline,
    regenerate::Bool=false,
    host_cache_bytes::Int=0,
) where {T<:AbstractFloat}
    nSam, nDyn = size(times_host)
    nVox = length(fieldmap_host)
    L_total = L_rank + rsvd_oversample
    primary_gpu = first(gpus)

    if use_distributed_rsvd
        q_host = Matrix{Complex{T}}(undef, nSam * nDyn, L_rank)
        source = regenerate ? RegeneratedSpatialBasis(times_host, fieldmap_host,
            bf_err_host, kspha_err_host, L_rank, L_total, host_cache_bytes) : nothing
        basis_host = source === nothing ? Matrix{Complex{T}}(undef, nVox, L_rank * nDyn) : source.host_basis
        workspace = DistributedRSVDWorkspace(
            fieldmap_host, bf_err_host, nSam, L_total, L_rank, gpus;
            rsvd_backend, rsvd_chunk,
        )
        timing = DistributedRSVDTiming()
        gather_s = 0.0
        started_setup = time_ns()
        gather = nothing
        try
            if gather_mode === :pipeline && size(basis_host, 2) > 0
                gather = DistributedFactorGather(basis_host, workspace)
            end
            host_parts = gather === nothing && size(basis_host, 2) > 0 ? distributed_rsvd_host_buffers(workspace) : nothing
            factorization = regenerate ? Ref{Any}() : nothing
            for dyn in 1:nDyn
                host_slot = source === nothing ? dyn : source.host_slot[dyn]
                u_trunc, _ = perform_rsvd_multi_gpu!(
                    workspace, times_host[:, dyn], kspha_err_host[:, :, dyn];
                    seed=rsvd_seed + dyn - 1, timing,
                    v_read_finished=gather === nothing ? nothing : gather.v_read_finished,
                    materialize=host_slot != 0, factorization_ref=factorization,
                )
                if source !== nothing
                    @views source.Q[:, :, dyn] .= factorization[].Q
                    @views source.Z[:, :, dyn] .= factorization[].Z
                end
                rows = ((dyn - 1) * nSam + 1):(dyn * nSam)
                columns = ((host_slot - 1) * L_rank + 1):(host_slot * L_rank)
                @views q_host[rows, :] .= u_trunc
                started_gather = time_ns()
                if host_slot == 0
                    # Uncached factors are represented only by Q/Z.
                elseif gather === nothing
                    gather_distributed_v_scaled!(view(basis_host, :, columns), workspace, host_parts)
                else
                    submit_factor_gather!(gather, host_slot)
                end
                gather_s += (time_ns() - started_gather) * 1e-9
                if verbose && (dyn == 1 || dyn % 64 == 0 || dyn == nDyn)
                    @info("Independent per-readout setup progress", readout=dyn, total=nDyn)
                end
            end
            started_gather = time_ns()
            gather === nothing || drain_factor_gather!(gather)
            gather_s += (time_ns() - started_gather) * 1e-9
            if verbose
                @info("Per-readout rSVD phase timing", forward_time=timing.forward_time,
                    qr_time=timing.qr_time, adjoint_gram_time=timing.adjoint_gram_time,
                    finalize_time=timing.finalize_time, gather_time=gather_s,
                    outer_time=(time_ns() - started_setup) * 1e-9)
                if gather !== nothing
                    @info("Factor gather pipeline timing", buffers_per_gpu=2,
                        host_write_worker_s=gather.host_write_s,
                        download_wait_worker_s=gather.download_wait_s,
                        device_copy_worker_s=gather.device_copy_s)
                end
            end
        finally
            try
                gather === nothing || close(gather)
            finally
                try
                    release_distributed_workspaces!(workspace)
                finally
                    shutdown_distributed_workers!(workspace.workers)
                end
            end
        end
        CUDA.device!(primary_gpu)
        return arrayType(q_host), source === nothing ?
            (host_storage ? basis_host : arrayType(basis_host)) : source
    end

    arrayType == CuArray && CUDA.device!(primary_gpu)
    times_device = arrayType(times_host)
    fieldmap_device = arrayType(fieldmap_host)
    bf_err_device = arrayType(bf_err_host)
    kspha_err_device = arrayType(kspha_err_host)
    q = similar(times_device, Complex{T}, nSam * nDyn, L_rank)
    basis = host_storage ? Matrix{Complex{T}}(undef, nVox, L_rank * nDyn) :
            similar(times_device, Complex{T}, nVox, L_rank * nDyn)
    v_scaled = similar(times_device, Complex{T}, nVox, L_rank)
    workspace = RSVDWorkspace(q, T, nSam, nVox, L_total, rsvd_chunk)
    for dyn in 1:nDyn
        times_dyn = times_device[:, dyn]
        kspha_err_dyn = kspha_err_device[:, :, dyn]
        if rsvd_finalize === :gram
            u_trunc, _ = perform_rsvd(
                times_dyn, fieldmap_device, bf_err_device, kspha_err_dyn,
                nVox, nSam, L_rank, rsvd_chunk, workspace;
                seed=rsvd_seed + dyn - 1, p_oversample=rsvd_oversample,
                rsvd_finalize, rsvd_backend, v_scaled, verbose,
            )
        else
            u_trunc, s_trunc, v_trunc = perform_rsvd(
                times_dyn, fieldmap_device, bf_err_device, kspha_err_dyn,
                nVox, nSam, L_rank, rsvd_chunk, workspace;
                seed=rsvd_seed + dyn - 1, p_oversample=rsvd_oversample,
                rsvd_finalize, rsvd_backend, verbose,
            )
            v_scaled .= v_trunc .* reshape(s_trunc, 1, L_rank)
        end
        rows = ((dyn - 1) * nSam + 1):(dyn * nSam)
        columns = ((dyn - 1) * L_rank + 1):(dyn * L_rank)
        @views q[rows, :] .= u_trunc
        if host_storage
            copyto!(basis, (dyn - 1) * nVox * L_rank + 1, v_scaled, 1, nVox * L_rank)
        else
            @views basis[:, columns] .= v_scaled
        end
    end
    return q, basis
end
