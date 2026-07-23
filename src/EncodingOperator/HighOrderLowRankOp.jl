using CUDA
using LinearAlgebra
using NFFT

export HighOrderLowRankOp

mutable struct HighOrderLowRankWorkspace{AV}
    grid_flat_prod   :: AV
    grid_flat_ctprod :: AV
    k_signal         :: AV
    k_weighted       :: AV
    x_weighted       :: AV
    x_masked         :: AV
    x_l              :: AV
end

"""
Configuration and owned state for an optional optimized normal-operator backend.

The backend is initialized lazily by `normalOperator`, because constructing the
forward operator alone should not allocate one NFFT plan and workspace per GPU.
`operator` is kept here so that `HighOrderLowRankOp` owns the multi-GPU resources
and can release them with `close(op)`.
"""
mutable struct HighOrderNormalBackend
    distribution :: Symbol
    gpus          :: Vector{Int}
    verbose       :: Bool
    detailed_timing :: Bool
    operator      :: Union{Nothing,AbstractLinearOperator}
    weights2      :: Union{Nothing,AbstractVector}
end

"""
HighOrderLowRankOp implements a high-order field encoding operator based on SVD low-rank approximation.
Automatically separates 0th-order (outer demodulation) and 1st-order (NFFT) terms, and performs extremely fast SVD dimensionality reduction only on the remaining smooth high-order errors.
Supports automatic switching between CPU and GPU compute backends.

For a `CuArray` operator, set `normal_distribution=:channel` and pass two or
more `normal_gpus` (or reuse `gpus`) to make `normalOperator(W ∘ op)` construct
a channel-sharded multi-GPU normal backend lazily. Release that backend with
`close(op)` after reconstruction. Set `normal_detailed_timing=true` to accumulate
per-GPU upload, forward, adjoint, download, and worker wall times.
"""
mutable struct HighOrderLowRankOp{
    T, F1, F2, 
    P  <: AbstractNFFTPlan{T},
    AQ <: AbstractArray{Complex{T}, 2},
    AB <: AbstractArray{Complex{T}, 2},
    AC <: AbstractArray{Complex{T}, 2},
    W  <: HighOrderLowRankWorkspace,
    S  <: AbstractVector{Complex{T}}
    } <: HOOp{Complex{T}}
    nSam       :: Int
    nVox       :: Int
    nCha       :: Int
    nDyn       :: Int
    L          :: Int                    # SVD truncation rank
    
    q          :: AQ                     # [nSam*nDyn, shared_rank], Q_d = U_d*C_dᴴ
    basis      :: AB                     # [nVox, shared_rank]
    csm        :: AC                     # [nVox, nCha]
    
    nfftplan   :: P                      # one reusable CPU/GPU NFFT plan
    nfft_traj  :: Array{T,3}             # [nDim, nSam, nDyn], stored on CPU
    mask_idx   :: AbstractVector{Int32}  # Extracted 3D mask indices
    grid_size  :: Tuple                  # gridding size

    workspace  :: W
    normal_backend :: HighOrderNormalBackend

    nrow       :: Int
    ncol       :: Int
    symmetric  :: Bool
    hermitian  :: Bool
    prod!      :: Function
    tprod!     :: F1
    ctprod!    :: F2
    nprod      :: Int
    ntprod     :: Int
    nctprod    :: Int
    Mv         :: S                  # Dynamically adapted CPU/GPU Vector
    Mtu        :: S                  # Dynamically adapted CPU/GPU Vector
end
LinearOperators.storage_type(op::HighOrderLowRankOp) = typeof(op.Mv)


function HighOrderLowRankOp(
    grid        :: Grid{T}                                                             ,
    kspha       :: AbstractArray{T, 2}                                                 , 
    times       :: AbstractArray{T, 1}                                                 ;
    kwargs...          
    ) where T <: AbstractFloat
    kspha = reshape(kspha, size(kspha, 1), size(kspha, 2), 1)
    times = reshape(times, :, 1)
    return HighOrderLowRankOp(grid, kspha, times; kwargs...)
end

"""
kspha: [nTerm, nSam, nDyn]
times: [nSam, nDyn]
"""
function HighOrderLowRankOp(
    grid              :: Grid{T}                                                             ,
    kspha             :: AbstractArray{T, 3}                                                 , 
    times             :: AbstractArray{T, 2}                                                 ;
    fieldmap          :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm               :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask              :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    recon_terms       :: String                    = nothing                                 ,
    k_nominal         :: AbstractArray{T, 3}       = kspha[2:4, :, :]                        ,
    kspha_dt                                       = nothing                                 ,
    nBlock            :: Int64                     = 50                                      , 
    
    arrayType         :: Type{<:AbstractArray}     = Array                                   ,
    gpus              :: Vector{Int}               = [0]                                     ,
    L_rank            :: Int                       = 15                                      , 
    rsvd_seed         :: Int                       = 1234                                    ,
    rsvd_chunk        :: Int                       = 4096                                    , 
    rsvd_oversample   :: Int                       = 5                                       ,
    rsvd_finalize     :: Symbol                    = :svd                                    ,
    rsvd_backend      :: Symbol                    = :auto                                   ,
    rsvd_distribution :: Symbol                    = :auto                                   ,
    rsvd_fastmath     :: Bool                      = false                                   ,
    rsvd_detailed_timing:: Bool                    = false                                   ,
    shared_rank_max   :: Int                       = 128                                     ,
    shared_basis_tol  :: T                         = T(1e-2)                                 , 
    normal_distribution:: Symbol                   = :single                                ,
    normal_gpus       :: Union{Nothing,Vector{Int}} = nothing                                 ,
    normal_detailed_timing:: Bool                  = false                                   ,
    verbose           :: Bool                      = false                                   ,   
    
    kwargs...          
    ) where T <: AbstractFloat

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam, nDyn = size(kspha)
    nCha = size(csm)[end]
    nRow = nSam * nCha * nDyn
    nCol = prod(grid.matrixSize)
    nVox = sum(mask)

    fieldmap = ndims(fieldmap) == 2 ? reshape(fieldmap, nX, nY, 1) : fieldmap
    csm      = ndims(csm) == 3      ? reshape(csm, nX, nY, 1, nCha) : csm
    mask     = ndims(mask) == 2     ? reshape(mask, nX, nY, 1) : mask

    @info "HighOrderLowRankOp Setup: ArrayType=$arrayType, nRow=$(nSam*nCha), nCol=$(prod(grid.matrixSize)), nVox in mask=$nVox, gpus=$gpus"

    @assert nTerm              in [9, 16]       "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3             "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY, nZ)  "FieldMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(csm)[1:3]     == (nX, nY, nZ)  "Coil-SensitivityMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(mask)         == (nX, nY, nZ)  "Mask must have same size as $((nX, nY, nZ)) in grid"
    @assert size(times)        == (nSam, nDyn)  "times must have size $((nSam, nDyn))"
    rsvd_finalize in (:svd, :gram) || throw(ArgumentError("rsvd_finalize must be :svd or :gram, " * "got $rsvd_finalize"))
    rsvd_backend in (:auto, :chunked, :adjoint_kernel, :kernel) || throw(ArgumentError("Unsupported rsvd_backend=$rsvd_backend"))
    rsvd_distribution in (:auto, :single, :voxel) || throw(ArgumentError("Unsupported rsvd_distribution=$rsvd_distribution; " * "expected :auto, :single, or :voxel"))
    normal_distribution in (:single, :channel) || throw(ArgumentError("Unsupported normal_distribution=$normal_distribution; expected :single or :channel"))
    isempty(gpus) && throw(ArgumentError("gpus must contain at least one GPU id"))
    length(unique(gpus)) == length(gpus) || throw(ArgumentError("gpus contains duplicate GPU ids: $gpus"))

    is_gpu = arrayType == CuArray
    primary_gpu = first(gpus)
    normal_gpu_ids = normal_gpus === nothing ? copy(gpus) : copy(normal_gpus)
    isempty(normal_gpu_ids) && throw(ArgumentError("normal_gpus must contain at least one GPU id"))
    length(unique(normal_gpu_ids)) == length(normal_gpu_ids) || throw(ArgumentError("normal_gpus contains duplicate GPU ids: $normal_gpu_ids"))
    first(normal_gpu_ids) == primary_gpu || throw(ArgumentError("the first normal GPU ($(first(normal_gpu_ids))) must match the operator GPU ($primary_gpu)"))
    if normal_distribution === :channel
        is_gpu || throw(ArgumentError("normal_distribution=:channel requires arrayType=CuArray"))
        length(normal_gpu_ids) >= 2 || throw(ArgumentError("normal_distribution=:channel requires at least two GPUs"))
    end
    rsvd_backend = rsvd_backend === :auto ? (is_gpu ? :kernel : :chunked) : rsvd_backend
    if rsvd_fastmath && !(is_gpu && rsvd_backend in (:kernel, :adjoint_kernel))
        throw(ArgumentError("rsvd_fastmath=true requires arrayType=CuArray and a kernel rSVD backend"))
    end
    use_distributed_rsvd =
    rsvd_distribution === :voxel || (rsvd_distribution === :auto && is_gpu && length(gpus) > 1 && rsvd_finalize === :gram && rsvd_backend === :kernel)

    if use_distributed_rsvd
        is_gpu || throw(ArgumentError("rsvd_distribution=:voxel requires arrayType=CuArray"))
        rsvd_finalize === :gram || throw(ArgumentError("Multi-GPU voxel-distributed rSVD currently requires " * "rsvd_finalize=:gram"))
        rsvd_backend === :kernel || throw(ArgumentError("Multi-GPU voxel-distributed rSVD currently requires " * "rsvd_backend=:kernel"))
    elseif rsvd_detailed_timing
        throw(ArgumentError("rsvd_detailed_timing=true currently requires multi-GPU voxel-distributed rSVD"))
    end
    if is_gpu CUDA.device!(primary_gpu) end
    
    if verbose
        @info(
            "rSVD execution configuration",
            backend=rsvd_backend,
            distribution=use_distributed_rsvd ? :voxel : :single,
            fastmath=rsvd_fastmath,
            detailed_timing=rsvd_detailed_timing,
            kspha_layout=use_distributed_rsvd ? :sample_major : :term_major,
            gpus=gpus,
            primary_gpu=is_gpu ? primary_gpu : nothing,
            normal_distribution,
            normal_gpus=normal_gpu_ids,
            normal_detailed_timing,
        )
    end


    # -------------------------------------------------------------
    # Host-side data preparation
    # -------------------------------------------------------------
    mask_host = vec(mask)   # [prod(MatrixSize)]
    kspha = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    
    csm_host = Matrix{Complex{T}}(reshape(csm, :, nCha)[mask_host .!= 0, :])   # [nVox, nCha]
    fieldmap_host = Vector{T}(vec(fieldmap)[mask_host .!= 0])                  # [nVox]
    bf_host = Matrix{T}(basisfunc_spha(grid.x[mask_host .!= 0], grid.y[mask_host .!= 0], grid.z[mask_host .!= 0], collect(1:nTerm)))  # compute basis functions (spherical harmonics)
    times_host = Matrix{T}(times)
    mask_idx_host = Int32.(findall(mask_host .!= 0))
    
    if nTerm >= 5
        kspha_err_host = Array{T,3}(kspha[5:end, :, :])
        bf_err_host = Matrix{T}(bf_host[:, 5:end])
    else
        kspha_err_host = zeros(T, 1, nSam, nDyn)
        bf_err_host = zeros(T, nVox, 1)
    end


    @info "nSam=$nSam, nVox=$nVox, nDyn=$nDyn, nTerm=$nTerm, nCha=$nCha, nBlock=$nBlock"

    @assert shared_rank_max > 0 "shared_rank_max must be positive"
    @assert rsvd_chunk > 0 "rsvd_chunk must be positive for chunked rSVD"
    @assert L_rank > 0 "L_rank must be positive"
    @assert rsvd_oversample >= 0 "rsvd_oversample must be non-negative"

    L_total = L_rank + rsvd_oversample

    @assert L_total <= min(nSam, nVox) "L_rank + rsvd_oversample exceeds matrix dimensions"

    rsvd_chunk = min(rsvd_chunk, nVox)

    max_possible_shared_rank = min(nVox, L_rank * nDyn)
    effective_shared_rank_max = min(shared_rank_max, max_possible_shared_rank)
    if verbose && effective_shared_rank_max != shared_rank_max
        @info("Clamping shared_rank_max", requested=shared_rank_max, effective=effective_shared_rank_max, max_possible=max_possible_shared_rank)
    end


    if use_distributed_rsvd
        q_host, basis_host, shared_rank, shared_errors = let
            distributed_workspace = DistributedRSVDWorkspace(fieldmap_host, bf_err_host, nSam, L_total, L_rank, gpus)
            distributed_shared = nothing
            try
                distributed_shared = DistributedSharedSpatialBasis(distributed_workspace, nDyn, effective_shared_rank_max, shared_basis_tol)
        
                # [nSam, L_rank, nDyn]
                u_host = zeros(Complex{T}, nSam, L_rank, nDyn)
        
                rsvd_time = 0.0
                shared_time = 0.0
                rsvd_timing = DistributedRSVDTiming(detailed=rsvd_detailed_timing)
                for dyn = 1:nDyn
                    times_dyn = times_host[:, dyn]
                    kspha_err_dyn = kspha_err_host[:, :, dyn]
        
                    t0 = time_ns()
                    u_trunc, total_energy = perform_rsvd_multi_gpu!(
                        distributed_workspace,
                        times_dyn,
                        kspha_err_dyn;
                        seed=rsvd_seed + dyn - 1,
                        timing=rsvd_timing,
                        fastmath=rsvd_fastmath,
                    )
                    rsvd_time += (time_ns() - t0) * 1e-9

                    @views u_host[:, :, dyn] .= u_trunc

                    t0 = time_ns()
                    relative_error, n_added = update_distributed_shared_basis!(distributed_shared, distributed_workspace, dyn, total_energy)
                    shared_time += (time_ns() - t0) * 1e-9

                    if verbose @info("Distributed shared spatial basis update", dyn=dyn, added=n_added, rank=distributed_shared.rank, relative_error=relative_error) end
                end
                profiled_rsvd_time = distributed_rsvd_total_time(rsvd_timing)
                timing_denominator = max(profiled_rsvd_time, eps(Float64))
                @info(
                    "Distributed rSVD phase timing",
                    n_dynamic=rsvd_timing.n_calls,
                    fastmath=rsvd_fastmath,
                    kspha_layout=:sample_major,
                    forward_time=rsvd_timing.forward_time,
                    qr_time=rsvd_timing.qr_time,
                    adjoint_gram_time=rsvd_timing.adjoint_gram_time,
                    finalize_time=rsvd_timing.finalize_time,
                    profiled_total=profiled_rsvd_time,
                    outer_total=rsvd_time,
                    unaccounted_time=max(rsvd_time - profiled_rsvd_time, 0.0),
                    mean_per_dynamic_ms=1e3 * profiled_rsvd_time / max(rsvd_timing.n_calls, 1),
                    forward_fraction=rsvd_timing.forward_time / timing_denominator,
                    qr_fraction=rsvd_timing.qr_time / timing_denominator,
                    adjoint_gram_fraction=rsvd_timing.adjoint_gram_time / timing_denominator,
                    finalize_fraction=rsvd_timing.finalize_time / timing_denominator,
                )
                if rsvd_timing.detailed
                    forward_denominator = max(rsvd_timing.forward_time, eps(Float64))
                    adjoint_denominator = max(rsvd_timing.adjoint_gram_time, eps(Float64))
                    @info(
                        "Distributed rSVD forward detail",
                        aggregation=:maximum_per_stage_across_gpus,
                        transpose_time=rsvd_timing.transpose_time,
                        upload_time=rsvd_timing.forward_upload_time,
                        sketch_time=rsvd_timing.forward_sketch_time,
                        kernel_time=rsvd_timing.forward_kernel_time,
                        download_time=rsvd_timing.forward_download_time,
                        reduce_time=rsvd_timing.forward_reduce_time,
                        transpose_fraction=rsvd_timing.transpose_time / forward_denominator,
                        upload_fraction=rsvd_timing.forward_upload_time / forward_denominator,
                        sketch_fraction=rsvd_timing.forward_sketch_time / forward_denominator,
                        kernel_fraction=rsvd_timing.forward_kernel_time / forward_denominator,
                        download_fraction=rsvd_timing.forward_download_time / forward_denominator,
                        reduce_fraction=rsvd_timing.forward_reduce_time / forward_denominator,
                    )
                    @info(
                        "Distributed rSVD adjoint/Gram detail",
                        aggregation=:maximum_per_stage_across_gpus,
                        upload_time=rsvd_timing.adjoint_upload_time,
                        kernel_time=rsvd_timing.adjoint_kernel_time,
                        gram_time=rsvd_timing.gram_time,
                        download_time=rsvd_timing.adjoint_download_time,
                        reduce_time=rsvd_timing.adjoint_reduce_time,
                        upload_fraction=rsvd_timing.adjoint_upload_time / adjoint_denominator,
                        kernel_fraction=rsvd_timing.adjoint_kernel_time / adjoint_denominator,
                        gram_fraction=rsvd_timing.gram_time / adjoint_denominator,
                        download_fraction=rsvd_timing.adjoint_download_time / adjoint_denominator,
                        reduce_fraction=rsvd_timing.adjoint_reduce_time / adjoint_denominator,
                    )
                end
                @info "Distributed setup timing" rsvd_time shared_time total=rsvd_time+shared_time
        
                shared_rank_local = distributed_shared.rank
                errors_local = copy(distributed_shared.errors)
        
                basis_host_local = gather_distributed_shared_basis(distributed_shared, distributed_workspace)
        
                # q_d = U_d * C_dᴴ
                q_host_local = zeros(Complex{T}, nSam * nDyn, shared_rank_local)
                q_dyn = zeros(Complex{T}, nSam, shared_rank_local)
                for dyn = 1:nDyn
                    rows = ((dyn - 1) * nSam + 1):(dyn * nSam)
        
                    U_dyn = @view u_host[:, :, dyn]
                    C_dyn = @view distributed_shared.coeff[1:shared_rank_local, :, dyn]
        
                    mul!(q_dyn, U_dyn, adjoint(C_dyn))
                    @views q_host_local[rows, :] .= q_dyn
                end
                (q_host_local, basis_host_local, shared_rank_local, errors_local)
            finally
                try
                    release_distributed_workspaces!(distributed_workspace, distributed_shared)
                finally
                    shutdown_distributed_workers!(distributed_workspace.workers)
                end
            end
        end
        if verbose @info("Multi-GPU rSVD workspace released", setup_gpus=gpus, operator_gpu=primary_gpu) end
        CUDA.device!(primary_gpu)
        q = arrayType(q_host)
        basis = arrayType(basis_host)
    else
        if is_gpu
            CUDA.device!(primary_gpu)
        end
        times_device = arrayType(times_host)
        fieldmap_device = arrayType(fieldmap_host)
        bf_err_device = arrayType(bf_err_host)
        kspha_err_device = arrayType(kspha_err_host)
    
        u = arrayType(zeros(Complex{T}, nSam, L_rank, nDyn))
    
        shared = SharedSpatialBasis(u, T, nVox, L_rank, nDyn, effective_shared_rank_max, shared_basis_tol)
        shared_workspace = SharedBasisUpdateWorkspace(u, T, nVox, L_rank, effective_shared_rank_max)
        rsvd_workspace = RSVDWorkspace(u, T, nSam, nVox, L_total, rsvd_chunk)
    
        for dyn = 1:nDyn
            times_dyn = times_device[:, dyn]
            kspha_err_dyn = kspha_err_device[:, :, dyn]
    
            v_scaled = shared_workspace.v_scaled
    
            if rsvd_finalize === :gram
                u_trunc, total_energy = perform_rsvd(
                    times_dyn, fieldmap_device, bf_err_device, kspha_err_dyn,
                    nVox, nSam, L_rank, rsvd_chunk, rsvd_workspace;
                    seed=rsvd_seed + dyn - 1, p_oversample=rsvd_oversample,
                    rsvd_finalize=:gram, rsvd_backend=rsvd_backend,
                    rsvd_fastmath=rsvd_fastmath,
                    v_scaled=v_scaled, verbose=verbose)
            else
                u_trunc, s_trunc, v_trunc = perform_rsvd(
                    times_dyn, fieldmap_device, bf_err_device, kspha_err_dyn,
                    nVox, nSam, L_rank, rsvd_chunk, rsvd_workspace;
                    seed=rsvd_seed + dyn - 1, p_oversample=rsvd_oversample, 
                    rsvd_finalize=:svd, rsvd_backend=rsvd_backend,
                    rsvd_fastmath=rsvd_fastmath,
                    verbose=verbose)
    
                v_scaled .= v_trunc .* reshape(s_trunc, 1, L_rank)
                total_energy = T(real(dot(s_trunc, s_trunc)))
            end
    
            @views u[:, :, dyn] .= u_trunc
            relative_error, n_added = update_shared_basis!(shared, shared_workspace, v_scaled, dyn, total_energy)
            if verbose @info("Shared spatial basis update", dyn=dyn, added=n_added, rank=shared.rank, relative_error=relative_error) end
        end
    
        shared_rank = shared.rank
        shared_errors = copy(shared.errors)
    
        q = similar(u, Complex{T}, nSam * nDyn, shared_rank)
        q_dyn = similar(u, Complex{T}, nSam, shared_rank)
    
        for dyn = 1:nDyn
            rows = ((dyn - 1) * nSam + 1):(dyn * nSam)
    
            U_dyn = @view u[:, :, dyn]
            C_dyn = @view shared.coeff[1:shared_rank, :, dyn]
    
            mul!(q_dyn, U_dyn, adjoint(C_dyn))
            @views q[rows, :] .= q_dyn
        end
    
        basis = copy(@view shared.basis[:, 1:shared_rank])
    end

    if is_gpu CUDA.device!(primary_gpu) end
    csm = arrayType(csm_host)
    mask_idx = arrayType(mask_idx_host)
    
    @info("Shared spatial basis complete", rank=shared_rank, max_error=maximum(shared_errors), mean_error=sum(shared_errors) / length(shared_errors))
    @info("Global temporal basis ready", shared_rank=shared_rank, nPoint=nSam * nDyn, size=size(q))
    if verbose && use_distributed_rsvd @info("Multi-GPU rSVD workspace released", setup_gpus=gpus, operator_gpu=primary_gpu) end


    if nZ == 1
        MatrixSize = (nX, nY)
        scale = inv(T(min(grid.Δx, grid.Δy)))
        k_range = 2:3
    else
        MatrixSize = (nX, nY, nZ)
        scale = inv(T(min(grid.Δx, grid.Δy, grid.Δz)))
        k_range = 2:4
    end

    nfft_traj = Array{T,3}(kspha[k_range, :, :] ./ scale)
    nfft_nodes = reshape(nfft_traj, length(k_range), nSam * nDyn) # dyn1 [all samples]、dyn2 [all samples] ……
    nfftplan = plan_nfft(arrayType, nfft_nodes, MatrixSize; m=3, σ=1.25)
    @assert eltype(nfftplan) == Complex{T} "NFFT plan precision must match the operator precision"
    if verbose @info("Global NFFT plan ready", trajectory_eltype=eltype(nfft_nodes), plan_eltype=eltype(nfftplan), nPoint=nSam * nDyn, nfft_per_forward=shared_rank * nCha, previous_nfft_per_forward=nDyn * L_rank * nCha) end

    nGrid = prod(grid.matrixSize)
    workspace = HighOrderLowRankWorkspace(
        fill!(similar(q, Complex{T}, nGrid), 0),
        similar(q, Complex{T}, nGrid),
        similar(q, Complex{T}, nSam * nDyn),
        similar(q, Complex{T}, nSam * nDyn),
        similar(q, Complex{T}, nVox),
        similar(q, Complex{T}, nVox),
        similar(q, Complex{T}, nVox),
    )
    normal_backend = HighOrderNormalBackend(
        normal_distribution,
        normal_gpu_ids,
        verbose,
        normal_detailed_timing,
        nothing,
        nothing,
    )

    Mv = Mtu = arrayType(Vector{Complex{T}}(undef, 0))  
    return HighOrderLowRankOp{T, Nothing, typeof(ctprod!), typeof(nfftplan), typeof(q), typeof(basis), typeof(csm), typeof(workspace), typeof(Mv)}(
        nSam, nVox, nCha, nDyn, L_rank,
        q, basis, csm,
        nfftplan, nfft_traj, mask_idx, grid.matrixSize, workspace, normal_backend,
        nRow, nCol, false, false, prod!, nothing, ctprod!,
        0, 0, 0, 
        Mv, Mtu,
    )
end

# -------------------------------------------------------------------------
# Fused Kernels
# -------------------------------------------------------------------------
# Kernel fusion: Computes x_masked * spatial_factor * csm and writes to the grid.
function kernel_scatter_csm!(grid_flat, x_w, csm, c, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds idx = mask_idx[v]
        @inbounds grid_flat[idx] = x_w[v] * csm[v, c]
    end
    return nothing
end

function kernel_gather_csm!(x_l, img_flat, csm, c, mask_idx, nVox)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= nVox
        @inbounds idx = mask_idx[v]
        @inbounds x_l[v] += img_flat[idx] * conj(csm[v, c])
    end
    return nothing
end


# -------------------------------------------------------------------------
# Zero-allocation prod! and ctprod!
# -------------------------------------------------------------------------
function prod!(y::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T

    workspace  = op.workspace
    grid_flat  = workspace.grid_flat_prod
    k_signal   = workspace.k_signal
    x_weighted = workspace.x_weighted

    x_masked = length(x) == prod(op.grid_size) ? view(x, op.mask_idx) : x

    nPoint = op.nSam * op.nDyn
    y_global = reshape(y, nPoint, op.nCha)
    fill!(y_global, zero(Complex{T}))
    
    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
    nfft_img  = reshape(grid_flat, nfft_dims)
    
    shared_rank = size(op.basis, 2)
    basis = op.basis

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    for r = 1:shared_rank
        @views @. x_weighted = x_masked * conj(basis[:, r])
        
        for c = 1:op.nCha
            if grid_flat isa CuArray
                @cuda threads=threads blocks=blocks_vox kernel_scatter_csm!(grid_flat, x_weighted, op.csm, c, op.mask_idx, op.nVox)
            else
                @views @. grid_flat[op.mask_idx] = x_weighted * op.csm[:, c]
            end
            
            mul!(k_signal, op.nfftplan, nfft_img)
            @views @. y_global[:, c] += k_signal * op.q[:, r]
        end
    end
    
    return y
end

function ctprod!(x::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, y::AbstractVector{Complex{T}}) where T

    nPoint = op.nSam * op.nDyn
    y_global = reshape(y, nPoint, op.nCha)

    workspace  = op.workspace
    grid_flat  = workspace.grid_flat_ctprod
    k_weighted = workspace.k_weighted
    x_masked   = workspace.x_masked
    x_l        = workspace.x_l

    fill!(x_masked, zero(Complex{T}))

    nfft_dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
    nfft_img  = reshape(grid_flat, nfft_dims)

    shared_rank = size(op.basis, 2)
    basis = op.basis

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    for r = 1:shared_rank
        fill!(x_l, zero(Complex{T}))
            
        for c = 1:op.nCha
            @views @. k_weighted = y_global[:, c] * conj(op.q[:, r])
            mul!(nfft_img, adjoint(op.nfftplan), k_weighted)
            
            if grid_flat isa CuArray
                @cuda threads=threads blocks=blocks_vox kernel_gather_csm!(x_l, grid_flat, op.csm, c, op.mask_idx, op.nVox)
            else
                @views @. x_l += grid_flat[op.mask_idx] * conj(op.csm[:, c])
            end
        end
        @views @. x_masked += x_l * basis[:, r]
        
    end
    
    if length(x) == prod(op.grid_size)
        fill!(x, zero(Complex{T}))
        view(x, op.mask_idx) .= x_masked
    else
        x .= x_masked
    end
    
    return x
end

# -------------------------------------------------------------------------
# Adjoint Operator Definition
# -------------------------------------------------------------------------
Base.eltype(::HighOrderLowRankOp{T}) where T = Complex{T}

function Base.adjoint(op::HighOrderLowRankOp{T}) where T
    return LinearOperator{Complex{T}}(
                              op.ncol, 
                              op.nrow, 
                              op.symmetric, 
                              op.hermitian,
                              (res, y) -> ctprod!(res, op, y), 
                              nothing, 
                              (res, x) -> prod!(res, op, x),
                              S=typeof(op.Mv))
  end

function Base.size(op::HighOrderLowRankOp)
    return (op.nSam * op.nCha * op.nDyn, prod(op.grid_size))
end

function Base.size(op::HighOrderLowRankOp, dim::Int)
    return dim == 1 ? op.nSam * op.nCha * op.nDyn : prod(op.grid_size)
end

import LinearAlgebra: mul!
function mul!(y::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T
    prod!(y, op, x)
end

function mul!(
    y::AbstractVector{Complex{T}},
    op::HighOrderLowRankOp{T},
    x::AbstractVector{Complex{T}},
    α,
    β,
) where T
    if iszero(β)
        prod!(y, op, x)
        α == one(α) || (@. y = α * y)
    else
        y_tmp = similar(y)
        prod!(y_tmp, op, x)
        @. y = α * y_tmp + β * y
    end

    return y
end

function Base.:*(op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T
    y = fill!(similar(op.q, Complex{T}, op.nSam * op.nCha * op.nDyn), 0)
    mul!(y, op, x)
    return y
end
