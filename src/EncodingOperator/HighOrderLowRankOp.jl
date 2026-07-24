using CUDA
using LinearAlgebra
using NFFT

export HighOrderLowRankOp, @rebuild_HOOp

"""
    @rebuild_HOOp variable constructor_expression

Safely replace a large high-order operator stored in `variable`.

If `variable` is already defined, its current value is closed when it supports
`Base.close`, and the caller's binding is then set to `nothing`. A full Julia
GC runs only after all temporary references to the old value have left scope.
Finally, `constructor_expression` is evaluated and assigned to `variable`.

The constructor expression is deliberately evaluated last, preventing the old
and new operators from coexisting during setup.

# Example

```julia
@rebuild_HOOp HOOp HighOrderLowRankOp(
    grid,
    kspha,
    times;
    arrayType=CuArray,
    gpus=[1, 2, 3],
)
```

The first argument must be a plain variable name. In a local scope, its type
must permit assignment of `nothing`.
"""
macro rebuild_HOOp(variable, constructor_expression)
    variable isa Symbol || throw(ArgumentError(
        "@rebuild_HOOp expects a variable name as its first argument",
    ))

    return esc(quote
        if @isdefined $variable
            let old_operator = $variable
                if old_operator !== nothing &&
                   Base.applicable(Base.close, old_operator)
                    Base.close(old_operator)
                end
            end
            $variable = nothing
        end

        # No temporary or caller binding now retains the old operator. Return
        # its arrays to CUDA.jl's pool before evaluating the new constructor.
        Base.GC.gc(true)
        $variable = $constructor_expression
    end)
end

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
    operator      :: Union{Nothing,AbstractLinearOperator}
    weights2      :: Union{Nothing,AbstractVector}
end

"""
High-order field encoding operator using a low-rank approximation of the
off-resonance and higher-order phase terms.

The zeroth-order term is applied as a temporal modulation, the first-order
terms are handled by the NFFT trajectory, and the remaining smooth phase is
represented by one global spatial basis shared by all dynamics. See the
`HighOrderLowRankOp` constructor documentation for input dimensions and
configuration options.
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
    nfft_traj  :: Array{T,3}             # signed normalized NFFT nodes [nDim, nSam, nDyn]
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


"""
    HighOrderLowRankOp(
        grid::Grid{T},
        kspha::AbstractArray{T,2},
        times::AbstractArray{T,1};
        kwargs...,
    ) where T<:AbstractFloat

# Description

Construct a single-dynamic high-order low-rank encoding operator. This is a
convenience overload: `kspha` and `times` are reshaped to add a singleton
dynamic dimension and are then passed to the full constructor.

# Arguments

* `grid::Grid{T}`                      - Cartesian reconstruction grid.
* `kspha::AbstractArray{T, 2}`         - [nTerm, nSam], coefficients of field
  dynamics. `nTerm` must be `9` (up to second order) or `16` (up to third
  order).
* `times::AbstractArray{T, 1}`         - [nSam], sampling time points.
* `kwargs...`                          - Keywords accepted by the dynamic-data
  constructor below.
"""
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
    HighOrderLowRankOp(
        grid::Grid{T},
        kspha::AbstractArray{T,3},
        times::AbstractArray{T,2};
        fieldmap=zeros(T, grid.matrixSize...),
        csm=ones(Complex{T}, grid.matrixSize..., 1),
        mask=trues(grid.matrixSize...),
        recon_terms=nothing,
        k_nominal=kspha[2:4, :, :],
        arrayType=Array,
        gpus=[0],
        L_rank=15,
        rsvd_seed=1234,
        rsvd_chunk=4096,
        rsvd_oversample=5,
        rsvd_finalize=:svd,
        rsvd_backend=:auto,
        rsvd_distribution=:auto,
        shared_rank_max=128,
        shared_basis_tol=T(1e-2),
        normal_distribution=:single,
        nfft_center_correction=true,
        verbose=false,
    ) where T<:AbstractFloat

# Description

Construct a high-order low-rank encoding operator for dynamic non-Cartesian
MRI. For each dynamic, rSVD approximates the off-resonance and higher-order
encoding matrix. The dynamic spatial bases are subsequently compressed into
one global spatial basis, allowing the final operator to use one global NFFT
plan instead of one NFFT per dynamic and low-rank term.

# Arguments

* `grid::Grid{T}`                      - Cartesian reconstruction grid;
  `grid.matrixSize` defines [nX, nY, nZ].
* `kspha::AbstractArray{T, 3}`         - [nTerm, nSam, nDyn], coefficients of
  field dynamics. `nTerm` must be `9` (up to second order) or `16` (up to
  third order).
* `times::AbstractArray{T, 2}`         - [nSam, nDyn], sampling time points.

# Keywords

* `fieldmap::AbstractArray{T}`          - [nX, nY, nZ], off-resonance map.
  Its units must be consistent with `times` so that their product gives phase
  in cycles. [nX, nY] is accepted when `nZ == 1`.
* `csm::AbstractArray{Complex{T}}`      - [nX, nY, nZ, nCha], complex coil
  sensitivity maps. [nX, nY, nCha] is accepted when `nZ == 1`.
* `mask::AbstractArray{Bool}`           - [nX, nY, nZ], reconstruction mask.
  [nX, nY] is accepted when `nZ == 1`; only masked voxels enter the low-rank
  approximation.
* `recon_terms::Union{Nothing, AbstractString}` - Binary order-selection
  string. Use three digits for `nTerm == 9` and four digits for
  `nTerm == 16`; the digits select zeroth-, first-, second-, and third-order
  terms. A `0` removes that order, except that disabling first-order error
  replaces it with `k_nominal`. The default is `"111"` or `"1111"`.
* `k_nominal::AbstractArray{T, 3}`      - [3, nSam, nDyn], nominal first-order
  trajectory ordered as [kx, ky, kz]. It is used when first-order error is
  disabled by `recon_terms`. Its units are cycles per physical-length unit
  used by `grid.x`, `grid.y`, and `grid.z`.
* `arrayType::Type{<:AbstractArray}`    - Storage and execution backend; use
  `Array` for CPU execution or `CuArray` for CUDA execution.
* `gpus::Vector{Int}`                   - Zero-based CUDA device IDs. The first
  entry is the primary GPU that owns the returned operator. Additional GPUs
  can participate in distributed rSVD setup and normal-operator evaluation.
* `L_rank::Int`                         - rSVD truncation rank used independently
  for each dynamic; this is not the final shared rank printed during setup.
* `rsvd_seed::Int`                      - Base random seed. Dynamic `d` uses
  `rsvd_seed + d - 1`, making repeated setups reproducible for a fixed
  configuration.
* `rsvd_chunk::Int`                     - Voxel chunk size for the `:chunked`
  rSVD backend. It limits temporary memory without changing the mathematical
  approximation.
* `rsvd_oversample::Int`                - rSVD oversampling parameter. The
  sketch width is `L_rank + rsvd_oversample` and must not exceed
  `min(nSam, nVox)`.
* `rsvd_finalize::Symbol`               - Finalization algorithm. `:svd`
  performs the conventional tall-skinny SVD; `:gram` diagonalizes a small
  Gram matrix and lowers peak memory for large 3D data.
* `rsvd_backend::Symbol`                - rSVD implementation: `:chunked`,
  `:kernel`, or `:auto`. `:auto` selects `:chunked` for `Array` and the fused
  CUDA `:kernel` backend for `CuArray`.
* `rsvd_distribution::Symbol`           - rSVD setup distribution: `:single`,
  `:voxel`, or `:auto`. `:voxel` partitions masked voxels across `gpus` and
  requires `arrayType=CuArray`, `rsvd_backend=:kernel`, and
  `rsvd_finalize=:gram`. `:auto` selects this mode when possible and more than
  one GPU is supplied.
* `shared_rank_max::Int`                - Upper bound on the global spatial
  basis rank, clamped to `min(nVox, L_rank * nDyn)`.
* `shared_basis_tol::T`                 - Relative approximation-error
  tolerance used while merging per-dynamic spatial bases. The final shared
  rank is adaptive and can be larger than `L_rank`.
* `normal_distribution::Symbol`         - Normal-operator distribution used by
  CG. `:single` uses the primary GPU; `:channel` partitions coil channels
  across `gpus` and lazily constructs the multi-GPU backend when
  `normalOperator(W ∘ op)` is requested.
* `nfft_center_correction::Bool`        - Apply the parity-aware NFFT centre
  correction that aligns AbstractNFFTs' integer-centred grid with `Grid`'s
  physical voxel centres. Even dimensions receive a half-voxel correction;
  odd dimensions receive none. Keep it enabled to match
  `HighOrderOp_Kernel`; disable it only for the legacy NFFT convention.
* `verbose::Bool`                       - Print rSVD configuration,
  shared-basis progress, timing, and resource-release information.

# Returns

A `HighOrderLowRankOp{Complex{T}}` with size
`(nSam * nDyn * nCha, prod(grid.matrixSize))`. Its `q` field has size
`(nSam * nDyn, shared_rank)` and its masked spatial `basis` has size
`(nVox, shared_rank)`. The output ordering is compatible with
`vec(data)`, where `data` has size `(nSam, nDyn, nCha)`.

# Notes

The forward phase convention is `exp(+2πim * phase)`, matching
`HighOrderOp_Kernel`. Internally, first-order physical coefficients are
converted to AbstractNFFTs nodes as `-kᵢ * Δᵢ`, independently for each active
dimension. The operator normalization is `1 / sqrt(nVox)` and is folded into
`q`. The output ordering is samples first, then dynamics, then channels.

Voxel-distributed rSVD accelerates operator setup; channel-distributed normal
evaluation accelerates CG iterations. They are independent and are controlled
by `rsvd_distribution` and `normal_distribution`, respectively. Start Julia
with at least one default thread per participating GPU for effective
concurrency. `recon_HOOp` releases its channel-distributed normal backend by
default. After direct use of `normalOperator`, or when calling
`recon_HOOp(...; release_backend=false)`, call `close(op)` when the backend is
no longer needed. This stops its worker tasks and returns explicit workspaces
to CUDA.jl's reusable memory pool; the reserved-memory value reported by
`nvidia-smi` does not necessarily decrease immediately.

Before replacing a large operator in the same variable, release the normal
backend, set the old variable to `nothing`, and run `GC.gc()` before
constructing the replacement. In an assignment such as
`op = HighOrderLowRankOp(...)`, Julia keeps the old value of `op` alive until
the new constructor finishes, which can otherwise make the old and new NFFT
plans and spatial bases coexist temporarily.
"""
function HighOrderLowRankOp(
    grid              :: Grid{T}                                                             ,
    kspha             :: AbstractArray{T, 3}                                                 , 
    times             :: AbstractArray{T, 2}                                                 ;
    fieldmap          :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm               :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask              :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    recon_terms       :: Union{Nothing,AbstractString} = nothing                            ,
    k_nominal         :: AbstractArray{T, 3}       = kspha[2:4, :, :]                        ,
    
    arrayType         :: Type{<:AbstractArray}     = Array                                   ,
    gpus              :: Vector{Int}               = [0]                                     ,
    L_rank            :: Int                       = 15                                      , 
    rsvd_seed         :: Int                       = 1234                                    ,
    rsvd_chunk        :: Int                       = 4096                                    , 
    rsvd_oversample   :: Int                       = 5                                       ,
    rsvd_finalize     :: Symbol                    = :svd                                    ,
    rsvd_backend      :: Symbol                    = :auto                                   ,
    rsvd_distribution :: Symbol                    = :auto                                   ,
    shared_rank_max   :: Int                       = 128                                     ,
    shared_basis_tol  :: T                         = T(1e-2)                                 , 
    normal_distribution:: Symbol                   = :single                                 ,
    nfft_center_correction :: Bool                 = true                                    ,
    verbose           :: Bool                      = false                                   ,   
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
    rsvd_backend in (:auto, :chunked, :kernel) || throw(ArgumentError("Unsupported rsvd_backend=$rsvd_backend"))
    rsvd_distribution in (:auto, :single, :voxel) || throw(ArgumentError("Unsupported rsvd_distribution=$rsvd_distribution; " * "expected :auto, :single, or :voxel"))
    normal_distribution in (:single, :channel) || throw(ArgumentError("Unsupported normal_distribution=$normal_distribution; expected :single or :channel"))
    isempty(gpus) && throw(ArgumentError("gpus must contain at least one GPU id"))
    length(unique(gpus)) == length(gpus) || throw(ArgumentError("gpus contains duplicate GPU ids: $gpus"))

    is_gpu = arrayType == CuArray
    primary_gpu = first(gpus)
    if normal_distribution === :channel
        is_gpu || throw(ArgumentError("normal_distribution=:channel requires arrayType=CuArray"))
        length(gpus) >= 2 || throw(ArgumentError("normal_distribution=:channel requires at least two GPUs"))
    end
    rsvd_backend = rsvd_backend === :auto ? (is_gpu ? :kernel : :chunked) : rsvd_backend
    use_distributed_rsvd =
    rsvd_distribution === :voxel || (rsvd_distribution === :auto && is_gpu && length(gpus) > 1 && rsvd_finalize === :gram && rsvd_backend === :kernel)

    if use_distributed_rsvd
        is_gpu || throw(ArgumentError("rsvd_distribution=:voxel requires arrayType=CuArray"))
        rsvd_finalize === :gram || throw(ArgumentError("Multi-GPU voxel-distributed rSVD currently requires " * "rsvd_finalize=:gram"))
        rsvd_backend === :kernel || throw(ArgumentError("Multi-GPU voxel-distributed rSVD currently requires " * "rsvd_backend=:kernel"))
    end
    if is_gpu CUDA.device!(primary_gpu) end
    
    if verbose
        @info(
            "rSVD execution configuration",
            backend=rsvd_backend,
            distribution=use_distributed_rsvd ? :voxel : :single,
            gpus=gpus,
            primary_gpu=is_gpu ? primary_gpu : nothing,
            normal_distribution,
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


    @info "nSam=$nSam, nVox=$nVox, nDyn=$nDyn, nTerm=$nTerm, nCha=$nCha"

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
                rsvd_timing = DistributedRSVDTiming()
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
                    v_scaled=v_scaled, verbose=verbose)
            else
                u_trunc, s_trunc, v_trunc = perform_rsvd(
                    times_dyn, fieldmap_device, bf_err_device, kspha_err_dyn,
                    nVox, nSam, L_rank, rsvd_chunk, rsvd_workspace;
                    seed=rsvd_seed + dyn - 1, p_oversample=rsvd_oversample, 
                    rsvd_finalize=:svd, rsvd_backend=rsvd_backend,
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
        k_range = 2:3
        voxel_spacing = T[grid.Δx, grid.Δy]
    else
        MatrixSize = (nX, nY, nZ)
        k_range = 2:4
        voxel_spacing = T[grid.Δx, grid.Δy, grid.Δz]
    end

    # AbstractNFFTs evaluates the forward transform with a negative Fourier
    # exponent on integer grid indices.  The explicit high-order operator uses
    # exp(+i2π k⋅r), so the physical first-order coefficients must be negated
    # and normalized independently by each voxel spacing.
    nfft_traj = -Array{T,3}(kspha[k_range, :, :]) .*
                reshape(voxel_spacing, :, 1, 1)
    nfft_nodes = reshape(nfft_traj, length(k_range), nSam * nDyn) # dyn1 [all samples]、dyn2 [all samples] ……
    nfftplan = plan_nfft(arrayType, nfft_nodes, MatrixSize; m=3, σ=1.25)
    @assert eltype(nfftplan) == Complex{T} "NFFT plan precision must match the operator precision"
    if verbose @info("Global NFFT plan ready", trajectory_eltype=eltype(nfft_nodes), plan_eltype=eltype(nfftplan), nPoint=nSam * nDyn, nfft_per_forward=shared_rank * nCha, previous_nfft_per_forward=nDyn * L_rank * nCha) end

    q_scale = inv(sqrt(T(nVox)))
    # `kspha` has already been processed by `prep_kspha`. Therefore a leading
    # zero in `recon_terms` makes this phase identically zero and disables the
    # zeroth-order correction without requiring a separate branch here.
    zeroth_phase = vec(Array{T,3}(kspha[1:1, :, :]))
    zeroth_correction = arrayType(
        exp.(Complex{T}(0, T(2π)) .* zeroth_phase),
    )
    if nfft_center_correction
        # For an even dimension, AbstractNFFTs' integer-centred indices and
        # Grid's physical voxel centres differ by +1/2 voxel.  Odd dimensions
        # already share the same centre and must not receive this correction.
        # Fold the zeroth-order temporal phase, parity-aware centre phase, and
        # 1/sqrt(nVox) normalization into q once; ctprod! then uses their
        # conjugates.
        center_offsets = T[iseven(n) ? -0.5 : 0.0 for n in MatrixSize]
        center_phase = vec(sum(
            nfft_traj .* reshape(center_offsets, :, 1, 1);
            dims=1,
        ))
        center_correction = arrayType(exp.(Complex{T}(0, T(2π)) .* center_phase))
        q .*= q_scale .* reshape(
            zeroth_correction .* center_correction,
            :,
            1,
        )
    else
        q .*= q_scale .* reshape(zeroth_correction, :, 1)
    end

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
        copy(gpus),
        verbose,
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
