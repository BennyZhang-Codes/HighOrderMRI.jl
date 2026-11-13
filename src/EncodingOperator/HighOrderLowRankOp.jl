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

The constructor expression is evaluated after the old binding is cleared.
Other references may still keep the old operator or its arrays alive. If the
new construction fails, the previous binding is not restored.

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
Configuration and owned state for a low-rank reconstruction backend.

Channel- and readout-distributed backends initialize on the first
`normalOperator` request. A rank-distributed backend also initializes on the
first forward or adjoint request. Before initialization, this object holds
configuration rather than worker NFFT plans and workspaces.

The parent `HighOrderLowRankOp` owns `operator` and its persistent GPU workers.
Shared and streamed per-readout adjoints can reuse channel/readout workers
after initialization. `release_highorder_normal_backend!(op)` releases the
owned backend and clears its cached weights; `close(op)` also closes a parent
per-readout cache when present. Closing a derived normal view does not release
these shared resources.
"""
mutable struct HighOrderNormalBackend
    distribution :: Symbol
    gpus          :: Vector{Int}
    verbose       :: Bool
    operator      :: Union{Nothing,AbstractLinearOperator}
    weights2      :: Union{Nothing,AbstractVector}
    sample_batch_size :: Int
    primary_gpu   :: Int
end

HighOrderNormalBackend(distribution, gpus, verbose, operator, weights2, sample_batch_size) =
    HighOrderNormalBackend(distribution, gpus, verbose, operator, weights2, sample_batch_size,
        isempty(gpus) ? 0 : first(gpus))

HighOrderNormalBackend(distribution, gpus, verbose, operator, weights2) =
    HighOrderNormalBackend(distribution, gpus, verbose, operator, weights2, 12_000_000)

"""
    HighOrderLowRankOp

Represent high-order MRI encoding with NFFT Fourier transforms and low-rank
residual-phase factors. Its element type is `Complex{T}`, with real precision
`T` as the first type parameter.

The zeroth-order coefficient supplies a temporal phase. In 3D, NFFT handles
`kx*x + ky*y + kz*z`; in 2D it handles only `kx*x + ky*y`, leaving `kz*z` in
the residual. Static off-resonance, selected higher-order terms, and optional
concomitant terms also enter the residual. The forward phase uses
`exp(+2π * im * phase)` and normalization `1/sqrt(nVox)`, where `nVox` is
the number of masked voxels.

Shared low-rank (`shared_basis_method=:shared_low_rank`, legacy alias
`:rsvd`) builds local factors for each readout `i`. The default remains
`:rsvd`. `basis_layout=:shared` incrementally merges their spatial factors;
`:per_readout` retains independent factors from the same rSVD front end.
Local truncation, incremental sharing, and optional final recompression introduce distinct approximation
errors. Incremental sharing is not a global optimal SVD of all readouts.

Construct the operator with either the single-readout or multi-readout
constructor below. The object owns mutable workspaces and optional GPU
workers; it and its derived normal views must not be used concurrently.
Use `close(op)` to release owned distributed resources and streaming caches.
See [Low-rank theory](@ref LowRankTheory) for the factorization and
[Resource ownership](@ref Resources) for reuse and cleanup.

Joint adaptive low-rank (`shared_basis_method=:joint_adaptive_low_rank`,
legacy alias `:adaptive`) selects the shared rank against global H error.
The older `:joint` option retains its separate per-readout validation.
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
    
    basis_layout :: Symbol               # :shared or :per_readout
    per_readout_storage :: Symbol        # :gpu, :cpu, or :regenerate
    per_readout_cache_size :: Int
    readout_cache :: Union{Nothing,PerReadoutCache{T,P}}
    q          :: AQ                     # temporal factors; rows grouped by readout
    basis      :: AB                     # shared columns or consecutive per-readout VΣ blocks
    csm        :: AC                     # [nVox, nCha]
    
    nfftplan   :: Union{Nothing,P}       # rank-distributed plans live on the workers
    readout_plans :: Vector{P}           # per-readout plans; empty for :shared
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
        grid :: Grid{T} ,
        kspha :: AbstractArray{T, 2} ,
        times :: AbstractArray{T, 1} ;
        kwargs...
    ) where T <: AbstractFloat

Construct the operator for one readout by adding a singleton readout dimension
and forwarding to the multi-readout constructor.

`kspha` has shape `(nTerm, nSam)` with 9 or 16 solid-harmonic coefficients,
and `times` has shape `(nSam,)`. Their real element type matches `Grid{T}`.
Use seconds for `times` when `fieldmap` is in Hz. Keyword options, coordinate
conventions, and resource ownership are those of the multi-readout constructor.

The default nominal trajectory is obtained after reshaping `kspha`. An
explicit `k_nominal` keyword must still have shape `(3, nSam, 1)`, because
keyword arguments are forwarded unchanged. `concomitant` accepts either
`(4, nSam)` or `(4, nSam, 1)` for this overload.

The result has dimensions `(nSam*nCha, prod(grid.matrixSize))` and element
type `Complex{T}`. The image vector covers the full grid; the forward ignores
masked-out voxels and the adjoint fills those voxels with zero.
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
        grid :: Grid{T} ,
        kspha :: AbstractArray{T, 3} ,
        times :: AbstractArray{T, 2} ;
        fieldmap :: AbstractArray{T} = zeros(T, grid.matrixSize...) ,
        csm :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) ,
        mask :: AbstractArray{Bool} = trues(grid.matrixSize...) ,
        concomitant = nothing ,
        recon_terms :: Union{Nothing,AbstractString} = nothing ,
        k_nominal :: AbstractArray{T, 3} = kspha[2:4, :, :] ,
        arrayType :: Type{<:AbstractArray} = Array ,
        gpus :: Vector{Int} = [0] ,
        L_rank :: Int = 15 ,
        rsvd_seed :: Int = 1234 ,
        rsvd_chunk :: Int = 4096 ,
        rsvd_oversample :: Int = 5 ,
        rsvd_finalize :: Symbol = :svd ,
        rsvd_backend :: Symbol = :auto ,
        rsvd_distribution :: Symbol = :auto ,
        basis_layout :: Symbol = :shared ,
        per_readout_storage:: Symbol = :gpu ,
        per_readout_cache_size:: Int = 5 ,
        per_readout_gather:: Symbol = :pipeline ,
        per_readout_host_cache_bytes:: Int = 0 ,
        shared_rank_max :: Int = 128 ,
        shared_basis_tol :: T = T(1e-2) ,
        shared_noadd_check::Bool = false ,
        global_basis_tol :: Union{Nothing,T} = nothing ,
        shared_basis_method:: Symbol = :rsvd ,
        epsilon_H :: Union{Nothing,Real} = nothing ,
        adaptive_options :: NamedTuple = (;) ,
        joint_snapshots :: Int = 256 ,
        joint_samples :: Int = 512 ,
        joint_roi_tol :: Union{Nothing,T} = nothing ,
        joint_basis_report:: Union{Nothing,Ref} = nothing ,
        normal_distribution:: Symbol = :single ,
        normal_sample_batch_size:: Int = 12_000_000 ,
        nfft_center_correction :: Bool = true ,
        verbose :: Bool = false ,
    ) where T <: AbstractFloat

Construct a low-rank encoding operator for `nDyn` readouts. The defaults use
CPU arrays, local rSVD, a shared spatial basis, and single-device execution.
The keyword `normal_distribution` selects reconstruction execution;
`rsvd_distribution` independently selects how the factors are built.

`shared_basis_method=:shared_low_rank` names the default Shared low-rank
front end; `:rsvd` remains its compatible default spelling.

`shared_basis_method=:joint_adaptive_low_rank, epsilon_H=0.01` opts into
experimental Joint adaptive low-rank. The legacy alias `:adaptive` remains
supported. It builds a stable snapshot master, fits all actual
time rows, then uses blocked QR/small SVD and paired residual checks to select
the final shared rank. K, J and the parent cap start at 128, 512 and 32;
only a failed check triggers bounded expansion. `shared_rank_max` remains
the hard parent cap. `adaptive_options` forwards resource and evaluation
keywords to `joint_adaptive_spatial_basis`; `rsvd_seed` and `rsvd_chunk`
provide its seed and chunk size. No NFFT setting changes.

The target is the global relative Frobenius error of the pure residual H
over all readouts and masked voxels. It excludes k0, first-order NFFT,
center phase, CSM, weights and normalization. The checks are empirical:
`sqrt(mean(loss)+4SE)` is a heuristic, not a probability certificate or an
image-error guarantee. `joint_basis_report` receives construction, empirical,
external-audit and certification states without retaining factor arrays.
Budget failure fills that report and throws instead of constructing an
operator from partial factors. Existing `shared_basis_tol` and
`global_basis_tol` retain their old meanings; Joint adaptive low-rank requires
`global_basis_tol=nothing` and does not impose `joint_roi_tol`.

The older `:joint` method and its `joint_*` sampling controls remain available
with their existing per-readout validation semantics.

# Inputs and coordinate convention

- `grid`: reconstruction grid with shape `(nX,nY,nZ)`. NFFT assumes the
  centered regular coordinates implied by its dimensions and voxel spacings
  on the active Fourier axes. Arbitrarily transforming those coordinates
  is not accounted for by `nfft_center_correction`.
- `kspha`: `(nTerm,nSam,nDyn)` time-integrated solid-harmonic coefficients,
  with `nTerm=9` or `16`. Readout index `i` selects `kspha[:,:,i]`. The basis
  ordering and normalization must match `basisfunc_spha`.
- `times`: `(nSam,nDyn)` sample times. Use seconds with a field map in Hz;
  each coefficient times its spatial basis must likewise give phase in cycles.
- `fieldmap`: real off-resonance map of shape `(nX,nY,nZ)`; the default is
  zero. `csm` contains complex coil maps of shape `(nX,nY,nZ,nCha)` and
  defaults to one coil with unit sensitivity. `mask` is a Boolean grid mask,
  all true by default. For `nZ==1`, shapes `(nX,nY)` for `fieldmap`/`mask`
  and `(nX,nY,nCha)` for `csm` are accepted. Use consistent real precision
  `T` and complex precision `Complex{T}`.
- `concomitant=nothing`: optional `(4,nSam,nDyn)` integrated coefficients
  for `z²`, `x²+y²`, `xz`, and `yz`. `(4,nSam)` is also accepted when
  `nDyn==1`. With coordinates in metres, coefficients are in cycles/m² in
  the same frame and origin. These terms enter the residual independently
  of `recon_terms`; they are already integrated and are not multiplied by
  `times` again.
- `recon_terms=nothing`: resolve to `"111"` for 9 coefficients or `"1111"`
  for 16. The binary digits select zeroth-, first-, second-, and third-order
  terms. A zero removes the selected order except at first order, where it
  substitutes `k_nominal`. This substitution preserves nominal Fourier
  encoding rather than removing it. `k_nominal` has shape `(3,nSam,nDyn)`,
  ordered `kx,ky,kz`; its default is the supplied first-order coefficients.
  First-order coefficients use cycles per coordinate-length unit.

In 2D, NFFT uses only `kx,ky`; any `kz*z` phase stays in the low-rank
residual. In 3D all three linear terms enter NFFT. The nodes on each active
axis are `-k*Δ`, with that axis's voxel spacing `Δ`. Keep
`nfft_center_correction=true` to include the half-voxel phase on even-sized
axes; odd-sized axes need no such phase. Zeroth-order phase, this correction,
and `1/sqrt(nVox)` are folded into `q` once.

# Local rSVD and shared approximation

- `L_rank=15` is the local truncation rank for every readout, not the final
  shared rank. `rsvd_oversample=5` sets the extra sketch width. Require
  `L_rank>0`, `rsvd_oversample>=0`, and
  `L_rank+rsvd_oversample <= min(nSam,nVox)`, where `nVox=sum(mask)`.
- `rsvd_seed=1234`: readout `i` uses seed `rsvd_seed+i-1`. Repeatability
  requires the same backend and setup configuration; changing device count
  or numerical libraries is not a bitwise-reproducibility guarantee.
- `rsvd_backend=:auto` chooses `:chunked` for `Array` and `:kernel` for
  `CuArray`. Explicit `:kernel` requires CUDA. `rsvd_chunk=4096` must be
  positive and is capped at `nVox`; it controls voxel chunks for the chunked
  phase products. It is not a limit on all setup arrays or factor storage.
- `rsvd_finalize=:svd` takes an SVD of the projected adjoint matrix.
  `:gram` uses a small Gram eigendecomposition to recover the retained
  factors. Both absorb the singular values into the spatial factors;
  finite-precision Gram calculations can lose accuracy in weak directions.
- `rsvd_distribution=:single` builds on the selected CPU or primary GPU.
  Explicit `:voxel` requires `CuArray`, `rsvd_finalize=:gram`, and backend
  `:kernel` or `:chunked`. One GPU is accepted; multiple GPUs divide masked
  voxels. The current partitioning requires at least one voxel per selected
  setup GPU. `:auto` chooses voxel distribution only with CUDA, multiple
  GPUs, kernel products, and Gram finalization. Chunked voxel setup requires
  explicit `:voxel`.
- `basis_layout=:shared` incrementally merges the local spatial factors.
  `shared_rank_max=128` is a positive cap, clamped to
  `min(nVox,L_rank*nDyn)` for this path; exceeding it throws rather than
  silently relaxing the tolerance. `shared_basis_tol=T(1e-2)` controls the
  additional relative Frobenius residual of each local spatial factor
  `V_i*Σ_i` under the shared projection. It is not an error target for the
  original phase matrix or reconstructed image. Use a finite nonnegative
  tolerance below one. Earlier readout coefficients receive zero rows when
  the basis grows; they are not recomputed by a final global projection.
- `global_basis_tol=nothing` preserves the incrementally shared factors.
  A nonnegative value of type `T` applies a final spectral recompression of
  that representation, introducing an additional error. This requires
  shared layout. Use a tolerance below one: the implementation accepts
  larger values, but its discarded-energy diagnostic is unreliable at
  `global_basis_tol>=1` because it retains at least one direction.

Local truncation, shared projection, final recompression, and NFFT
approximation are separate sources of error. A fixed `L_rank` does not
certify a total phase-matrix or image error. See
[Low-rank theory](@ref LowRankTheory) for the factor definitions and references.

# Optional no-add check

`shared_noadd_check=false` by default. Enabling it requires shared,
voxel-distributed rSVD. Once a shared basis exists, the update keeps its two
projection passes and forms the same explicit residual. It accumulates the
residual energy in FP64 and applies a guard near the tolerance threshold.
When that check accepts the existing basis, it skips the residual Gram matrix
and eigendecomposition; otherwise it follows the usual update.

This option changes basis construction only. It does not select a CG
execution strategy, reduce `L_rank`, change the tolerance, or remove the
residual workspace. Its early-return diagnostic uses the full residual norm,
which can differ from the ordinary diagnostic after tiny residual eigenvalues
are discarded. That diagnostic difference is not fed back into the factors.
The FP64 reduction does not make the preceding projection arithmetic FP64.

# Independent factors and storage

`basis_layout=:per_readout` keeps each readout's local factors independently.
The shared-rank cap and shared tolerance do not control those factors;
`global_basis_tol` and `shared_noadd_check=true` are unsupported here.

- `per_readout_storage=:gpu` materializes the spatial bank on `arrayType`.
  Despite the name, it uses CPU arrays with `arrayType=Array`.
- `:cpu` requires `CuArray` execution and keeps the complete spatial bank in
  host RAM, transferring readout blocks into a bounded GPU cache.
- `:regenerate` also requires CUDA and voxel-distributed kernel/Gram setup.
  It retains the original rSVD `Q/Z` and model maps, then generates uncached
  spatial blocks as `(H_iᴴ*Q_i)*Z_i`. It avoids materializing the full spatial
  bank, but retains metadata and regeneration workspaces and adds arithmetic.
  Regenerated factors are not promised to be bitwise identical to materialized
  factors.
- `per_readout_cache_size=5` is the positive number of resident readout slots
  per GPU, capped by the number of readouts. The streaming cache uses one
  pinned host staging buffer and completion events to protect reuse. With
  regeneration, one slot serializes production/use; multiple slots permit
  lookahead. These caches are used by forward, adjoint, and normal execution.
- `per_readout_gather=:pipeline` overlaps voxel-distributed setup downloads
  and host writes using two pinned buffers per GPU. `:synchronous` uses a
  blocking gather. This choice does not change single-device setup.
- `per_readout_host_cache_bytes=0` is a nonnegative host budget for complete
  spatial blocks in regeneration mode. A fixed, evenly spaced subset of
  readouts is materialized during setup, shared by worker caches, and reused.
  Q/Z and model maps are additional storage outside this budget.

# Execution, communication and sample batching

Use `arrayType=Array` for CPU execution or `CuArray` for CUDA. `gpus=[0]`
contains unique zero-based CUDA-visible IDs; it must be nonempty.
`first(gpus)` is the primary GPU for solver vectors. The public constructor
has no separate `primary_gpu` keyword. Request at least two IDs for
`:channel`, `:readout`, or `:rank`, all of which require CUDA:

- `normal_distribution=:single` is the default CPU or primary-GPU path.
- `:channel` divides coils, retaining complete shared factors on each active
  worker. Independent factors can instead use the selected per-readout
  storage scheme. This mode does not partition shared-rank columns.
- `:readout` divides independent readouts and requires per-readout layout;
  each worker retains all coil maps. Workers sum contributions from disjoint
  observation rows.
- `:rank` divides shared temporal/spatial columns and coil-map storage while
  retaining complete host master factors. It requires shared layout and
  either voxel-distributed ordinary rSVD with Gram finalization or adaptive
  global-H setup. At most
  `min(length(gpus),shared_rank,nCha)` workers are used. For each coil and
  sample batch, partial forward signals are copied to host buffers, summed,
  and broadcast to the workers before weighting and the adjoint. This retains
  the cross-rank terms of `Aᴴ Wᴴ W A`. Image contributions are also summed
  through host buffers. Channel/readout normal products likewise stage
  image broadcasts and reductions through the host.

Channel/readout workers initialize on the first normal request. Selected
adjoint paths reuse them afterward; a standalone forward still uses the
parent's single-device path. Rank workers initialize on the first forward,
adjoint, or normal request. Allow at least one Julia default thread per active
GPU worker, and a coordinator thread where possible.

`normal_sample_batch_size=12_000_000` must be positive in every mode, but only
rank execution uses it. Its effective size is
`min(nSam*nDyn,normal_sample_batch_size)`, in sample/readout points per coil.
It bounds sample workspaces, not full temporal-factor columns, spatial
factors, coil maps, image/NFFT grids, solver vectors, complete input/output
data, or setup allocations. Smaller batches repeat transforms and transfers.

With the default NFFT.jl 0.14.3 GPU plans, use a full batch or a divisor of
`nSam*nDyn`: the current node-update path does not update all output-shape
metadata for a shorter final batch. The optional NonuniformFFTs backend
supports variable-length nodes; select the backend before construction and
keep it fixed while the operator is in use. See
[Multi-GPU execution](@ref MultiGPU) for configuration and memory tradeoffs.

# Result, weighting and resources

The result is a `HighOrderLowRankOp` with element type `Complex{T}` and shape
`(nSam*nDyn*nCha, prod(grid.matrixSize))`. Forward data order matches
`vec(data)` for data shaped `(nSam,nDyn,nCha)`: samples vary fastest, then
readouts, then coils. Image vectors cover the full grid; the forward ignores
masked-out voxels and the adjoint sets them to zero.

For shared layout, `q` has shape `(nSam*nDyn,shared_rank)` and `basis` has
shape `(nVox,shared_rank)`. For per-readout layout, `q` has shape
`(nSam*nDyn,L_rank)` and the logical spatial bank has shape
`(nVox,L_rank*nDyn)`. Regeneration stores that bank implicitly.

In rank mode, `q`, `basis`, and `csm` are host matrices, while solver storage
is CUDA storage on the primary GPU. The allocating expression `op*x` follows
`q` and therefore returns a host vector. `mul!` accepts a preallocated GPU
output. For CUDA `recon_HOOp`, explicitly place data and raw weights on the
primary GPU; that wrapper does not move them automatically.

`recon_HOOp` receives raw weights `w` and constructs `Aᴴ diag(abs2(w)) A`.
A direct optimized `normalOperator(op,D)` instead expects real, nonnegative
weights already representing `WᴴW`, identical for each coil. For `:channel`,
`:readout`, or `:rank`, release the cached distributed backend before changing
these normal weights and create a fresh view. The `:single` path has no such
owned distributed backend. Positive semidefiniteness of the data normal alone
does not establish the
positive-definite conditions needed for the usual CG convergence guarantee.

The parent owns distributed workers, plans, and workspaces. By default,
`recon_HOOp` releases that backend in `finally`; it retains the parent's
factors and source cache. For direct use, or `release_backend=false`, call
`close(op)` in a `finally` block. `close(op)` also closes the parent's
streaming/regeneration cache; closing a derived normal view alone does not.
Factor arrays and any parent-resident plan remain referenced by the parent,
so closing is not equivalent to dropping the entire operator. Later use can
rebuild owned resources; obtain fresh normal views after release.

The operator and its views share mutable workspaces and are not reentrant.
When replacing a large operator, close it and drop old references before
constructing the replacement; `@rebuild_HOOp` provides this sequence for a
plain variable. See [Resource ownership](@ref Resources) for details.

`verbose=false` suppresses optional setup progress and diagnostic timing;
some constructor information is still logged unconditionally. Execution cost
and memory feasibility depend on the data, rank, backend and hardware.
"""
function HighOrderLowRankOp(
    grid              :: Grid{T}                                                             ,
    kspha             :: AbstractArray{T, 3}                                                 , 
    times             :: AbstractArray{T, 2}                                                 ;
    fieldmap          :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm               :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask              :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    concomitant                                    = nothing                                 ,
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
    basis_layout      :: Symbol                    = :shared                                 ,
    per_readout_storage:: Symbol                   = :gpu                                    ,
    per_readout_cache_size:: Int                   = 5                                       ,
    per_readout_gather:: Symbol                    = :pipeline                               ,
    per_readout_host_cache_bytes:: Int             = 0                                       ,
    shared_rank_max   :: Int                       = 128                                     ,
    shared_basis_tol  :: T                         = T(1e-2)                                 , 
    shared_noadd_check::Bool                       = false                                   ,
    global_basis_tol  :: Union{Nothing,T}          = nothing                                 ,
    shared_basis_method:: Symbol                    = :rsvd                                   ,
    epsilon_H         :: Union{Nothing,Real}        = nothing                                 ,
    adaptive_options  :: NamedTuple                = (;)                                     ,
    joint_snapshots   :: Int                       = 256                                     ,
    joint_samples     :: Int                       = 512                                     ,
    joint_roi_tol     :: Union{Nothing,T}          = nothing                                 ,
    joint_basis_report:: Union{Nothing,Ref}        = nothing                                 ,
    normal_distribution:: Symbol                   = :single                                 ,
    normal_sample_batch_size:: Int                 = 12_000_000                              ,
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
    normal_distribution in (:single, :channel, :readout, :rank) || throw(ArgumentError("Unsupported normal_distribution=$normal_distribution; expected :single, :channel, :readout, or :rank"))
    normal_sample_batch_size > 0 || throw(ArgumentError("normal_sample_batch_size must be positive"))
    basis_layout in (:shared, :per_readout) || throw(ArgumentError("basis_layout must be :shared or :per_readout"))
    normal_distribution === :readout && basis_layout !== :per_readout &&
        throw(ArgumentError("normal_distribution=:readout requires basis_layout=:per_readout"))
    per_readout_storage in (:gpu, :cpu, :regenerate) || throw(ArgumentError("per_readout_storage must be :gpu, :cpu, or :regenerate"))
    per_readout_host_cache_bytes >= 0 || throw(ArgumentError("per_readout_host_cache_bytes must be non-negative"))
    per_readout_cache_size > 0 || throw(ArgumentError("per_readout_cache_size must be positive"))
    per_readout_gather in (:pipeline, :synchronous) || throw(ArgumentError("per_readout_gather must be :pipeline or :synchronous"))
    if per_readout_storage in (:cpu, :regenerate)
        basis_layout === :per_readout || throw(ArgumentError("CPU factor storage requires basis_layout=:per_readout"))
        arrayType == CuArray || throw(ArgumentError("CPU factor storage with a GPU cache requires arrayType=CuArray"))
    end
    shared_basis_method === :rsvd && (shared_basis_method = :shared_low_rank)
    shared_basis_method === :adaptive && (shared_basis_method = :joint_adaptive_low_rank)
    shared_basis_method in (:shared_low_rank, :joint, :joint_adaptive_low_rank) || throw(ArgumentError(
        "shared_basis_method must be :shared_low_rank, :joint or :joint_adaptive_low_rank; legacy aliases :rsvd and :adaptive remain supported"))
    use_joint_adaptive_basis = shared_basis_method === :joint_adaptive_low_rank
    if use_joint_adaptive_basis
        isnothing(joint_basis_report) || (joint_basis_report[] = (
            construction_status=:not_computed, empirical_status=:not_evaluated,
            certification_status=:not_requested, external_audit_status=:not_requested,
            error_metric=:global_relative_frobenius, failure_reason=:not_completed, rank=0))
        isnothing(epsilon_H) && throw(ArgumentError("Joint adaptive low-rank requires epsilon_H"))
        isfinite(epsilon_H) && 0 < epsilon_H < 1 || throw(ArgumentError("epsilon_H must lie in (0,1)"))
        isnothing(joint_roi_tol) || throw(ArgumentError("Joint adaptive low-rank uses global H error, not joint_roi_tol"))
        any(k -> k in (:epsilon_H, :arrayType, :seed, :chunk_size, :rank_max), keys(adaptive_options)) &&
            throw(ArgumentError("Use the constructor keywords for epsilon_H, arrayType, rsvd_seed, rsvd_chunk and shared_rank_max"))
    else
        isnothing(epsilon_H) && isempty(adaptive_options) || throw(ArgumentError("epsilon_H and adaptive_options require shared_basis_method=:joint_adaptive_low_rank (legacy alias :adaptive)"))
    end
    if basis_layout === :per_readout
        shared_basis_method === :shared_low_rank || throw(ArgumentError("basis_layout=:per_readout requires the rSVD front end, shared_basis_method=:shared_low_rank (legacy alias :rsvd)"))
        isnothing(global_basis_tol) || throw(ArgumentError("basis_layout=:per_readout does not support global_basis_tol"))
    end
    isempty(gpus) && throw(ArgumentError("gpus must contain at least one GPU id"))
    length(unique(gpus)) == length(gpus) || throw(ArgumentError("gpus contains duplicate GPU ids: $gpus"))

    is_gpu = arrayType == CuArray
    use_joint_basis = shared_basis_method in (:joint, :joint_adaptive_low_rank)
    if use_joint_basis
        isnothing(global_basis_tol) || throw(ArgumentError("shared_basis_method=$shared_basis_method requires global_basis_tol=nothing to preserve its validation"))
        rsvd_distribution !== :voxel || throw(ArgumentError("Joint basis setup uses the primary device; use rsvd_distribution=:auto or :single"))
        nCol <= typemax(Int32) || throw(ArgumentError("Joint operator grid indices exceed Int32"))
    end
    primary_gpu = first(gpus)
    if normal_distribution in (:channel, :readout, :rank)
        is_gpu || throw(ArgumentError("normal_distribution=$normal_distribution requires arrayType=CuArray"))
        length(gpus) >= 2 || throw(ArgumentError("normal_distribution=$normal_distribution requires at least two GPUs"))
    end
    rsvd_backend = rsvd_backend === :auto ? (is_gpu ? :kernel : :chunked) : rsvd_backend
    use_distributed_rsvd =
    !use_joint_basis && (rsvd_distribution === :voxel || (rsvd_distribution === :auto && is_gpu && length(gpus) > 1 && rsvd_finalize === :gram && rsvd_backend === :kernel))

    if normal_distribution === :rank
        basis_layout === :shared && (use_distributed_rsvd || use_joint_adaptive_basis) || throw(ArgumentError(
            "normal_distribution=:rank requires shared, voxel-distributed rSVD or Joint adaptive low-rank"))
    end

    if shared_noadd_check
        use_distributed_rsvd && basis_layout === :shared || throw(ArgumentError(
            "shared_noadd_check requires voxel-distributed shared rSVD"))
    end

    if use_distributed_rsvd
        is_gpu || throw(ArgumentError("rsvd_distribution=:voxel requires arrayType=CuArray"))
        rsvd_finalize === :gram || throw(ArgumentError("Multi-GPU voxel-distributed rSVD currently requires " * "rsvd_finalize=:gram"))
        rsvd_backend in (:kernel, :chunked) || throw(ArgumentError("Multi-GPU voxel-distributed rSVD currently requires " * "rsvd_backend=:kernel or :chunked"))
    end
    if per_readout_storage === :regenerate
        use_distributed_rsvd && rsvd_backend === :kernel || throw(ArgumentError(
            "per_readout_storage=:regenerate requires voxel-distributed kernel/Gram rSVD; use rsvd_distribution=:voxel, rsvd_backend=:kernel, rsvd_finalize=:gram"))
    end
    if is_gpu CUDA.device!(primary_gpu) end
    
    if verbose && use_joint_basis
        @info("Joint shared-basis execution configuration", primary_gpu=is_gpu ? primary_gpu : nothing,
            snapshots=joint_snapshots, samples=joint_samples, normal_distribution)
    elseif verbose
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
    # Split first-order NFFT terms and residual low-rank terms
    if nZ == 1 # kx and ky are handled by the 2D NFFT. kz*z must remain in the explicit residual phase.
        nfft_term_range = 2:3
        residual_term_range = 4:nTerm
        MatrixSize = (nX, nY)
        voxel_spacing = T[grid.Δx, grid.Δy]
    else # kx, ky and kz are handled by the 3D NFFT.
        nfft_term_range = 2:4
        residual_term_range = 5:nTerm
        MatrixSize = (nX, nY, nZ)
        voxel_spacing = T[grid.Δx, grid.Δy, grid.Δz]
    end


    mask_host = vec(mask)   # [prod(MatrixSize)]
    kspha = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    
    csm_host = Matrix{Complex{T}}(reshape(csm, :, nCha)[mask_host .!= 0, :])   # [nVox, nCha]
    fieldmap_host = Vector{T}(vec(fieldmap)[mask_host .!= 0])                  # [nVox]
    bf_host = Matrix{T}(basisfunc_spha(grid.x[mask_host .!= 0], grid.y[mask_host .!= 0], grid.z[mask_host .!= 0], collect(1:nTerm)))  # compute basis functions (spherical harmonics)
    times_host = Matrix{T}(times)
    mask_idx_host = Int32.(findall(mask_host .!= 0))
    
    if nTerm >= 5
        kspha_err_host = Array{T,3}(kspha[residual_term_range, :, :])
        bf_err_host = Matrix{T}(bf_host[:, residual_term_range])
    else
        kspha_err_host = zeros(T, 1, nSam, nDyn)
        bf_err_host = zeros(T, nVox, 1)
    end
    @assert size(kspha_err_host, 1) == size(bf_err_host, 2)
    bf_err_host, kspha_err_host = prep_concomitant(bf_err_host, kspha_err_host, grid, mask_host; concomitant)

    @info "nSam=$nSam, nVox=$nVox, nDyn=$nDyn, nTerm=$nTerm, nCha=$nCha"

    if basis_layout === :shared
        @assert shared_rank_max > 0 "shared_rank_max must be positive"
    end
    isnothing(global_basis_tol) || global_basis_tol >= zero(T) || throw(ArgumentError("global_basis_tol must be non-negative"))
    @assert rsvd_chunk > 0 "rsvd_chunk must be positive for chunked rSVD"
    @assert L_rank > 0 "L_rank must be positive"
    @assert rsvd_oversample >= 0 "rsvd_oversample must be non-negative"

    L_total = L_rank + rsvd_oversample

    if !use_joint_basis
        @assert L_total <= min(nSam, nVox) "L_rank + rsvd_oversample exceeds matrix dimensions"
    end

    rsvd_chunk = min(rsvd_chunk, nVox)

    max_possible_shared_rank = use_joint_basis ? min(nVox, nSam * nDyn) : min(nVox, L_rank * nDyn)
    effective_shared_rank_max = min(shared_rank_max, max_possible_shared_rank)
    if verbose && basis_layout === :shared && effective_shared_rank_max != shared_rank_max
        @info("Clamping shared_rank_max", requested=shared_rank_max, effective=effective_shared_rank_max, max_possible=max_possible_shared_rank)
    end


    if basis_layout === :per_readout
        q, basis = per_readout_basis(
            times_host, fieldmap_host, bf_err_host, kspha_err_host;
            arrayType, gpus, L_rank, rsvd_seed, rsvd_chunk, rsvd_oversample,
            rsvd_finalize, rsvd_backend, use_distributed_rsvd, verbose,
            host_storage=per_readout_storage === :cpu,
            gather_mode=per_readout_gather,
            regenerate=per_readout_storage === :regenerate,
            host_cache_bytes=per_readout_host_cache_bytes,
        )
        shared_rank = L_rank
        shared_errors = T[]
    elseif use_joint_adaptive_basis
        result = joint_adaptive_spatial_basis(times_host, fieldmap_host, bf_err_host, kspha_err_host;
            epsilon_H, arrayType, rank_max=effective_shared_rank_max,
            seed=rsvd_seed, chunk_size=rsvd_chunk, adaptive_options...)
        joint_report = (; (k => v for (k, v) in pairs(result) if k ∉ (:q, :S))...)
        isnothing(joint_basis_report) || (joint_basis_report[] = joint_report)
        result.construction_status === :computed || error("Adaptive basis budget exhausted: $(result.failure_reason)")
        q = normal_distribution === :rank ? result.q : arrayType(result.q)
        basis = normal_distribution === :rank ? result.S : arrayType(result.S)
        shared_rank = result.rank
        shared_errors = [sqrt(result.checks.a2[shared_rank])]
    elseif use_joint_basis
        q, basis, joint_report = joint_spatial_basis(
            times_host, fieldmap_host, bf_err_host, kspha_err_host;
            arrayType, rank_max=effective_shared_rank_max, tol=shared_basis_tol,
            snapshots=joint_snapshots, sample_count=joint_samples, chunk_size=rsvd_chunk,
            seed=rsvd_seed, roi_tol=joint_roi_tol, report_ref=joint_basis_report, verbose,
        )
        shared_rank = joint_report.rank
        shared_errors = joint_report.audit_errors
    elseif use_distributed_rsvd
        q_host, basis_host, shared_rank, shared_errors = let
            distributed_workspace = DistributedRSVDWorkspace(
                fieldmap_host,
                bf_err_host,
                nSam,
                L_total,
                L_rank,
                gpus;
                rsvd_backend,
                rsvd_chunk,
                kernel_tiling=true,
            )
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
                    relative_error, n_added = update_distributed_shared_basis!(distributed_shared, distributed_workspace, dyn, total_energy;
                        no_add_check=shared_noadd_check)
                    shared_time += (time_ns() - t0) * 1e-9

                    if verbose @info("Distributed shared spatial basis update", dyn=dyn, added=n_added, rank=distributed_shared.rank, relative_error=relative_error) end
                end
                profiled_rsvd_time = distributed_rsvd_total_time(rsvd_timing)
                timing_denominator = max(profiled_rsvd_time, eps(Float64))
                if verbose @info(
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
                ) end
                if verbose @info "Distributed setup timing" rsvd_time shared_time total=rsvd_time+shared_time end

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
        q = normal_distribution === :rank ? q_host : arrayType(q_host)
        basis = normal_distribution === :rank ? basis_host : arrayType(basis_host)
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

    # The incremental merge produces an orthonormal spatial basis up to roundoff.
    # Optionally rotate it using the eigensystem of qᴴq and discard weak global
    # directions.  This is an *additional* error relative to the already shared-
    # compressed representation, not an estimate of the total model error.
    if !isnothing(global_basis_tol)
        q, basis, global_compression = global_recompress_shared_basis(q, basis, global_basis_tol)
        shared_rank = global_compression.rank
        if verbose @info(
            "Global shared-basis recompression complete",
            pre_rank=global_compression.pre_rank,
            post_rank=global_compression.rank,
            additional_relative_error=global_compression.relative_error,
            basis_orthogonality_error=global_compression.basis_orthogonality_error,
            global_basis_tol,
        ) end
    end

    if is_gpu CUDA.device!(primary_gpu) end
    csm = normal_distribution === :rank ? csm_host : arrayType(csm_host)
    mask_idx = normal_distribution === :rank ? mask_idx_host : arrayType(mask_idx_host)
    
    if verbose && basis_layout === :shared
        @info("Shared spatial basis complete", rank=shared_rank, max_error=maximum(shared_errors), mean_error=sum(shared_errors) / length(shared_errors))
        @info("Global temporal basis ready", shared_rank=shared_rank, nPoint=nSam * nDyn, size=size(q))
    elseif verbose
        @info("Per-readout factors ready", rank_per_readout=L_rank, readouts=nDyn, temporal_size=size(q), spatial_size=size(basis), per_readout_storage, per_readout_cache_size, spatial_bytes=sizeof(basis))
        basis isa RegeneratedSpatialBasis && @info("Regenerated factor storage", stats=regenerated_basis_stats(basis))
    end


    # AbstractNFFTs evaluates the forward transform with a negative Fourier
    # exponent on integer grid indices.  The explicit high-order operator uses
    # exp(+i2π k⋅r), so the physical first-order coefficients must be negated
    # and normalized independently by each voxel spacing.
    nfft_traj = -Array{T,3}(kspha[nfft_term_range, :, :]) .* reshape(voxel_spacing, :, 1, 1)
    if normal_distribution === :rank
        fold_rank_shared_phases!(q, kspha, nfft_traj, MatrixSize, nVox, nfft_center_correction)
        return rank_shared_operator(q, basis, csm, mask_idx, nfft_traj, Tuple(MatrixSize), L_rank;
            gpus, sample_batch_size=normal_sample_batch_size, verbose)
    end
    nfft_nodes = reshape(nfft_traj, length(nfft_term_range), nSam * nDyn) # dyn1 [all samples]、dyn2 [all samples] ……
    readout_cache = nothing
    if basis_layout === :per_readout && per_readout_storage in (:cpu, :regenerate)
        nfftplan = plan_nfft(arrayType, nfft_traj[:, :, 1], MatrixSize; m=3, σ=1.25)
        readout_plans = typeof(nfftplan)[]
        readout_cache = PerReadoutCache(basis, nfft_traj, MatrixSize;
            capacity=per_readout_cache_size, first_plan=nfftplan)
    elseif basis_layout === :per_readout
        readout_plans = [plan_nfft(arrayType, nfft_traj[:, :, dyn], MatrixSize; m=3, σ=1.25) for dyn in 1:nDyn]
        nfftplan = first(readout_plans)
    else
        nfftplan = plan_nfft(arrayType, nfft_nodes, MatrixSize; m=3, σ=1.25)
        readout_plans = typeof(nfftplan)[]
    end
    @assert eltype(nfftplan) == Complex{T} "NFFT plan precision must match the operator precision"
    if verbose @info("NFFT plans ready", basis_layout, trajectory_eltype=eltype(nfft_nodes), plan_eltype=eltype(nfftplan), nPoint=nSam * nDyn, nfft_per_forward=(basis_layout === :shared ? shared_rank : nDyn * L_rank) * nCha) end

    q_scale = inv(sqrt(T(nVox)))
    # `kspha` has already been processed by `prep_kspha`. Therefore a leading
    # zero in `recon_terms` makes this phase identically zero and disables the
    # zeroth-order correction without requiring a separate branch here.
    zeroth_phase = vec(Array{T,3}(kspha[1:1, :, :]))
    zeroth_correction = arrayType(exp.(Complex{T}(0, T(2π)) .* zeroth_phase))
    if nfft_center_correction
        # For an even dimension, AbstractNFFTs' integer-centred indices and
        # Grid's physical voxel centres differ by +1/2 voxel.  Odd dimensions
        # already share the same centre and must not receive this correction.
        # Fold the zeroth-order temporal phase, parity-aware centre phase, and
        # 1/sqrt(nVox) normalization into q once; ctprod! then uses their conjugates.
        center_offsets = T[iseven(n) ? -0.5 : 0.0 for n in MatrixSize]
        center_phase = vec(sum(nfft_traj .* reshape(center_offsets, :, 1, 1); dims=1))
        center_correction = arrayType(exp.(Complex{T}(0, T(2π)) .* center_phase))
        q .*= q_scale .* reshape(zeroth_correction .* center_correction, :, 1)
    else
        q .*= q_scale .* reshape(zeroth_correction, :, 1)
    end

    nGrid = prod(grid.matrixSize)
    workspace_samples = basis_layout === :per_readout ? nSam : nSam * nDyn
    workspace = HighOrderLowRankWorkspace(
        fill!(similar(q, Complex{T}, nGrid), 0),
        similar(q, Complex{T}, nGrid),
        similar(q, Complex{T}, workspace_samples),
        similar(q, Complex{T}, workspace_samples),
        similar(q, Complex{T}, nVox),
        similar(q, Complex{T}, nVox),
        similar(q, Complex{T}, nVox),
    )
    normal_backend = HighOrderNormalBackend(normal_distribution, copy(gpus), verbose, nothing, nothing)

    Mv = Mtu = arrayType(Vector{Complex{T}}(undef, 0))  
    return HighOrderLowRankOp{T, Nothing, typeof(ctprod!), typeof(nfftplan), typeof(q), typeof(basis), typeof(csm), typeof(workspace), typeof(Mv)}(
        nSam, nVox, nCha, nDyn, L_rank,
        basis_layout, per_readout_storage, per_readout_cache_size, readout_cache, q, basis, csm,
        nfftplan, readout_plans, nfft_traj, mask_idx, grid.matrixSize, workspace, normal_backend,
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
function ensure_per_readout_cache!(op::HighOrderLowRankOp)
    if op.basis_layout === :per_readout && op.per_readout_storage in (:cpu, :regenerate)
        if op.readout_cache === nothing || op.readout_cache.released
            dims = op.grid_size[end] == 1 ? op.grid_size[1:2] : op.grid_size
            op.readout_cache = PerReadoutCache(op.basis, op.nfft_traj, dims;
                capacity=op.per_readout_cache_size, first_plan=op.nfftplan)
        end
    end
    return nothing
end

function prod!(y::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, x::AbstractVector{Complex{T}}) where T
    op.normal_backend.distribution === :rank && return rank_shared_forward!(y, op, x)
    ensure_per_readout_cache!(op)

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
    
    rank = size(op.q, 2)
    basis = op.basis

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    nBlocks = op.basis_layout === :per_readout ? op.nDyn : 1
    cache = op.readout_cache
    cache === nothing || prime_readout_cache!(cache)
    try
    for dyn in 1:nBlocks
        rows = op.basis_layout === :per_readout ? (((dyn - 1) * op.nSam + 1):(dyn * op.nSam)) : (1:nPoint)
        if cache === nothing
            plan = op.basis_layout === :per_readout ? op.readout_plans[dyn] : op.nfftplan
        else
            slot = acquire_readout!(cache, dyn)
            plan = first(cache.plans)
            basis = cache.basis[slot]
        end
        for r = 1:rank
            column = op.basis_layout === :per_readout && cache === nothing ? (dyn - 1) * op.L + r : r
            @views @. x_weighted = x_masked * conj(basis[:, column])
            for c = 1:op.nCha
                if grid_flat isa CuArray
                    @cuda threads=threads blocks=blocks_vox kernel_scatter_csm!(grid_flat, x_weighted, op.csm, c, op.mask_idx, op.nVox)
                else
                    @views @. grid_flat[op.mask_idx] = x_weighted * op.csm[:, c]
                end
                mul!(k_signal, plan, nfft_img)
                @views @. y_global[rows, c] += k_signal * op.q[rows, r]
            end
        end
        cache === nothing || finish_readout!(cache, dyn)
    end
    finally
        cache === nothing || drain_readout_copies!(cache)
    end

    return y
end

function ctprod!(x::AbstractVector{Complex{T}}, op::HighOrderLowRankOp{T}, y::AbstractVector{Complex{T}}) where T
    op.normal_backend.distribution === :rank && return rank_shared_adjoint!(x, op, y)
    ensure_per_readout_cache!(op)

    # A solver normally builds the multi-GPU normal backend before its initial
    # adjoint. Reuse its workers for shared and streamed per-readout RHSs.
    # Standalone adjoints retain their existing single-device execution path.
    cached = op.normal_backend.operator
    if cached isa MultiGPUHighOrderNormalOp && !cached.state.released &&
       x isa CuArray && y isa CuArray && length(x) == prod(op.grid_size)
        if op.basis_layout === :shared
            return multi_gpu_highorder_adjoint!(x, cached.state, y)
        elseif op.per_readout_storage in (:cpu, :regenerate)
            return multi_gpu_highorder_adjoint!(x, cached.state, y, op.csm)
        end
    end

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

    rank = size(op.q, 2)
    basis = op.basis

    threads = 256
    blocks_vox = cld(op.nVox, threads)

    nBlocks = op.basis_layout === :per_readout ? op.nDyn : 1
    cache = op.readout_cache
    cache === nothing || prime_readout_cache!(cache)
    try
    for dyn in 1:nBlocks
        rows = op.basis_layout === :per_readout ? (((dyn - 1) * op.nSam + 1):(dyn * op.nSam)) : (1:nPoint)
        if cache === nothing
            plan = op.basis_layout === :per_readout ? op.readout_plans[dyn] : op.nfftplan
        else
            slot = acquire_readout!(cache, dyn)
            plan = first(cache.plans)
            basis = cache.basis[slot]
        end
        for r = 1:rank
            column = op.basis_layout === :per_readout && cache === nothing ? (dyn - 1) * op.L + r : r
            fill!(x_l, zero(Complex{T}))
            for c = 1:op.nCha
                @views @. k_weighted = y_global[rows, c] * conj(op.q[rows, r])
                mul!(nfft_img, adjoint(plan), k_weighted)
                if grid_flat isa CuArray
                    @cuda threads=threads blocks=blocks_vox kernel_gather_csm!(x_l, grid_flat, op.csm, c, op.mask_idx, op.nVox)
                else
                    @views @. x_l += grid_flat[op.mask_idx] * conj(op.csm[:, c])
                end
            end
            @views @. x_masked += x_l * basis[:, column]
        end
        cache === nothing || finish_readout!(cache, dyn)
    end
    finally
        cache === nothing || drain_readout_copies!(cache)
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
