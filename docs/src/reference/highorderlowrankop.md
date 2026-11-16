# `HighOrderLowRankOp`

Use `HighOrderLowRankOp` to apply an MRI encoding model repeatedly without
materializing its full sample-by-voxel phase matrix. It combines NFFT Fourier
encoding with low-rank factors for the remaining phase. The default construction
is Incremental Shared: local randomized SVD (rSVD) followed by incremental spatial sharing.
See the [low-rank derivation](/theory/low-rank) for the equations, approximation
stages and references.

Readout index $i$ runs over `nDyn` encoding profiles of the **same image**.
Input coefficients have shape `(nTerm,nSam,nDyn)`, and times have shape
`(nSam,nDyn)`. The single-readout constructor also accepts a coefficient matrix
and a time vector. The generated [API below](#api) gives the complete signatures,
defaults and return layout.

::: info Algorithm selection
Use `method=:per_readout`, `:incremental_shared`, or `:direct_shared`.
Direct Shared selects `strategy=:per_readout` (default for that method) or
`:global`; Global additionally requires `epsilon_H`. Construction settings
are independent of execution and storage. Use the
[method guide](/guide/low-rank) for the implementation pages.
:::

## A complete CPU example

This example defines a small Cartesian grid, two readouts, two coils, a nonzero
static field map and a varying second-order field term. It computes synthetic
samples and an adjoint backprojection; it does not solve an inverse problem or
test convergence on measured data.

```julia
using HighOrderMRI
using LinearAlgebra

T = Float32
nX, nY = 8, 8
nSam, nDyn, nCha = nX * nY, 2, 2
grid = Grid(nX, nY, 1, T(1e-3), T(1e-3), T(1e-3))

# Metres, seconds, Hz; the phase coefficients use cycles and cycles/m^p.
kx = T.(-4:3) ./ (T(nX) * grid.Δx)
ky = T.(-4:3) ./ (T(nY) * grid.Δy)
times = repeat(reshape(T.(0:nSam-1) .* T(1e-5), :, 1), 1, nDyn)
kspha = zeros(T, 9, nSam, nDyn)
for i in 1:nDyn
    kspha[1, :, i] .= T(0.03) * T(i)
    kspha[2, :, i] .= repeat(kx, nY)
    ky_samples = repeat(ky; inner=nX)
    kspha[3, :, i] .= i == 1 ? ky_samples : reverse(ky_samples)
    kspha[5, :, i] .= T(i) .* range(T(-500), T(500); length=nSam)
end

fieldmap = reshape(collect(range(T(-15), T(15); length=nX*nY)), nX, nY, 1)
csm = ones(Complex{T}, nX, nY, 1, nCha)
csm[:, :, 1, 2] .= Complex{T}(T(0.8), T(0.2))
mask = trues(nX, nY, 1)
image = zeros(Complex{T}, nX, nY, 1)
image[3:6, 3:6, 1] .= Complex{T}(one(T), T(0.25))
image[4:5, 4:5, 1] .= Complex{T}(T(0.5), T(-0.2))

op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    arrayType=Array,
    method=:incremental_shared,
    normal_distribution=:single,
    L_rank=4,
    rsvd_oversample=4,
    rsvd_seed=1234,
    shared_rank_max=8,
    shared_basis_tol=T(1e-3),
)

signal, backprojection = try
    samples = reshape(op * vec(image), nSam, nDyn, nCha)
    adjoint_image = reshape(adjoint(op) * vec(samples), nX, nY, 1)
    samples, adjoint_image
finally
    close(op)
end

@assert size(signal) == (nSam, nDyn, nCha)
@assert norm(signal) > zero(T)
```

Samples vary fastest, then readouts, then coils. For reconstruction, preserve
that ordering and form `reshape(signal, nSam*nDyn, nCha)` for
[`recon_HOOp`](/reference/recon-hoop). The backprojection above is $A^H y$,
not a reconstructed solution. Rank and tolerance values in this small example
are illustrative choices, not settings validated for a full brain.

Use the default centred Cartesian coordinates on the active NFFT axes. In 2D,
only $k_x$ and $k_y$ enter the NFFT; $k_z z$ remains in the residual phase.
Changing grid coordinate vectors alone does not change the NFFT geometry.
See [coordinate scope](/theory/encoding-model#coordinate-scope) before applying
permutations, reversals or offsets.

## Incremental Shared construction {#shared-basis-parameters}

For each readout $i$, local rSVD retains `L_rank` factors of the residual phase
matrix. The sketch must satisfy
`L_rank + rsvd_oversample <= min(nSam,count(mask))`. Incremental sharing then
combines those retained spatial factors. When the basis grows, earlier saved
coefficient blocks are zero-padded; they are not recomputed as projections onto
the completed basis. This procedure is not a global optimal SVD.

`shared_basis_tol` controls the incremental compression of retained local
factors. It is not a bound on the original full phase matrix or image error.
Optional `global_basis_tol` introduces a further error relative to the already
shared representation. NFFT approximation and local rSVD truncation remain
separate error sources.

### Optional sampled construction and weighted compression

These additions are disabled by default. `shared_readout_count=nothing`
uses every readout, `shared_fit_samples=nothing` keeps the original local
temporal projection, and no new compression runs unless explicitly requested.
Existing `global_basis_tol` calls keep the original unweighted path.

For shared rSVD, use `shared_readout_count=64, shared_fit_samples=512` to
train a spatial basis on representative readouts and fit coefficients for
every original time sample. The numbers are configurable experiment settings,
not general accuracy recommendations. Readout selection uses a sampled
phase-geometry proxy; it is not a full-time coverage certificate. Supplying
`shared_basis_report=Ref{Any}()` records selected readouts, training voxels,
conditioning and phase checks, including a post-compression check.

`global_basis_rank=18` selects a fixed final rank instead of
`global_basis_tol`. With either selector, `global_basis_weights=D` uses real
nonnegative sample weights already equal to `abs2.(w)`; they are not squared
again. `global_basis_coil_weighting=true` also uses summed CSM energy.
The weights define the compression metric only. Reconstruction still receives
its own original weights. This path uses blocked FP64 Grams and accounts for
the actual spatial Gram. Its tail error concerns the existing factors, not the
original phase matrix or reconstructed image. A fixed `global_basis_rank` is
not an image-error target; inspect `compressed_phase_check` and complete
reconstruction metrics before choosing it. Direct Shared retains its
own validation and rejects these post-compression options.

### Optional no-add check

`shared_noadd_check` affects **basis construction**, independently of the
reconstruction distribution. It is disabled by default and requires the shared,
voxel-distributed CUDA rSVD path with Gram finalization.

The update still performs both projection passes and forms the full residual.
It accumulates residual energy in FP64 and, when a numerical guard places that
energy safely below the tolerance allowance, skips the residual-Gram matrix
and its eigendecomposition. Other cases use the original path. The check does
not change the requested local rank or reconstruction sample batches. It does
not establish a memory or runtime benefit for a particular measured dataset.

## Independent readouts {#per-readout-parameters}

`method=:per_readout` retains the local factors independently, without
incremental sharing or final shared-basis recompression. The resident option
stores those factors on the selected array backend. CUDA also supports host
factor streaming (`per_readout_storage=:cpu`) and regeneration from retained
Q/Z factors (`:regenerate`), with their additional setup and cache constraints
listed in the generated API. These storage modes do not select the CG
partition by themselves.

## Choose setup and reconstruction execution separately

`rsvd_distribution` controls factor construction. `normal_distribution` controls
operator execution during iterative reconstruction; its default is `:single`.
For rank execution it also selects the direct forward and adjoint paths.

| `normal_distribution` | Work and persistent storage |
|:--|:--|
| `:single` | CPU with `arrayType=Array`, or the primary GPU with `CuArray`. |
| `:channel` | Coil groups. In the shared layout, every worker retains the full temporal and spatial factors; coil-map storage is partitioned. |
| `:rank` | Shared factor columns and coil-map storage are partitioned. Complete factor and coil-map masters remain in host RAM. |
| `:readout` | Independent readout groups, requiring `method=:per_readout`; every worker retains all coil maps. |

The distributed reconstruction modes require CUDA and at least two unique
GPU IDs. IDs are zero based within CUDA's visible device list, and `first(gpus)`
is the primary GPU for solver vectors. Rank execution supports Incremental Shared with voxel-distributed Gram setup
and the Direct Shared Global strategy. It does not support the Direct Shared
Per-readout strategy.
Its worker count is at most `min(length(gpus), shared_rank, nCha)`, so selecting
several devices does not guarantee they all participate.

The setup requirement is different: explicit `rsvd_distribution=:voxel` accepts
**one or more unique GPU IDs**; the list must be nonempty and must not contain more devices than masked voxels. Automatic voxel setup
requires multiple GPUs, the kernel backend and Gram finalization; chunked voxel
setup requires explicit selection.
The generated API gives the exact combinations. Use the
[multi-GPU guide](/guide/multi-gpu) for complete device configurations.

### Optional batched CUDA execution

`HighOrderMRINonuniformFFTsExt` is an automatically loaded Julia package
extension for HighOrderMRI and NonuniformFFTs. It adds methods for the existing
NonuniformFFTs backend; it does not register an independent NUFFT backend or
provide another `backend` value. Loading the extension alone leaves
`normal_method=:standard` in effect. The selections have separate roles:

| Setting | Role | Default |
|:--|:--|:--|
| `method` | Build Per-readout, Incremental Shared or Direct Shared factors. | `:incremental_shared` |
| `strategy` | Select Direct Shared's Per-readout or Global strategy. | `nothing`, selecting Per-readout for Direct Shared |
| `normal_distribution` | Assign the reconstruction work to devices. | `:single` |
| `normal_method` | Evaluate completed factors with standard or batched kernels. | `:standard` |
| `AbstractNFFTs.set_active_backend!` | Choose the NUFFT library. | Existing active backend. |

`normal_method=:standard` keeps existing execution. `normal_method=:batched`
opts into Julia CUDA kernels and batched cuFFT for 3D Float32 shared factors,
with either Incremental Shared or Direct Shared construction. Load the optional backend first:

```julia
using HighOrderMRI, CUDA, AbstractNFFTs, NonuniformFFTs
AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
```

NonuniformFFTs itself activates its backend when imported, as it did before
this extension; the explicit selection above makes the intended library clear.
The HighOrderMRI extension does not switch backends. Check that it loaded with
`Base.get_extension(HighOrderMRI, :HighOrderMRINonuniformFFTsExt)`. For an
updated path-tracked checkout, `Pkg.resolve()` in the active environment may
be needed to refresh extension metadata; restart Julia afterward.

Then pass `normal_method=:batched, normal_coil_batch_size=8` to
`HighOrderLowRankOp`. Coil batches 2, 4 and 8 are supported, including a partial
last batch. Larger batches use more memory. `normal_distribution=:single`
uses one GPU; `:channel` retains the existing multi-GPU coil partition.
CPU, 2D, Float64, independent-readout and rank-sharded evaluation continue to
use the standard path. Explicitly requesting an unsupported batched
configuration throws an error.

The extension is version-bound to the tested NonuniformFFTs 0.9.6 internals.
It does not change the selected backend globally, trajectory nodes, logical
FFT sizes, NUFFT support or factors. It uses no native CUDA binaries or VkFFT.
The initial normal request owns its workspaces through the existing parent;
`close(op)` releases them, including batched single-GPU execution. Bare
adjoints never apply the normal weights again. Floating-point accumulation
order may differ from the standard implementation. The parent's ordinary
forward remains on its existing path; a bare adjoint reuses the batched workers
only after the first normal request initializes them.

For existing 3D Float32 inputs, execution-only opt-in looks like:

```julia
using LinearOperatorCollection: normalOperator

op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    arrayType=CuArray, gpus=[0, 1],
    normal_distribution=:channel,
    normal_method=:batched, normal_coil_batch_size=8,
)
try
    D = CuArray(abs2.(w))             # One WᴴW value per sample, not per coil.
    N = normalOperator(op, D)        # Initialize before a solver's bare adjoint.
    # Apply N or run the existing reconstruction workflow with the same settings.
finally
    close(op)
end
```

This leaves the factor-construction settings unchanged. Incremental Shared sampling,
coefficient refitting and weighted compression in the earlier section are
separate opt-ins that change the approximation. Direct Shared retains its own rank
selection and checks; batching does not remove a high-B0 audit warning.

### Experimental pruned FFT combination

`normal_options=(;)` retains the batched Julia kernels and cuFFT. Optional native
build sources and instructions are in `deps/pruned_fft/README.md`. Set the
documented build environment variables and explicitly run
`bash deps/pruned_fft/build.sh <output-directory>` from the package root.
After building local resources, the batched path also accepts:

```julia
normal_options=(
    fft=:vkfft_pruned,
    native_ptx="/path/to/resources/highorder_native.ptx",
    vkfft_library="/path/to/resources/libhighorder_vkfft.so",
    coefficient_cache_bytes=1 << 30,
)
```

This experiment combines native scatter/unrolled gather, VkFFT pruning of known
padding intervals, compact image padding and budgeted Float32 coefficient
caching. The budget is per worker; zero disables the cache. Full weights, masked
voxels, coils, factors and the logical NUFFT grid remain unchanged. The parent
owns and closes each worker's foreign FFT plan and native module. Missing
resources and unsupported configurations throw errors. The experiment requires
local compiled resources; it is not a portable binary distribution. It is
independent of factor construction and remains disabled by default.

### Why rank execution must combine forward contributions

Coil groups contain independent observation rows, so their normal contributions
can be summed. Rank groups instead contribute to the same samples. If
$A=\sum_g A_g$, then

$$
A^H W^H W A = \sum_g\sum_h A_g^H W^H W A_h.
$$

The implementation combines forward samples across rank workers before applying
the weights and adjoint. Summing only $A_g^H W^H W A_g$ would omit the cross-rank
terms. The reductions and transfers use host staging buffers; GPU memory is
not a single shared pool. Each rank worker evaluates all coils. If a coil map is absent from its local
storage, the worker uploads that map from the host master.

### Sample batching and data placement

`normal_sample_batch_size` limits the number of sample/readout points in each
rank-mode batch. Its effective size is
`min(nSam*nDyn, normal_sample_batch_size)`. It bounds sample workspaces, not
complete factor columns, coil maps, image grids, solver vectors, input/output
data, host masters or setup buffers. The other reconstruction distributions do
not use this parameter for batching.

With the current NFFT.jl 0.14.3 GPU backend, use a full batch or an exact divisor
of `nSam*nDyn`: its node-update path does not support a shorter final output
batch. See [sample batching](/guide/multi-gpu#sample-batching-and-memory) for the
backend-specific limitation. Reducing a batch cap therefore is not sufficient
to establish that the complete problem fits in GPU memory.

In rank mode, `op.q`, `op.basis` and `op.csm` are host matrices, while solver
storage is on the primary GPU. Consequently, allocating `op * x` returns a host
vector in this mode. Use `mul!` with an appropriately sized GPU output, or
transfer the result before reconstruction. CUDA LowRank reconstruction requires
the data matrix and raw weight vector on the primary GPU; `recon_HOOp` does not
move them automatically. It returns a CPU image.

## Release owned resources

Use `try`/`finally` and `close(op)` after the final direct forward, adjoint or
normal use. `recon_HOOp` releases the distributed backend by default; if
`release_backend=false`, the caller retains that responsibility.

The parent operator owns the workers. `release_highorder_normal_backend!(op)`
releases that backend; `close(op)` also releases any source streaming or
regeneration queue. Closing a derived normal view alone does not release the
parent's workers. For distributed normal execution, release the existing backend before changing
normal weights and obtain a fresh normal view. Host factor masters remain referenced until
the parent and other references are dropped. See
[resource lifecycle](/reference/resources) before replacing a large operator.

## API

The following source-generated reference separates the operator type, the
single-readout convenience constructor and the multi-readout constructor.

<!--@include: ../../.generated/api/highorderlowrankop.md-->
