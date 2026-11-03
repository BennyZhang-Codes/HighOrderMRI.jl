# `HighOrderLowRankOp`

Use `HighOrderLowRankOp` to apply an MRI encoding model repeatedly without
materializing its full sample-by-voxel phase matrix. It combines NFFT Fourier
encoding with low-rank factors for the remaining phase. The default construction
uses local randomized SVD (rSVD) followed by an incremental shared spatial basis.
See the [low-rank derivation](/theory/low-rank) for the equations, approximation
stages and references.

Readout index $i$ runs over `nDyn` encoding profiles of the **same image**.
Input coefficients have shape `(nTerm,nSam,nDyn)`, and times have shape
`(nSam,nDyn)`. The single-readout constructor also accepts a coefficient matrix
and a time vector. The generated [API below](#api) gives the complete signatures,
defaults and return layout.

::: info Construction covered here
This page covers `shared_basis_method=:rsvd`. The generated signature also lists
the implemented `joint_*` keywords; their construction method and evaluation
are outside this page's scope.
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
    shared_basis_method=:rsvd,
    basis_layout=:shared,
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

## Shared-basis construction {#shared-basis-parameters}

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

`basis_layout=:per_readout` retains the local factors independently, without
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
| `:readout` | Independent readout groups, requiring `basis_layout=:per_readout`; every worker retains all coil maps. |

The distributed reconstruction modes require CUDA and at least two unique
GPU IDs. IDs are zero based within CUDA's visible device list, and `first(gpus)`
is the primary GPU for solver vectors. Rank execution additionally requires
shared rSVD factors built by the voxel-distributed path with Gram finalization.
Its worker count is at most `min(length(gpus), shared_rank, nCha)`, so selecting
several devices does not guarantee they all participate.

The setup requirement is different: explicit `rsvd_distribution=:voxel` accepts
**one or more unique GPU IDs**; the list must be nonempty and must not contain more devices than masked voxels. Automatic voxel setup
requires multiple GPUs, the kernel backend and Gram finalization; chunked voxel
setup requires explicit selection.
The generated API gives the exact combinations. Use the
[multi-GPU guide](/guide/multi-gpu) for complete device configurations.

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
