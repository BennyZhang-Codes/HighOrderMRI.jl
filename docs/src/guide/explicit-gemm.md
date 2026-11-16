# Explicit: GEMM and phase-reuse {#explicit-gemm-and-phase-reuse}

**GEMM** means **General Matrix–Matrix Multiplication**. It is a standard
matrix operation, not an MRI-specific algorithm or a GPU-only operation:

$$
C \leftarrow \alpha\,\operatorname{op}(A)\operatorname{op}(B)+\beta C.
$$

Here `op` can leave a matrix unchanged, transpose it, or take its conjugate
transpose. Our GEMM and phase-reuse path constructs a complex encoding-phase
block $P_b$ and uses cuBLAS for its matrix products. Custom CUDA kernels still
handle phase/exponential generation, masks and coil reduction.

Unlike [Direct kernel](/guide/explicit-kernel), this path stores a
phase block and delegates the encoding sum to matrix multiplication. Even a
standalone forward uses GEMM. Phase reuse in a normal application is a further
execution optimization, not the definition of GEMM.

Use `HighOrderExplicitOp(...; implementation=:kernel)` with `backend=:blocked`, `:cached`, or `:cached_full`. `gpus=[id]` selects one GPU and multiple IDs select the internal multi-GPU implementation. The default remains `backend=:legacy`; choose a resident backend explicitly.

```julia
using CUDA

CUDA.math_mode!(CUDA.PEDANTIC_MATH)
op = HighOrderExplicitOp(grid, kspha, times;
    implementation=:kernel,
    fieldmap, csm, mask,
    gpus=[0, 1],
    backend=:blocked,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    block_size=1024,
    phase_cache_bytes=0,
)
try
    image = recon_HOOp(op, CuArray(ComplexF32.(data_matrix)),
        CuArray(ComplexF32.(weight_vector)), rec_params)
finally
    close(op)
end
```

Use Float32 model inputs with ComplexF32 data, or matching Float64/ComplexF64 inputs. Construct the model from host arrays. The first selected GPU owns the data, weights, and solver vectors; the CPU receives only the returned image. IDs are zero-based and relative to `CUDA_VISIBLE_DEVICES` when the visible list is filtered.

## Phase reuse and caching

`block_size` counts sample rows, whereas legacy `nBlock` counts blocks. A blocked normal generates each complex phase block once and reuses it for all coils, forward multiplication, weighting, and adjoint multiplication. `:direct` uses direct encoding kernels; `:cached` stores the complete phase matrix but applies it in blocks; `:cached_full` uses one full GEMM per forward/adjoint. The two full-cache modes initialize in blocks.

For ComplexF32, the full complex matrix requires `8 * nSam * count(mask)` bytes, before phase scratch, CSM, coil images, data, solver, and library workspace. Feasibility must hold on each participating GPU. Include cache initialization and all required transfers in complete reconstruction timing.

With `backend=:blocked`, `phase_cache_bytes` limits a cached prefix on each GPU separately; zero is the default. The budget counts only complex phase entries and rounds down to complete sample blocks. Remaining samples are regenerated. It does not reduce the problem or approximate the phase, and excludes other workspaces. Tune the budget and block size together on the complete problem.

## Weighting and resource lifetime

`explicit_gpu_normal(op; weights)` accepts raw square-root weights and squares them once. `normalOperator(op, D)` instead accepts the already squared diagonal `D=WᴴW`; `normalOperator(W ∘ op)` supplies that diagonal automatically. Regularization remains in the solver. See the [API](/reference/explicit-cuda#weighted-normal).

Operators and their normals share mutable workspaces. Use them sequentially and call `close(op)` after the final use, including after `recon_HOOp`. For multi-GPU sample partitioning, see [Multi-GPU execution](/guide/multi-gpu#resident-explicit-encoding-sample-decomposition).


## Phase generation and normal reuse

The generated complex block has entries
$P_b[s,v]=\exp(2\pi\mathrm{i}\phi_{sv})$. `phase_backend=:gemm` uses a
real GEMM for the dynamic field/basis product, then adds the B0 phase and
evaluates the complex exponential. `phase_backend=:kernel` evaluates that
phase sum in a CUDA kernel. Both produce the same explicit representation.

For coil images $Z$, a normal application evaluates $Y=P_bZ$, weights $Y$
with the already squared diagonal, and accumulates $P_b^H Y$ using the same
block. Coil reduction and voxel normalization follow. `gemm_backend=:cublas`
uses native cuBLAS for the matrix products. This is phase reuse within an
explicit normal, not low-rank factorization. Without a cache, separately
requested forward and adjoint applications each generate their phase blocks;
reuse across those calls or CG iterations requires an explicit cache.

A partial cache keeps only complete blocks of a prefix; a full cache stores
all samples. The array dimensionality alone does not establish feasibility.
For ComplexF64 the complex phase storage doubles relative to ComplexF32.
Include setup, phase generation, cache initialization, RHS, solver and all
required transfers when comparing complete reconstruction costs.
