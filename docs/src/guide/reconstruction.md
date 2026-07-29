# Reconstruction workflow

A reproducible reconstruction has five stages:

1. prepare synchronized field coefficients and sampling times;
2. construct the physical grid and spatial inputs;
3. select and validate an encoding operator;
4. create density weights and solver settings;
5. reconstruct and report raw complex accuracy, convergence, timing, and
   provenance.

## Prepare data

For a low-rank dynamic operator, use:

```text
kspha  :: (nTerm, nSam, nDyn)
times  :: (nSam, nDyn)
data   :: (nSam, nDyn, nCha)
weight :: (nSam, nDyn)
```

Create the 2D matrix accepted by [`recon_HOOp`](@ref) without changing the
vector order:

```julia
data_matrix = reshape(data, nSam * nDyn, nCha)
weight_vector = vec(weight)
```

Apply every axis exchange, reversal, and circular shift consistently to the
image, mask, field map, and coil maps before constructing the operator.

## Density weights

For a 2D first-order trajectory:

```julia
weight_vector = samplingDensity(
    reshape(kspha[2:3, :, :], 2, :),
    (grid.nX, grid.nY),
)
```

The helper returns square-root density compensation weights compatible with
the weighting operator used by `recon_HOOp`. Reuse the exact same weights
when comparing encoding implementations.

## Solver configuration

The reconstruction wrapper uses MRIReco.jl and
RegularizedLeastSquares.jl:

```julia
using RegularizedLeastSquares

rec_params = Dict{Symbol,Any}(
    :reconSize => (grid.nX, grid.nY),
    :reg => L2Regularization(1f-9),
    :iterations => 20,
    :solver => CGNR,
)
```

For 3D, set `:reconSize => grid.matrixSize`. Keep the solver, regularizer,
initialization, iteration count, and stopping condition identical across a
comparison.

## Reconstruct

```julia
image = recon_HOOp(
    op,
    ComplexF32.(data_matrix),
    ComplexF32.(weight_vector),
    rec_params,
)
```

For a channel-distributed `HighOrderLowRankOp`, the multi-GPU normal backend
is created lazily. `recon_HOOp` releases it in a `finally` block by default.
Use `release_backend=false` only when the same operator and normal backend
will immediately be reused, then call `close(op)` after the final solve.

## Accuracy metrics

Raw complex NRMSE is the primary comparison:

```julia
complex_error = raw_complex_nrmse(image, reference)
mag_error = magnitude_nrmse(image, reference)
ssim = magnitude_ssim(image, reference)
```

`aligned_complex_nrmse` and `magnitude_nrmse(...; align=true)` remove a global
least-squares complex scale. They are secondary metrics and must be labelled
as aligned.

For a low-rank study, also record:

- `L_rank`, final shared rank, `shared_basis_tol`, and `shared_rank_max`;
- rSVD backend, distribution, finalization, oversampling, and seed;
- forward, adjoint, normal, and adjoint-identity errors;
- setup, one forward/adjoint/normal evaluation, solver, and end-to-end time;
- fixed-iteration and common-stopping-condition results;
- peak host and device memory;
- Git SHA, dirty state, Julia/CUDA/package versions, and GPU UUIDs.

The [reconstruction protocol](reconstruction-protocol.md) gives the full
frozen comparison rules.

## Timing CUDA work

Compilation and asynchronous execution can make a single wall-clock timing
misleading. Warm the exact path first and synchronize every participating GPU
around measured regions. Use repeated steady-state measurements and report a
distribution such as median and IQR.

Do not use post-reconstruction intensity fitting to hide a sign, centring,
normalization, or data-layout mismatch.
