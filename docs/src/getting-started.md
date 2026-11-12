# Getting Started

This guide walks through installation, a small CPU encoding problem, and a CG reconstruction. It then shows how to use LowRank with the same inputs. The example generates its own data so that the input layout and operator calls can be followed without an external dataset.

## Requirements

- Use Julia 1.12, the baseline declared in the code checkout's `Project.toml`.
- The CPU examples below explicitly select `arrayType=Array`.
- CUDA execution additionally requires an NVIDIA GPU and a functional CUDA.jl installation. Memory requirements depend on the grid, mask, samples, coils, and selected representation.

The package resolves `MRIGeometry` from its included local source tree. These
development pages are bound to the code SHA shown by the site's source links.
Install from a separate clone of `dev_jinyuan` for the documented opt-in Joint
adaptive low-rank path; `main` may not yet contain it. For an exact reproduction,
check out the displayed source SHA before instantiating.

```bash
git clone --branch dev_jinyuan https://github.com/BennyZhang-Codes/HighOrderMRI.jl.git HighOrderMRI-code
cd HighOrderMRI-code
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

For a release installation, run `git checkout` with the tag displayed in the version menu after `cd` and before `Pkg.instantiate()`. The full code commit shown in the footer selects the exact documented implementation.

Start Julia in that environment:

```bash
julia --project=.
```

Run the Julia blocks below in order. The `docs` branch supplies the VitePress website; use the package checkout and its Julia project to run reconstruction code.

## First CPU operator

`HighOrderOp` evaluates the full encoding phase explicitly. Its forward product maps an image to samples; its adjoint maps samples back to the image grid. Reconstructing an image from measured data is a separate inverse problem.

The example uses an 8 × 8 grid, one readout, one coil, and a complete Cartesian first-order trajectory. Coordinates are in metres, `times` in seconds, and `fieldmap` in Hz. `kspha` contains time-integrated coefficients in cycles per corresponding basis unit; the constructor does not integrate them.

```julia
using HighOrderMRI

T = Float32
nX, nY, nZ = 8, 8, 1
nSam = nX * nY

grid = Grid(nX, nY, nZ, T(1e-3), T(1.2e-3), T(5e-3))
times = collect(range(zero(T), T(2e-3); length=nSam))

# Rows: zeroth order, x, y, z, and five second-order terms.
kspha = zeros(T, 9, nSam)
kx = T.(-4:3) ./ (T(nX) * grid.Δx)
ky = T.(-4:3) ./ (T(nY) * grid.Δy)
kspha[1, :] .= T(0.05)                            # cycles
kspha[2, :] .= repeat(kx, nY)                     # cycles/m
kspha[3, :] .= repeat(ky; inner=nX)               # cycles/m
kspha[5, :] .= range(T(-500), T(500); length=nSam) # cycles/m²

fieldmap = reshape(
    collect(range(T(-15), T(15); length=nX * nY)), nX, nY)
csm = ones(Complex{T}, nX, nY, 1)
mask = trues(nX, nY)

op = HighOrderOp(grid, kspha, times;
    fieldmap, csm, mask, arrayType=Array, nBlock=4)

image = zeros(Complex{T}, nX, nY)
image[3:6, 3:6] .= one(T)
signal = reshape(op * vec(image), nSam, 1)
backprojection = reshape(adjoint(op) * vec(signal), nX, nY)
```

`signal` has shape `(nSam,nCha)`. `backprojection` is the adjoint result, rather than the solution of the inverse problem. The forward phase is positive and the symmetric normalization is `1/sqrt(count(mask))`; use the [encoding model](/theory/encoding-model) for the exact equation, field-row order, and coordinate conventions.

For multiple readouts, LowRank uses `kspha` shaped `(nTerm,nSam,nDyn)`, `times` shaped `(nSam,nDyn)`, and data shaped `(nSam,nDyn,nCha)`. Here readout index $i$ runs from 1 to $N_{\mathrm{ro}}=\mathtt{nDyn}$, with $N_t=\mathtt{nSam}$ samples per readout. All readouts encode the same unknown image. Prepare the matrix passed to the reconstruction wrapper with `reshape(data, nSam * nDyn, nCha)`, without permuting axes.

## First reconstruction

The following block continues the CPU example. `recon_HOOp` applies weights to both the data and operator, constructs the data normal, and calls the selected solver. This example chooses CGNR with L2 regularization:

```julia
using RegularizedLeastSquares

weight = ones(Complex{T}, nSam)
rec_params = Dict{Symbol,Any}(
    :solver => CGNR,
    :reg => L2Regularization(T(1e-4)),
    :iterations => 20,
    :reconSize => (nX, nY),
)
reconstructed = recon_HOOp(op, signal, weight, rec_params)
```

The result is a CPU array with `size(reconstructed) == (nX,nY)`. Supply one raw complex weight per sample/readout point; the wrapper repeats it across coils. Its normal uses `abs2.(weight)`, so do not square these weights before passing them. The unit weights above are sufficient for this Cartesian example. For non-Cartesian sampling and measured data, see [density weighting and solver setup](/guide/reconstruction) and the [`recon_HOOp` reference](/reference/recon-hoop).

## Use the LowRank representation

LowRank keeps first-order Fourier encoding in the NFFT and approximates the remaining off-resonance/higher-order phase. The default layout incrementally builds a shared spatial basis across readouts; the default reconstruction distribution is `:single`.

Use the same CPU model inputs from above:

```julia
op_lr = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    arrayType=Array,
    L_rank=4,
    rsvd_oversample=4,
    rsvd_seed=1234,
    shared_basis_tol=T(1e-3),
)
signal_lr, reconstructed_lr = try
    samples = reshape(op_lr * vec(image), nSam, 1)
    result = recon_HOOp(op_lr, signal, weight, rec_params)
    samples, result
finally
    close(op_lr)
end
```

The rank and tolerance here are choices for this small example. `L_rank + rsvd_oversample` must not exceed `min(nSam,count(mask))`. The shared tolerance controls compression of retained local factors; it is not an image-error tolerance. Read the [low-rank derivation](/theory/low-rank) for this distinction and the [LowRank API](/reference/highorderlowrankop) for exact defaults, storage choices, and cleanup.

## CUDA availability

Inspect CUDA before selecting a CUDA backend:

```julia
using CUDA

CUDA.functional()
CUDA.devices()
```

For a CUDA operator use `arrayType=CuArray`. Keep real model inputs and complex data/CSM at matching precision. For CUDA LowRank reconstruction and resident explicit backends, pass both the data matrix and weight vector as CUDA arrays on the primary GPU; `recon_HOOp` does not transfer them automatically. It still returns a CPU image.

`gpus` uses zero-based CUDA-visible device IDs, for example `[0,1]`. Multi-GPU setup and reconstruction are separate stages: `rsvd_distribution` controls LowRank setup, while `normal_distribution` selects `:single`, `:channel`, `:rank`, or `:readout` subject to their layout/backend restrictions. Rank execution partitions shared factor columns; channel execution partitions receive coils. The [multi-GPU guide](/guide/multi-gpu) gives the exact restrictions, sample batching behavior, memory scope, cleanup, and a complete rank example.

For the multi-GPU task/stream paths, start Julia with sufficient default threads:

```bash
julia --threads=auto --project=.
```

Use the default centred Cartesian grid on active NFFT axes; editing coordinate vectors alone does not update LowRank's Fourier encoding. See [coordinate scope](/theory/encoding-model#coordinate-scope) before applying a permutation, reversal, or physical offset.

## Build the documentation locally

The website source lives in this package's `docs/` directory on `main` and
`dev_jinyuan`. It uses VitePress with Node.js 22 and Julia compatible with the
selected code commits. Publication, page validation and preview tools use an
independent Julia environment in `docs/publish`; PyCall/PyPlot remain the
existing plotting backend for package examples.

Pushes to `dev_jinyuan` publish `dev/`; pushes to `main` publish `main/`.
A published, non-prerelease `vX.Y.Z` release adds its version directory and
advances `stable/` to the newest formal release. The root URL opens stable.
Pull requests build labelled candidates without publishing.

The workflow resolves exact code and documentation commits at build time.
Branch pages use the `docs/` stored with their code commit; new release pages
use the tagged commit's own `docs/`. Existing archived release identities,
including the original separately frozen `v0.1.0` documentation, remain fixed.
The complete versioned build still checks API coverage, CPU examples, links,
math and figures before publishing any pages.

`gh-pages` stores only generated output and provenance. After all versions
pass, the workflow updates that archive and deploys the same archive commit.
Archive updates retain history, protect published release identities and skip
superseded branch builds. No dedicated documentation source branch or SHA
synchronization commit is required. Handwritten guides and theory are edited
in `docs/src/`; API fragments are generated from code docstrings.

For an existing formal release, run the Documentation workflow from `main`
or `dev_jinyuan` with `release_tag` set to that existing published tag. This
rebuilds its frozen inputs without moving the tag. See `docs/RELEASING.md`
for the release and local-build contract.

The legacy `code-version.json` and `versions.json` retain the original frozen
snapshots for explicit local builds. CI uses its resolved publication plan,
recorded in the generated `publication-plan.json` and `versions.json`.
For a local fixed-snapshot preview, prepare exact clean code checkouts under
`DOCS_CODE_ROOT`, one directory per full code SHA, then run:

```bash
julia --startup-file=no --project=docs/publish -e 'using Pkg; Pkg.instantiate()'
npm ci --prefix docs
export DOCS_CODE_ROOT=/absolute/path/to/prepared-code-checkouts
export DOCS_CHECK_EXAMPLES=1
npm run docs:build-site --prefix docs -- --preview
npm run docs:preview --prefix docs
```

To edit interactively, supply the selected exact code checkout as
`DOCS_CODE_PATH`, run `npm run docs:generate-api --prefix docs -- --preview`,
then `npm run docs:dev --prefix docs`. The generated API fragments and website
output are disposable build products.

## Reading paths

**Understand the model**

- [Architecture overview](/concepts-overview): fields, measurements, operators, and reconstruction.
- [Symbols and notation](/theory/symbols): indices, units, matrix dimensions, and Julia array names.
- [Expanded encoding model](/theory/encoding-model): the complete phase and its Fourier/residual separation.
- [Low-rank shared subspace](/theory/low-rank): local rSVD, adaptive sharing, optional final recompression, and approximation error.

**Prepare measured data and reconstruct**

- [Field preprocessing](/guide/field-preprocessing): GIRF prediction, interpolation, and field/data synchronization.
- [Coil compression](/guide/coil-compression): applying the same fitted transform to data and CSM.
- [Encoding operators](/guide/operators): choosing explicit or LowRank execution.
- [Reconstruction workflow](/guide/reconstruction): data layout, density weights, and solver use.
- [Multi-GPU execution](/guide/multi-gpu): setup and reconstruction distributions, including rank sharding.

**Look up an interface or source**

- [API reference](/api): constructors, reconstruction, preprocessing, and resource management.
- [Literature references](/references): the scientific sources cited by the principles pages.
- [Scientific validation strategy](/guide/validation): the distinction between implementation consistency and physical validation.
