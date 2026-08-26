# Getting Started

## Requirements

- Julia 1.12 or later;
- an NVIDIA GPU and a functional CUDA.jl installation for CUDA execution;
- sufficient host and device memory for the selected operator, spatial mask, and receive-coil count.

The current repository resolves `MRIGeometry` from the included local source tree, so installation from a repository clone preserves the tested dependency layout:

```bash
git clone https://github.com/BennyZhang-Codes/HighOrderMRI.jl.git
cd HighOrderMRI.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Start Julia in the project environment:

```bash
julia --project=.
```

and load the package:

```julia
using HighOrderMRI
```

## First CPU operator

The following example constructs a small two-dimensional, single-coil, second-order encoding problem. Spatial coordinates are expressed in metres, `times` in seconds, and `fieldmap` in hertz. Each product of a temporal field coefficient and its corresponding spatial basis function contributes phase in cycles.

```julia
using HighOrderMRI

T = Float32
nX, nY, nZ = 16, 16, 1
nSam = 32

grid = Grid(nX, nY, nZ, T(1e-3), T(1e-3), T(5e-3))
times = collect(range(zero(T), T(2e-3); length=nSam))

# Rows are h0, x, y, z, and the five second-order terms.
kspha = zeros(T, 9, nSam)
kspha[2, :] .= range(T(-20), T(20); length=nSam)
kspha[3, :] .= range(T(-20), T(20); length=nSam)

fieldmap = zeros(T, nX, nY)
csm = ones(Complex{T}, nX, nY, 1)
mask = trues(nX, nY)

op = HighOrderOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    arrayType=Array,
    nBlock=4,
)

image = ones(Complex{T}, nX, nY)
signal = reshape(op * vec(image), nSam, 1)
```

The resulting signal uses samples as the first dimension and receive channels as the second. For dynamic data, the vector returned by `HighOrderLowRankOp` follows `vec(data)` for an array shaped `(nSam, nDyn, nCha)`.

The example is intentionally small and explicit. Before using measured or predicted field coefficients, review [Symbols and notation](/theory/symbols) and the [expanded encoding model](/theory/encoding-model) for phase sign, units, solid-harmonic order, NFFT mapping, and two- versus three-dimensional conventions.

## CUDA availability

CUDA operators should be constructed only after CUDA.jl reports a functional device:

```julia
using CUDA

CUDA.functional()
CUDA.devices()
```

Device identifiers supplied through `gpus` are zero based, for example `gpus=[0, 1]`. For multi-GPU execution, start Julia with sufficient default threads:

```bash
julia --threads=auto --project=.
```

The stage-specific voxel and channel decompositions are described in [Multi-GPU execution](/guide/multi-gpu).

## Build the documentation locally

The documentation site uses VitePress. From the repository root:

```bash
npm install --prefix docs
npm run docs:dev --prefix docs
```

For a production build:

```bash
npm run docs:build --prefix docs
```

The generated site is written to `docs/src/.vitepress/dist/`.

## Reading paths

**Method and signal model**

- [Architecture overview](/concepts-overview): relation between MRI measurements, dynamic fields, encoding operators, and iterative reconstruction.
- [Symbols and notation](/theory/symbols): indices, units, matrices, low-rank factors, and Julia array names.
- [Expanded encoding model](/theory/encoding-model): continuous signal model, discrete dynamic formulation, matrix factorization, explicit implementation, and low-rank separation.
- [Low-rank shared subspace](/theory/low-rank): per-dynamic rSVD, incremental shared spatial recompression, streaming coefficients, and two-stage approximation error.

**Reconstruction workflow**

- [Encoding operators](/guide/operators): constructor dimensions, numerical representations, and backend behavior.
- [Field preprocessing](/guide/field-preprocessing): GIRF prediction, interpolation, and field/data synchronization.
- [Coil compression](/guide/coil-compression): optional noise-whitened global SVD compression applied consistently to data and CSM.
- [Reconstruction workflow](/guide/reconstruction): data layout, field preparation, density weighting, solver configuration, reconstruction, and metrics.
- [Multi-GPU execution](/guide/multi-gpu): voxel- and channel-partitioned execution paths.

**Validation and reporting**

- [Scientific validation strategy](/guide/validation): evidence hierarchy, operator-level checks, low-rank baselines, and physical validation.
- [Reconstruction protocol](/guide/reconstruction-protocol): fixed conventions for reproducible numerical comparisons.
- [Performance & benchmarking](/guide/performance): timing, memory, accuracy, and baseline reporting requirements.

**Interface reference**

- [API reference](/api): public constructors, reconstruction functions, preprocessing helpers, metrics, and resource-cleanup interfaces.
- [Literature references](/references): numbered references cited throughout the scientific documentation, with DOI links to the corresponding publications.
