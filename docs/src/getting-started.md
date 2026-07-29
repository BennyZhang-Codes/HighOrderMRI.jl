# Getting started

## Requirements

- Julia 1.10 or later (CI tests Julia 1.10 LTS and the latest stable Julia 1.x
  release);
- an NVIDIA GPU and a functional CUDA.jl installation for CUDA paths;
- enough host and device memory for the selected operator, mask, and coil
  count.

The package currently includes `MRIGeometry` as a package inside the same
repository. The most reliable installation path is therefore a local clone.

```bash
git clone https://github.com/BennyZhang-Codes/HighOrderMRI.jl.git
cd HighOrderMRI.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Start Julia in the project:

```bash
julia --project=.
```

Then load the package:

```julia
using HighOrderMRI
```

## First CPU operator

The following example constructs a small 2D, single-coil, second-order model.
Coordinates use metres, `times` uses seconds, `fieldmap` uses hertz, and every
entry of `kspha .* basis` contributes phase in cycles.

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

The result has samples in the first dimension and channels in the second.
For dynamic data, [`HighOrderLowRankOp`](@ref) returns a vector compatible
with `vec(data)` for `data` shaped `(nSam, nDyn, nCha)`.

## Check CUDA

CUDA support is loaded by the package, but a CUDA operator should only be
created when CUDA.jl reports a functional device:

```julia
using CUDA

CUDA.functional()
CUDA.devices()
```

Use zero-based device identifiers in `gpus`, for example `gpus=[0, 1]`. Start
Julia with enough default threads when several GPUs participate:

```bash
julia --threads=auto --project=.
```

See [Multi-GPU execution](guide/multi-gpu.md) before enabling distributed rSVD
or the channel-distributed normal operator.

## Build the documentation locally

From the repository root:

```bash
julia --project=docs -e '
using Pkg
Pkg.develop([
    PackageSpec(path=pwd()),
    PackageSpec(path=joinpath(pwd(), "MRIGeometry")),
])
Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`, or serve `docs/build` with a local HTTP server.
Generated files are ignored by Git.

## Next steps

- [Expanded encoding model](theory/encoding-model.md): units, signs, basis
  order, 2D/3D term assignment, and layout.
- [Low-rank shared subspace](theory/low-rank.md): rSVD, Small-Gram
  finalization, shared rank, and the two-stage error budget.
- [Choose an operator](guide/operators.md): constructor shapes and backend
  trade-offs.
