# `recon_HOOp`

Solve a weighted inverse problem using an already constructed operator.
The [reconstruction workflow](/guide/reconstruction) explains data preparation,
weights, regularization and GPU placement. A complete CPU example follows;
its synthetic full Cartesian model is a usage check, not physical validation.

## Example

```julia
using HighOrderMRI, RegularizedLeastSquares, LinearAlgebra

T = Float64
nX, nY = 8, 8
grid = Grid(nX, nY, 1, T(1e-3), T(1e-3), T(5e-3))
nSam = nX * nY
kspha = zeros(T, 9, nSam)
kspha[2, :] .= repeat(T.(-4:3) ./ (nX * grid.Δx), nY)
kspha[3, :] .= repeat(T.(-4:3) ./ (nY * grid.Δy); inner=nX)
times = zeros(T, nSam)
op = HighOrderOp(grid, kspha, times; arrayType=Array, nBlock=8)
reference = zeros(Complex{T}, grid.matrixSize)
reference[3:6, 3:6, 1] .= 1
samples = reshape(op * vec(reference), nSam, 1)
weights = ones(Complex{T}, nSam)  # raw sample weights
rec_params = Dict{Symbol,Any}(
    :reconSize => grid.matrixSize,
    :solver => CGNR,
    :reg => L2Regularization(T(1e-3)),
    :iterations => 10,
)
image = recon_HOOp(op, samples, weights, rec_params)
@assert size(image) == grid.matrixSize
@assert all(isfinite, image)
```

## Solver and storage choices

The raw weights are applied to both data and encoding, giving the data normal
$A^H W^H W A$. That matrix is Hermitian positive semidefinite. Standard CG
convergence assumptions require positive definiteness on the solved space;
mask nullspaces and undersampling can make the data normal singular. Choose
regularization and a solver appropriate to the problem, and check convergence.
An adjoint backprojection alone does not solve this objective.

For CUDA LowRank and resident explicit execution, place the data matrix and
raw weight vector on `first(gpus)` using the operator's complex precision.
The wrapper returns a CPU image. The legacy explicit kernel path instead uses
host solver vectors. See [resource ownership](/reference/resources).


## API

<!--@include: ../../.generated/api/recon-hoop.md-->
