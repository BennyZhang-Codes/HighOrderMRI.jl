# `HighOrderOp`

Use the explicit CPU path to inspect a small encoding problem before choosing
low-rank approximation or GPU storage. It evaluates the physical phase directly;
its ordinary adjoint is a backprojection, not an inverse. See the
[encoding model](/theory/encoding-model) for units, normalization and mask support.

## Example

```julia
using HighOrderMRI

grid = Grid(8, 8, 1, 1e-3, 1e-3, 5e-3)
kspha = zeros(Float64, 9, 32)
kspha[2, :] .= range(-200.0, 200.0; length=32)
times = collect(range(0.0, 3e-3; length=32))
op = HighOrderOp(grid, kspha, times; arrayType=Array, nBlock=4)
x = ones(ComplexF64, prod(grid.matrixSize))
y = op * x
backprojection = reshape(adjoint(op) * y, grid.matrixSize)
@assert size(op) == (32, 64)
```

## Current boundaries

When `kspha_dt` is supplied, only the forward action becomes a derivative
product. Its adjoint remains the ordinary encoding adjoint; do not use this
configuration in an adjoint-based solve or a normal product. The CUDA array
helpers can downcast Float64 input vectors internally; use the CPU path for
an independently checked Float64 reference. These limitations are not changed
by the documentation migration.


## API

<!--@include: ../../.generated/api/highorderop.md-->
