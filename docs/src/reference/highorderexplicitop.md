# `HighOrderExplicitOp`

`HighOrderExplicitOp` is the Explicit family type and constructor.
`implementation=:array` selects Array evaluation; `implementation=:kernel`
selects CUDA direct kernels or GEMM and phase-reuse. Concrete implementation
types are internal. See the [implementation guide](/guide/explicit),
[Array options](/reference/explicit-array) and [CUDA options](/reference/explicit-cuda).

## Complete CPU example

```julia
using HighOrderMRI, LinearAlgebra
using LinearOperatorCollection

grid = Grid(4,3,1,1f-3,1f-3,5f-3)
kspha = zeros(Float32,9,7)
kspha[2,:] .= range(-100f0,100f0;length=7)
times = collect(range(0f0,2f-3;length=7))
fieldmap = reshape(collect(range(-20f0,20f0;length=12)),4,3)
op = HighOrderExplicitOp(grid,kspha,times;
    implementation=:array, arrayType=Array, fieldmap, nBlock=3)
x = ones(ComplexF32,12)
signal = op*x
D = Float32.(range(0.2,1;length=7))
normal = normalOperator(op,D)*x
@assert size(op) == (7,12)
@assert all(isfinite,normal)
@assert norm(normal-op'*(D.*signal))/norm(normal) < 1e-5
```

`normalOperator` receives the already squared diagonal. `recon_HOOp` receives
raw square-root weights instead. For CUDA paths, explicitly close the
operator after its final use. Array `close` is a no-op; its model arrays stay
referenced until the operator is dropped.

## Constructor API

<!--@include: ../../.generated/api/highorderexplicitop.md-->
