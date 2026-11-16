# Explicit: Array

`HighOrderExplicitOp(...; implementation=:array)` selects Array evaluation.
This is the default implementation. `arrayType` chooses CPU or CUDA arrays.

A CPU explicit operator can be constructed with:

```julia
op = HighOrderExplicitOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    k_nominal=kspha[2:4, :],
    nBlock=32,
    arrayType=Array,
)
```

Array-based CUDA execution is selected with:

```julia
using CUDA

op = HighOrderExplicitOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
)
```

`nBlock` divides the samples into explicit phase blocks. Increasing the number of blocks reduces peak temporary memory at the cost of additional launches and allocations.

Supplying `kspha_dt` changes the forward action to the time-derivative product $Bx$ used internally by `FindDelay`. The adjoint remains the ordinary encoding adjoint. An operator constructed with `kspha_dt` should therefore not be interpreted as a general derivative linear operator, and do not use that forward/adjoint pair in adjoint-based solves or normal products.

The array-based CUDA helpers can downcast Float64 input vectors internally.
For a Float64 explicit reference, use `arrayType=Array`.


The family wrapper forwards `nBlock`, `kspha_dt`, the field model and all
other options unchanged. Array execution does not retain a phase cache or
reuse a generated block across a separately requested forward and adjoint.
For GPU-resident CG with explicit phase reuse, see [GEMM and phase-reuse](/guide/explicit-gemm).

See the [array API](/reference/explicit-array) for its original defaults and limitations.
