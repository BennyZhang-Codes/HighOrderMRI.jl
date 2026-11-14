# Performance and memory

Choose an execution strategy using the complete memory footprint and the cost of the operations you will repeat. Setup time, steady-state operator time and complete reconstruction time answer different questions.

## Multi-GPU execution

Different stages distribute different objects:

| Stage | Distribution | Practical consequence |
| --- | --- | --- |
| Legacy explicit kernels | Masked voxels | Partial signals must be combined. |
| Resident explicit operators | Samples | Each worker evaluates its sample block. |
| LowRank voxel-distributed rSVD | Masked voxels | Distributes setup work and spatial factors. |
| LowRank channel normal | Receive channels | Workers hold their assigned coil observations. |
| LowRank independent-readout normal | Readouts | Requires the corresponding independent factor representation. |
| LowRank rank execution | Shared-basis columns | Workers contribute to the same samples; their forward results are combined before adjoint evaluation. |

Setup distribution and normal-operator distribution are separate constructor choices. See [Multi-GPU execution](/guide/multi-gpu) for restrictions and communication. Rank sharding is not a guarantee that a problem fits: data, image vectors, NFFT plans and communication buffers still contribute to memory.

## Explicit phase storage and fair timing

For `ComplexF32`, a complete phase matrix alone requires `8 * nSam * count(mask)` bytes. Coil maps, model arrays, data, solver vectors and library workspaces are additional. The resident `:blocked` backend generates phase blocks; `:cached` and `:cached_full` retain the complete phase matrix. The default `:legacy` uses its existing direct path.

`phase_cache_bytes` budgets a prefix of complex phase blocks for the blocked backend, not the total device allocation. Changing block size also changes how many complete blocks fit. Include construction and cache generation when measuring a one-use reconstruction; report reuse separately.

### Relation to the non-Fourier SENSE preprint

Full and split phase storage are related to the strategies described by Bianchi and Pruessmann [24](/references#ref-24). This package adapts those encoding operations with its own phase, weighting and execution conventions. It does not reproduce every part of the preprint's reconstruction workflow. Low-rank residual-phase approximation is a separate strategy.

## Low-rank acceleration

For the shared representation, one unbatched forward or adjoint uses one transform for each shared component and receive channel: $R N_c$ transform applications. A normal contains both directions. With sample batching, the number of transform calls also depends on the number of batches; the same formula must not be used as a batched call count.

Independent per-readout factors instead use $N_{\mathrm{ro}} L N_c$ applications per direction when the retained local rank is $L$. Their full spatial bank has $N_v N_{\mathrm{ro}} L$ complex entries. Host streaming moves that bank to host memory; regeneration trades storage for additional computation. These counts describe work and storage, not measured speedups.

The [LowRank reference](/reference/highorderlowrankop) distinguishes setup workspace, retained factors, NFFT plans, sample batches and solver storage. Its documented backend restrictions apply when choosing a batch size.

## Backend and precision tradeoffs

Record element types, NFFT backend, rSVD finalization and effective CUDA math mode. Gram finalization squares the condition number of the projected matrix; an FP64 eigendecomposition cannot recover directions already lost in lower-precision Gram accumulation. A different finalization method may change workspace and communication as well as numerical error. See [low-rank principles](/theory/low-rank).

## Setup cost and amortization

Measure construction, forward, adjoint, normal, solver and complete reconstruction separately. When an operator is reused, an approximate break-even application count is

$$
N_{\mathrm{break}}\approx\frac{T_{\mathrm{setup}}}{T_{\mathrm{ref}}-T_{\mathrm{lr}}},
$$

provided the denominator is positive and the setup cost is the additional cost relative to the chosen reference. Define the application being counted and include necessary transfers in both timings.

## Accuracy-matched timing

Keep data, geometry, mask, coil maps, weights, regularization and stopping conditions fixed. Compare approximation error together with time and peak memory. Different error levels are different operating points, rather than a single speedup result.

Warm up compilation separately and synchronize every participating GPU around elapsed-time measurements. Repeat complete trials and report their spread. Do not add overlapping worker, transfer and device timings as though they were sequential wall time. Distinguish live allocations from memory reserved by the allocator.

For a reproducible comparison record the code revision, package environment, Julia/Python/CUDA versions, GPU identities, dimensions, precision, random seed and execution parameters. The [validation guide](/guide/validation) explains which conclusions numerical comparisons support. Synthetic checks and successful website builds do not establish full-brain performance.
