# Scientific validation strategy

Validation of HighOrderMRI is separated into four levels that address different questions in a methods study:

1. **Implementation consistency:** do separate implementations of the same encoding equation agree numerically?
2. **Approximation accuracy:** how closely does `HighOrderLowRankOp` reproduce the corresponding explicit high-order operator?
3. **Independent numerical accuracy:** does the implemented signal model agree with a separately assembled numerical reference on a tractable problem?
4. **End-to-end physical validation:** does the complete reconstruction agree with simulation or measured reference data when field evolution, synchronization, geometry, and receiver effects are included?

Regression testing directly addresses the first level but does not, by itself, establish physical accuracy. Symbols and the reconstruction-weighting convention follow [Symbols and notation](/theory/symbols).

## Evidence hierarchy

| Evidence level | Primary comparison | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Implementation consistency | `HighOrderOp` vs `HighOrderKernelOp` | Agreement of explicit array-based and fused CUDA implementations | Absolute correctness of the shared physical model |
| Low-rank approximation | `HighOrderLowRankOp` vs a fixed explicit operator | Error introduced by local rSVD and shared-basis compression | Accuracy of the underlying field model |
| Independent numerical validation | Operator output vs a directly assembled encoding matrix on a small problem | Sign, units, basis order, centring, normalization, and data-layout consistency without reusing the same operator implementation | Scanner- or subject-specific physical accuracy |
| End-to-end physical validation | Simulation truth, measured trajectory/field reference, phantom, or in-vivo data | Behavior of the complete reconstruction chain under the stated acquisition conditions | Generalization beyond the tested acquisition and hardware conditions |

The [Reconstruction protocol](/guide/reconstruction-protocol) defines the conventions that must remain fixed across controlled comparisons.

## Independence of the reference

A comparison is only independent to the extent that the reference does not reuse the same implementation choices that are being tested. For example, a dense matrix assembled from the same phase equation can provide an independent check of operator construction, indexing, centring, and normalization if it does not call the production forward/adjoint kernels. It is not, however, an independent validation of the assumed field model itself.

For physical validation, the reference should therefore be external to the software path under test whenever feasible—for example simulation truth, independently measured trajectories/fields, or a separately calibrated phantom experiment. The degree of independence should be stated explicitly rather than inferred from agreement alone.

## Operator-level checks

Before image-domain comparisons, evaluate forward and adjoint errors on tractable data:

$$
\varepsilon_{\mathrm{fwd}}
=
\frac{\lVert A_{\mathrm{test}}x-A_{\mathrm{ref}}x\rVert_2}
{\lVert A_{\mathrm{ref}}x\rVert_2},
$$

$$
\varepsilon_{\mathrm{adj}}
=
\frac{\lVert A_{\mathrm{test}}^Hy-A_{\mathrm{ref}}^Hy\rVert_2}
{\lVert A_{\mathrm{ref}}^Hy\rVert_2},
$$

and, for iterative reconstruction, the weighted normal-operator error

$$
\varepsilon_{\mathrm{normal}}
=
\frac{
\lVert
A_{\mathrm{test}}^H W^H W A_{\mathrm{test}}x
-
A_{\mathrm{ref}}^H W^H W A_{\mathrm{ref}}x
\rVert_2
}
{
\lVert
A_{\mathrm{ref}}^H W^H W A_{\mathrm{ref}}x
\rVert_2
}.
$$

For the square-root density weights used by `recon_HOOp`, $W$ is diagonal. If the weights are real and non-negative, $W^H W=W^2$; the Hermitian form is retained because it remains valid for a general complex diagonal weighting operator.

Adjointness should also be evaluated using

$$
\varepsilon_{\mathrm{ip}}
=
\frac{|\langle Ax,y\rangle-\langle x,A^Hy\rangle|}
{\max(|\langle Ax,y\rangle|,\epsilon)}.
$$

These primary operator checks are performed on the unaligned complex data. Global phase or intensity alignment should not be applied before computing the reported operator error.

## Low-rank comparison baselines

Low-rank separation of higher-order encoding has been established previously, and the local randomized factorization used here builds on standard rSVD methodology. [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403.") [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 2011;53:217-288.")

To isolate the error introduced by the incremental shared spatial representation, compare the following baselines when computationally feasible:

1. the explicit high-order operator;
2. independent per-dynamic rank-$L$ factors without shared spatial recompression;
3. a direct global SVD/rSVD or equivalent post-hoc global subspace construction;
4. the incremental shared-basis implementation used by `HighOrderLowRankOp`.

A direct global construction is a useful approximation-quality reference because all dynamics can be revisited jointly. It may therefore achieve a smaller approximation error for a fixed final rank than a streaming incremental construction. The incremental implementation instead avoids retaining all dynamic-specific spatial factors and is compatible with matrix-free GPU and multi-GPU setup. The precise coefficient convention is described in [Low-rank shared subspace](/theory/low-rank).

## Reconstruction metrics

For simulations or comparisons against an explicit complex reference, raw complex NRMSE is used as the primary image-domain metric. Magnitude NRMSE and SSIM can be reported as complementary metrics, together with data-consistency residuals when relevant. The metric definitions are collected in [Reconstruction metrics](/reference/image-metrics).

For each low-rank experiment, record at least:

- `L_rank`, final shared rank, `shared_basis_tol`, and `shared_rank_max`;
- rSVD backend, finalization method, oversampling, distribution, and random seed;
- forward, adjoint, normal-operator, and inner-product adjointness errors;
- raw complex reconstruction error and secondary magnitude metrics;
- setup time, steady-state forward/adjoint/normal time, solver time, and end-to-end time;
- peak host/device memory and the hardware/software provenance defined by the reconstruction protocol.

Rank should be interpreted jointly with approximation error, runtime, and memory. A visually acceptable reconstructed image alone is insufficient for quantitative selection of the low-rank parameters.

## End-to-end physical validation

When measured dynamic fields are incorporated, document the complete preprocessing and reconstruction chain, including:

- field source and solid-harmonic calibration;
- field/data synchronization and delay convention;
- interpolation to ADC sampling times;
- spatial coordinate system and units;
- nominal, measured, or GIRF-predicted first-order trajectory convention;
- static off-resonance map and coil-sensitivity estimation;
- reconstruction mask, density weights, solver, and regularization.

Concurrent field monitoring and GIRF-based system characterization are established approaches for measuring or predicting the spatiotemporal encoding fields. [[6]](/references#ref-6 "Barmet C, De Zanche N, Pruessmann KP. Spatiotemporal magnetic field monitoring for MR. Magn Reson Med. 2008;60:187-197.") [[7]](/references#ref-7 "Vannesjo SJ, Haeberlin M, Kasper L, et al. Gradient system characterization by impulse response measurements with a dynamic field camera. Magn Reson Med. 2013;69:583-593.") Improvement in measured data demonstrates behavior under the tested acquisition conditions; claims of absolute encoding accuracy require an independent physical or simulation reference that can test the corresponding quantity.

## Interpretation and scope

Higher-order field-aware reconstruction, singular-vector separation, randomized SVD, and low-dimensional subspace representations have established methodological precedents. [[1]](/references#ref-1 "Wilm BJ, Barmet C, Pavan M, Pruessmann KP. Higher order reconstruction for MRI in the presence of spatiotemporal field perturbations. Magn Reson Med. 2011;65:1690-1701.") [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403.") [[3]](/references#ref-3 "Lee NG, Ramasawmy R, Lim Y, Campbell-Washburn AE, Nayak KS. MaxGIRF: Image reconstruction incorporating concomitant field and gradient impulse response function effects. Magn Reson Med. 2022;88:691-710.") [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 2011;53:217-288.")

Accordingly, the documentation distinguishes established methodological components from the specific software implementation examined here: a common explicit field-aware encoding model, matrix-free per-dynamic low-rank factorization, incremental shared spatial recompression, a global-trajectory NFFT representation, and GPU/multi-GPU execution. Quantitative claims should be tied to the corresponding validation level and to the acquisition and computational conditions under which they were measured.
