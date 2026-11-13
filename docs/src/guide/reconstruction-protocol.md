# Reconstruction conventions

Use the same conventions throughout data preparation, operator construction and reconstruction. This page is a practical checklist for comparing implementations; it does not prescribe a particular experiment or report dataset results. Detailed definitions are in [Symbols and notation](/theory/symbols) and the [encoding model](/theory/encoding-model).

## Encoding model

The package uses positive forward phase, in cycles before multiplication by $2\pi$, and conjugate phase and coil sensitivities in the adjoint. Masked-voxel normalization is $1/\sqrt{N_v}$. These are project conventions, not the only valid MRI Fourier convention.

LowRank approximates the residual encoding while retaining Fourier encoding on its active NFFT axes. In 2D, the remaining $k_z z$ term belongs to the residual; do not assume all three first-order terms are evaluated by a 2D NFFT.

Optional concomitant coefficients must be time-integrated and use the documented basis and units. `recon_terms` selects the original solid-harmonic rows; it does not independently switch concomitant contributions.

## Spatial grid

Use coordinates, field maps, masks and coil sensitivities in the same frame and physical units. The default axis $a$ of length $N_a$ and spacing $\Delta_a$ has voxel centres

$$
r_{a,n}=\left(n-\frac{N_a+1}{2}\right)\Delta_a,\qquad n=1,\ldots,N_a.
$$

Odd grids include zero; even grids use symmetric half-integer offsets. LowRank assumes its documented Cartesian lattice on NFFT axes. Editing a coordinate vector alone does not change the Fourier plan. Geometry export and resampling have additional [implementation limits](/reference/geometry).

## Solid-harmonic term order

Keep `kspha` rows in the [documented order](/theory/encoding-model#solid-harmonic-basis-order). A coefficient times its spatial polynomial must give phase in cycles. The first row is spatially constant; rows 2–4 are x, y and z. Fieldmap values in Hz require time values in seconds. No implicit gyromagnetic-ratio conversion should be assumed when converting gradient waveforms.

## Input shapes and vector ordering

| Operator/input | Layout |
| --- | --- |
| Explicit `kspha`, `times` | `(nTerm,nSam)`, `(nSam,)` |
| LowRank `kspha`, `times` | `(nTerm,nSam,nDyn)`, `(nSam,nDyn)` |
| LowRank data | `(nSam,nDyn,nCha)` |
| LowRank vector output | Samples vary fastest, then readouts, then coils. |

Here `nDyn` is the code's readout count, indexed by $i$ in the theory. The synchronization helpers accept coefficient arrays in their own documented orientation; transpose explicitly when crossing those interfaces. Match dimensional forms of the mask, fieldmap and CSM to the selected constructor rather than relying on arbitrary reshaping.

## Reconstruction-term selection

For nine coefficient rows, the three digits of `recon_terms` select zeroth, first and second order. Sixteen rows use four digits, adding third order. In the shared preparation path, disabled first-order measured coefficients are replaced by the supplied nominal trajectory; disabled zeroth and higher-order rows are zeroed. Check each constructor's actual default and nominal-trajectory requirements.

## NFFT convention

The physical Fourier coefficient is scaled by voxel spacing and negated to match the package's positive-phase model through the NFFT's negative-exponent convention. Even-axis centring corrections are part of the implementation. Use the [encoding derivation](/theory/encoding-model), not a second hand-maintained phase convention.

## Weights and regularization

`recon_HOOp` takes raw sample weights $W$; its data term is

$$
\lVert W(Ax-y)\rVert_2^2,
$$

with normal $A^H W^H W A$. `samplingDensity` returns square-root density weights. Do not square these again before passing them to the reconstruction wrapper. Direct optimized normal interfaces have their own documented squared-diagonal contract.

The unregularized normal is positive semidefinite; positive definiteness and a unique solution require additional rank/regularization conditions. Fix the regularizer and inner solver settings before comparing operators. Keep image-domain phase or intensity alignment separate from raw complex reconstruction error.

## Reproducibility and interpretation

Record the code revision, data preparation, units, coordinate transformations, mask, coil transform, weights, solver settings and LowRank approximation parameters. Apply a fitted coil transform consistently to data and CSM. A lower residual or a visually different image does not independently prove a more accurate physical model; use the [validation guide](/guide/validation).
