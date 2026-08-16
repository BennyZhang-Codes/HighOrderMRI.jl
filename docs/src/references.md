# References and source notes

## Scientific background

1. Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image
   reconstruction using singular-vector separation. *IEEE Transactions on
   Medical Imaging*. 2012;31:1396–1403.
   [doi:10.1109/TMI.2012.2190991](https://doi.org/10.1109/TMI.2012.2190991)
2. Lee NG, Ramasawmy R, Lim Y, Campbell-Washburn AE, Nayak KS. MaxGIRF:
   Image reconstruction incorporating concomitant field and gradient impulse
   response function effects. *Magnetic Resonance in Medicine*.
   2022;88:691–710.
   [doi:10.1002/mrm.29232](https://doi.org/10.1002/mrm.29232)
3. Halko N, Martinsson PG, Tropp JA. Finding structure with randomness:
   Probabilistic algorithms for constructing approximate matrix
   decompositions. *SIAM Review*. 2011;53:217–288.
   [doi:10.1137/090771806](https://doi.org/10.1137/090771806)
4. Dubovan PI, Baron CA. Model-based determination of the synchronization
   delay between MRI and trajectory data. *Magnetic Resonance in Medicine*.
   2023;89:721–728.
   [doi:10.1002/mrm.29460](https://doi.org/10.1002/mrm.29460)
5. Robison RK, Devaraj A, Pipe JG. Fast, simple gradient delay estimation for
   spiral MRI. *Magnetic Resonance in Medicine*. 2010;63:1683–1690.
   [doi:10.1002/mrm.22327](https://doi.org/10.1002/mrm.22327)
6. Rosenzweig S, Holme HCM, Uecker M. Simple auto-calibrated gradient delay
   estimation from few spokes using radial intersections (RING). *Magnetic
   Resonance in Medicine*. 2019;81:1898–1906.
   [doi:10.1002/mrm.27506](https://doi.org/10.1002/mrm.27506)
7. Ianni JD, Grissom WA. Trajectory Auto-Corrected image reconstruction.
   *Magnetic Resonance in Medicine*. 2016;76:757–768.
   [doi:10.1002/mrm.25916](https://doi.org/10.1002/mrm.25916)
8. Eckart C, Young G. The approximation of one matrix by another of lower
   rank. *Psychometrika*. 1936;1:211–218.
   [doi:10.1007/BF02288367](https://doi.org/10.1007/BF02288367)
9. Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. A software channel
   compression technique for faster reconstruction with many channels.
   *Magnetic Resonance Imaging*. 2008;26:133–141.
   [doi:10.1016/j.mri.2007.04.010](https://doi.org/10.1016/j.mri.2007.04.010)
10. Buehrer M, Pruessmann KP, Boesiger P, Kozerke S. Array compression for MRI
    with large coil arrays. *Magnetic Resonance in Medicine*.
    2007;57:1131–1139.
    [doi:10.1002/mrm.21237](https://doi.org/10.1002/mrm.21237)
11. Zhang T, Pauly JM, Vasanawala SS, Lustig M. Coil compression for
    accelerated imaging with Cartesian sampling. *Magnetic Resonance in
    Medicine*. 2013;69:571–582.
    [doi:10.1002/mrm.24267](https://doi.org/10.1002/mrm.24267)
12. Kellman P, McVeigh ER. Image reconstruction in SNR units: A general method
    for SNR measurement. *Magnetic Resonance in Medicine*.
    2005;54:1439–1447.
    [doi:10.1002/mrm.20713](https://doi.org/10.1002/mrm.20713)

### Coil-compression implementation boundary

HighOrderMRI implements the global software PCA/SVD family represented by
Huang et al. When a receive-noise covariance is provided, it first performs a
right Cholesky whitening and then retains leading right singular vectors. For
calibration matrix ``D``, noise-prescan covariance ``\Psi_{noise}``, and
bandwidth/dwell-time scale
``s=(T_{acq}/T_{noise})R_{BW}``, the effective acquisition covariance is
``\Psi_{acq}=\Psi_{noise}/s``. The returned combined matrix is ``C=WV_r``,
with ``W^H\Psi_{acq}W=I`` and therefore ``C^H\Psi_{acq}C=I``. The default
``s=1`` preserves the previous convention. The same ``C`` is applied to
measured data and CSM. Truncated
SVD supplies the Eckart--Young optimum only for the stated whitened Frobenius
objective and one global linear transform; this is not a blanket optimum for
parallel-imaging g-factor or every sampling geometry.

The bandwidth/dwell-time scale convention was cross-checked against the
[ISMRMRD Python coil utilities](https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/coils.py)
and
[mrpro prewhitening documentation](https://docs.mrpro.rocks/_autosummary/mrpro.algorithms.prewhiten_kspace.html).
HighOrderMRI uses the complex covariance convention ``E[n^Hn]`` and therefore
does not silently add the separate ``\sqrt{2}`` normalization used by some SNR
scaling pipelines.

Zhang et al.'s GCC is included as an important comparison, not as an
implementation citation. GCC exploits nonsubsampled Cartesian dimensions and
uses aligned spatially varying transforms. The current arbitrary-trajectory
operators require one global right-side coil transform, so silently labelling
the implementation as GCC would be incorrect. A future GCC implementation
would require an explicit hybrid-space/operator interface and separate
validation.

The legacy synchronization implementation was cross-checked against the
authors' public [MatMRI `findDelAuto`
source](https://gitlab.com/cfmm/matlab/matmri/-/blob/master/findDel/findDelAuto.m),
including its five-point derivative, sign-change acceleration update, and
final-delay return rule. The current Julia interface does not expose the
reference implementation's coarse-search initialization or a maximum outer
iteration count.

The local singular-vector separation and randomized range-finding stages build
on prior work. HighOrderMRI's implementation-specific extension is the
streaming recompression of dynamic-specific weighted spatial factors into an
adaptive shared basis, its coupling to a global-trajectory NFFT, and the
matrix-free multi-GPU execution strategy. The documentation does not claim
that low-rank higher-order encoding, randomized SVD, or shared subspaces are
new mathematical concepts.

## Internal design sources

The theory and implementation pages synthesize the following project Notion
pages, last consulted on 2026-08-03:

- [HighOrderMRI.jl](https://app.notion.com/p/3ab5433ed4de80bf9f6afdb373afc4e3)
- [HighOrderMRI.jl Methods: High-order Encoding, LowRank, and multi-GPU
  implementation](https://app.notion.com/p/3a95433ed4de8160b038fe8c7e2bc455)
- [HighOrderLowRankOp: SVD separation and global shared spatial
  basis](https://app.notion.com/p/3ac5433ed4de8147aeb3cb6afc9c51b6)
- [Global shared spatial subspace: novelty, evidence strength, and publication
  strategy](https://app.notion.com/p/3ab5433ed4de814eaeaee15e4db75f80)
- [Phase 0 encoding-model convention test record,
  2026-07-24](https://app.notion.com/p/3a75433ed4de81aaa759dc22c61d063b)
- [HighOrderMRI.jl MRM Toolbox manuscript experiment
  design](https://app.notion.com/p/3a95433ed4de815eb9a3e28cc2d09592)

These links may require access to the project's Notion workspace. They are
design records, not a replacement for source code, tests, or peer-reviewed
references.

Where a note and the current implementation differ, the current source and
regression tests take precedence. In particular, the explicit multi-GPU
operator currently shards masked voxels and reduces partial forward signals;
it does not shard forward samples as described in an earlier explanatory
note.

## Documentation and deployment model

The organization and deployment workflow were informed by
[KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl): separate
getting-started, tutorial/how-to, explanation, and API material; a dedicated
`docs/Project.toml`; source-backed API pages; and automated deployment from
GitHub Actions. HighOrderMRI uses Documenter.jl's native HTML renderer rather
than KomaMRI's larger DocumenterVitepress/Literate/Pluto stack because the
current package does not yet require generated notebooks or interactive
VitePress components.

Useful infrastructure references:

- [Documenter.jl guide](https://documenter.juliadocs.org/stable/man/guide/)
- [Hosting Documenter.jl output on
  GitHub](https://documenter.juliadocs.org/stable/man/hosting/)
- [KomaMRI.jl documentation](https://juliahealth.org/KomaMRI.jl/stable/)

## Validation boundary

The documented equations are checked against current array shapes, term
ranges, kernels, and convention tests. Published documentation should still
distinguish:

- internal consistency with the explicit operator;
- approximation error relative to that operator;
- independent numerical validation against a directly constructed matrix;
- physical validation against simulation or measured reference data.

Only the last two can close the strongest claims of absolute numerical and
physical accuracy.
