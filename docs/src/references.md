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
   2023.
   [doi:10.1002/mrm.29460](https://doi.org/10.1002/mrm.29460)
5. Eckart C, Young G. The approximation of one matrix by another of lower
   rank. *Psychometrika*. 1936;1:211–218.
   [doi:10.1007/BF02288367](https://doi.org/10.1007/BF02288367)

The local singular-vector separation and randomized range-finding stages build
on prior work. HighOrderMRI's implementation-specific extension is the
streaming recompression of dynamic-specific weighted spatial factors into an
adaptive shared basis, its coupling to a global-trajectory NFFT, and the
matrix-free multi-GPU execution strategy. The documentation does not claim
that low-rank higher-order encoding, randomized SVD, or shared subspaces are
new mathematical concepts.

## Internal design sources

The theory and implementation pages synthesize the following project Notion
pages, fetched on 2026-07-29:

- [HighOrderMRI.jl](https://app.notion.com/p/3ab5433ed4de80bf9f6afdb373afc4e3)
- [HighOrderMRI.jl Methods: High-order Encoding, LowRank, and multi-GPU
  implementation](https://app.notion.com/p/3a95433ed4de8160b038fe8c7e2bc455)
- [HighOrderLowRankOp: SVD separation and global shared spatial
  basis](https://app.notion.com/p/3ac5433ed4de8147aeb3cb6afc9c51b6)
- [Global shared spatial subspace: novelty, evidence strength, and publication
  strategy](https://app.notion.com/p/3ab5433ed4de814eaeaee15e4db75f80)
- [Phase 0 encoding-model convention test record,
  2026-07-24](https://app.notion.com/p/3a75433ed4de81aaa759dc22c61d063b)

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
