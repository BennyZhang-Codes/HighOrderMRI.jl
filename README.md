# HighOrderMRI.jl

[![Julia 1.12](https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white)](https://julialang.org/downloads/)
[![CI](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/actions/workflows/ci.yml)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bennyzhang-codes.github.io/HighOrderMRI.jl/dev/)
[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bennyzhang-codes.github.io/HighOrderMRI.jl/stable/)

HighOrderMRI.jl is a Julia toolbox for MRI reconstruction with measured or
predicted dynamic higher-order fields. It builds on the higher-order
reconstruction module originally developed in
[DataStitching](https://github.com/BennyZhang-Codes/DataStitching), a toolbox for
dynamic field monitoring with NMR probes. It provides explicit encoding operators
for numerical reference and a matrix-free shared-subspace operator for large
2D and 3D reconstruction.

## Features

- Expanded signal encoding with real solid-harmonic terms through third order
  and optional concomitant-field phase terms.
- 2D and 3D parallel imaging with static off-resonance correction.
- `HighOrderOp`: array-based explicit evaluation on CPU or CUDA.
- `HighOrderKernelOp`: explicit CUDA kernels, GPU-resident cuBLAS backends,
  and optional phase caching.
- `HighOrderLowRankOp`: matrix-free residual-phase approximation with shared
  or per-readout spatial bases and NFFT-based Fourier encoding.
- Single- and multi-GPU CUDA execution for explicit and low-rank operators,
  including distributed low-rank factor construction.
- Global SVD coil compression with optional receive-noise prewhitening,
  using the same transform for data and coil-sensitivity maps.
- Weighted iterative reconstruction through `recon_HOOp`, with solver and
  regularization choices from RegularizedLeastSquares.jl.
- GIRF-based prediction of dynamic field coefficients.
- Model-based field/data synchronization for 2D acquisitions
  ([Dubovan and Baron, 2023](https://doi.org/10.1002/mrm.29460)).

## How to Cite

If you use this package, please cite the original
[DataStitching](https://github.com/BennyZhang-Codes/DataStitching) paper and the
high-order MRI workshop abstract below.

### Plain Text

Zhang J, Zhang Z, Zuo Z, Xue R, Zhuo Y, Cushing C, Bratch A, Auerbach E,
Grant A, An J, Ugurbil K, Wu X. Data stitching for dynamic field monitoring
with NMR probes. *Magnetic Resonance in Medicine*. 2026;95(4):1944–1958.
[doi:10.1002/mrm.70164](https://doi.org/10.1002/mrm.70164).

Related workshop abstract:

Zhang J, Jaffray A, Qu S, Cushing C, Xue R, Ugurbil K, Rauscher A, Zhang Z,
Wu X. High Impact Project: A comprehensive open-source framework for high-order
MRI. In: *Proceedings of the 2026 ISMRM Workshop on Data Sampling and Image
Reconstruction*, Sedona, AZ, United States of America.
[Program #302-01-055](https://echo.ismrm.org/abstracts/view/234b9199-c446-4181-ad46-684761459f67).

Related work:

Zhang J, Zhen T, Li S, Zuo Z, Cushing C, An J, Xue R, Zhuo Y, Ugurbil K,
Wu X, Zhang Z. Generalized data stitching for dynamic field monitoring with NMR
probes: improving acquisition efficiency and accuracy. In: *Proceedings of the
2026 ISMRM-ISMRT Annual Meeting and Exhibition*, Cape Town, South Africa.
[Program #605-02-004](https://echo.ismrm.org/abstracts/view/c6ce7146-215e-47e3-beed-12bd0bb2233b).

<details>
<summary>BibTeX</summary>

```bibtex
@article{Zhang2026DataStitching,
  author = {Zhang, Jinyuan and Zhang, Zihao and Zuo, Zhentao and Xue, Rong and Zhuo, Yan and Cushing, Cameron and Bratch, Alexander and Auerbach, Edward and Grant, Andrea and An, Jing and Ugurbil, Kamil and Wu, Xiaoping},
  title = {Data stitching for dynamic field monitoring with {NMR} probes},
  journal = {Magnetic Resonance in Medicine},
  year = {2026},
  month = apr,
  volume = {95},
  number = {4},
  pages = {1944--1958},
  doi = {10.1002/mrm.70164},
  url = {https://doi.org/10.1002/mrm.70164}
}

@inproceedings{Zhang2026HighOrderMRI,
  author = {Zhang, Jinyuan and Jaffray, Alexander and Qu, Shuxian and Cushing, Cameron and Xue, Rong and Ugurbil, Kamil and Rauscher, Alexander and Zhang, Zihao and Wu, Xiaoping},
  title = {High Impact Project: A comprehensive open-source framework for high-order {MRI}},
  booktitle = {2026 {ISMRM} Workshop on Data Sampling and Image Reconstruction},
  year = {2026},
  month = jan,
  address = {Sedona, AZ, United States of America},
  organization = {ISMRM},
  note = {Program Number: 302-01-055},
  url = {https://echo.ismrm.org/abstracts/view/234b9199-c446-4181-ad46-684761459f67}
}

@inproceedings{Zhang2026GeneralizedDataStitching,
  author = {Zhang, Jinyuan and Zhen, Tianxiang and Li, Sisi and Zuo, Zhentao and Cushing, Cameron and An, Jing and Xue, Rong and Zhuo, Yan and Ugurbil, Kamil and Wu, Xiaoping and Zhang, Zihao},
  title = {Generalized data stitching for dynamic field monitoring with {NMR} probes: improving acquisition efficiency and accuracy},
  booktitle = {2026 {ISMRM-ISMRT} Annual Meeting and Exhibition},
  year = {2026},
  month = may,
  address = {Cape Town, South Africa},
  organization = {ISMRM},
  note = {Program Number: 605-02-004},
  url = {https://echo.ismrm.org/abstracts/view/c6ce7146-215e-47e3-beed-12bd0bb2233b}
}
```

</details>

## Installation

HighOrderMRI.jl requires Julia 1.12 or later. CI tests both the minimum
supported Julia 1.12 release and the latest stable Julia 1.x release; the
documentation checks use Julia 1.12. The repository
currently carries `MRIGeometry` as a local subpackage, so clone the complete
repository:

```bash
git clone https://github.com/BennyZhang-Codes/HighOrderMRI.jl.git
cd HighOrderMRI.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then start Julia with the project and load the package:

```julia
using HighOrderMRI
```

CUDA execution additionally requires a functional
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installation. Start Julia with
`--threads=auto` when multiple GPUs participate.

## Demo

For MRI reconstruction incorporating measured field dynamics, we first estimate the synchronization delay between the MRI data and the field measurements. The final reconstruction is then performed using the synchronized field dynamics.

The demo includes 2D single-shot spiral (7 T, 1 mm in-plane resolution,
approximately 29 ms readout) and 2D single-shot EPI (7 T, 1 mm in-plane
resolution, approximately 40 ms readout) data. It compares a nominal
k-space trajectory with field dynamics measured using a Dynamic Field Camera.

<table>
  <tr>
    <th colspan="4" style="text-align:center">2D Spiral (left) & 2D EPI (right)</th>
  </tr>
  <tr>
    <td><img src="demo/result/7T_2D_Spiral_1p0_200_r4_Nominal.png" width="250"/></td>
    <td><img src="demo/result/7T_2D_Spiral_1p0_200_r4_Measured.png" width="250"/></td>
    <td><img src="demo/result/7T_2D_EPI_1p0_200_r4_Nominal.png" width="250"/></td>
    <td><img src="demo/result/7T_2D_EPI_1p0_200_r4_Measured.png" width="250"/></td>
  </tr>
</table>

## License

[LICENSE](LICENSE) · [MRIGeometry MIT license](MRIGeometry/LICENSE)
