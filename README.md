## Features

* Supports **2D** and **3D** image reconstruction with **high-order dynamic field changes**.
* Supports **up to 3rd order** spherical harmonic terms.
* Implements **parallel imaging** and **off-resonance correction** using the extended signal encoding operators:
  * `HighOrderOp`: array-based implementation, usable on  **CPU or GPU**.
  * `HighOrderOp_Kernel`: kernel-based implementation with **multiple GPUs** via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), faster and more memory-efficient than `HighOrderOp`.
  * `fastHighOrderOp`: SVD-based implementation...

* Integrates a **model-based synchronization delay estimation algorithm** ([Dubovan PI, Baron CA, 2023](https://doi.org/10.1002/mrm.29460)).

## Installation

This package relies on and **MRIReco.jl** (version 0.9.0).

```julia
using Pkg
Pkg.add(url="https://github.com/BennyZhang-Codes/HighOrderMRI.jl.git")
```

## Demo

For MRI reconstruction incorporating measured field dynamics, we first estimate the synchronization delay between the MRI data and the field measurements. The final reconstruction is then performed using the synchronized field dynamics.

This demo includes 2D single-shot spiral (7T, 1 mm in-plane resolution, ~29 ms readout) and 2D single-shot EPI (7T, 1 mm in-plane resolution, ~40 ms readout) imaging data, nominal kspace trajectory (Nominal) and measured field dynamics (using Dynamic Field Camera).

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

## Copyright & License Notice

This software is copyrighted by the Regents of the University of Minnesota and the Institute of Biophysics, Chinese Academy of Sciences. It can be freely used for educational and research purposes by non-profit institutions, US government agencies, and Chinese government agencies only.
Other organizations are allowed to use this software only for evaluation purposes, and any further uses will require prior approval. The software may not be sold or redistributed without prior approval.
One may make copies of the software for their use provided that the copies are not sold or distributed, and are used under the same terms and conditions.
As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied.
The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.
