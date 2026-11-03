# MRIGeometry.jl

`MRIGeometry.jl` provides acquisition geometry, coordinate transforms, gradient
rotation, grid construction, resampling, slab assembly and NIfTI export.
HighOrderMRI re-exports its public interfaces; its encoding `Grid` is a separate
type.

## Installation

This subpackage is included in the HighOrderMRI repository. From a local clone,
use it in a Julia environment with:

```julia
using Pkg
Pkg.develop(path="/path/to/HighOrderMRI.jl/MRIGeometry")
```

## Usage

```julia
using MRIGeometry

grid = gen_RPS_grid([0.2, 0.2, 0.2], [128, 128, 128])
```

FOV and translations are in metres; the returned array contains voxel-centre
coordinates. See the [geometry guide](https://bennyzhang-codes.github.io/HighOrderMRI.jl/dev/reference/geometry)
and source docstrings for shapes, coordinate conventions and current limits.
GE/PHILIPS/UIH mappings are placeholders. Gapped-slice merging, singleton-source
resampling and vector-of-geometries NIfTI export have documented limitations;
do not assume those paths preserve physical geometry without independent checks.

## License

MIT. See [LICENSE](LICENSE).
