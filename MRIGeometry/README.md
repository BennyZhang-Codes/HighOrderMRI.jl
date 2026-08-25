# MRIGeometry.jl

`MRIGeometry.jl` provides geometry types and utilities for MRI acquisition and
reconstruction workflows, including coordinate transforms, gradient conversion,
grid construction, resampling, multi-slab assembly, and NIfTI export.

## Installation

After registration, install the package with:

```julia
using Pkg
Pkg.add("MRIGeometry")
```

## Usage

```julia
using MRIGeometry

grid = gen_RPS_grid([0.2, 0.2], [128, 128])
```

See the exported API in the source for the currently supported geometry and
conversion utilities.

## License

MIT. See [LICENSE](LICENSE).
