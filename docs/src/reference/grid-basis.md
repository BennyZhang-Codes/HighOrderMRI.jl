# Grid and solid-harmonic basis

`Grid` belongs to HighOrderMRI itself. It stores the physical coordinates used
by the encoding operators. Scanner geometry transforms are provided separately
by [MRIGeometry](/reference/geometry); do not assume that an encoding grid is
already in the scanner's coordinate frame.

For a centred grid with metres as the length unit, a complete example is:

```julia
using HighOrderMRI

grid = Grid(8, 8, 1, 1e-3, 1e-3, 5e-3)
basis = basisfunc_spha(grid.x, grid.y, grid.z, collect(1:9))
@assert size(basis) == (64, 9)
@assert all(basis[:, 1] .== 1)
metadata = SphericalHarmonics()
metadata.h4  # xy, with instantaneous field units T/m²
```

These polynomials are unnormalized real **solid** harmonics in Cartesian
coordinates. `basisfunc_spha` supports 36 terms; the MRI constructors accept
9 or 16 coefficient rows. The display metadata cover the first 16. See the
[basis ordering](/theory/encoding-model#solid-harmonic-basis-order),
[coordinate contract](/theory/encoding-model#coordinate-scope), and
[symbol definitions](/theory/symbols) before supplying measured coefficients.

## Grid API

<!--@include: ../../.generated/api/grid.md-->

## Basis evaluation API

<!--@include: ../../.generated/api/basisfunc-spha.md-->

## Metadata API

<!--@include: ../../.generated/api/spherical-harmonics.md-->

## Display arrays

`spha_basis`, `spha_basis_latex`, and `spha_color` are exported, mutable arrays
of 16 polynomial labels, LaTeX labels and colour strings. Their ordering
matches `SphericalHarmonics()`. They are display metadata, not numerical
configuration; changing them does not change encoding or add supported terms.
