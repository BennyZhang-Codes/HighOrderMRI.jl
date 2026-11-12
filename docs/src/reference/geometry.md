# Acquisition geometry

`MRIGeometry` supplies acquisition geometry, coordinate transforms, gradient
rotation, resampling, slab assembly and NIfTI export. `HighOrderMRI` re-exports
the interfaces listed here. The root package's encoding [`Grid`](/reference/grid-basis)
is a separate type; an arbitrary geometry transformation does not change an
existing NFFT operator's Fourier grid.

## Coordinate and unit conventions

RPS means local read/phase/slice coordinates. PCS is the patient LPS frame;
DCS is the device frame selected by the vendor table. FOV and translations are
stored in metres. `T_PCS` is the slice/volume centre. Voxel spacing is `FOV/N`,
and index $j=1,\ldots,N$ has local centre
$(j-1-(N-1)/2)\,\mathrm{FOV}/N$. This differs from the position of the first
voxel in a DICOM image. For physical image-coordinate definitions, see
[DICOM PS3.3 C.7.6.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).

The four coordinate-transform functions apply rotation and translation to
three-component coordinates on the last array axis. Their inverse mappings
use the matrix transpose and therefore assume orthogonal rotations. Grid
coordinates should be floating point; integer inputs can lose physical units.

::: warning Vendor and mutable-state limits
GE, PHILIPS and UIH device/polarity entries are placeholders, not validated
scanner support. Confirm the chosen device convention from acquisition
metadata and an independent reference, including for Siemens. A transform
followed by its algebraic inverse cannot establish physical correctness.

`Geometry` stores both original and derived fields. Direct field mutation does
not recalculate dependent transforms; dictionary construction also accepts
stored derived values. The current constructors do not fully validate matrix
orthogonality, positive sizes or consistency. Keep the complete state coherent.
:::

## Construct a regular geometry

This example checks patient-coordinate algebra on a small regular volume. It
does not validate the device mapping for a scanner.

```julia
using HighOrderMRI
using LinearAlgebra

T = Float64
geo = Geometry("SIEMENS", 3, T[0.04, 0.04, 0.02], Int[4, 4, 2],
    "HFS", T[0.01, -0.02, 0.03], Matrix{T}(I, 3, 3), 1)
rps = gen_RPS_grid(geo)
pcs = RPS2PCS(geo, rps)
@assert size(pcs) == (4, 4, 2, 3)
@assert PCS2RPS(geo, pcs) ≈ rps
```

## Geometry and grid APIs {#geometry-and-grid-apis}

- **`MRIGeometry`** is the submodule/package that owns these interfaces. Loading
  `HighOrderMRI` makes its exported names available; it is not a constructor.
- **`Geometry(SystemVendor, Dimension, FOV, MatrixSize, PatientPosition, T_PCS,
  R_RPS_PCS, Idx_Slice)`** creates mutable `Geometry{T,D}`. The three FOV/centre
  components and 3×3 rotation must share floating-point T; dimension, three
  matrix counts and one-based slice/slab label share integer D. The vendor
  name is uppercased, and the constructor computes DCS fields from its tables.
  `Geometry(d::AbstractDict; T=Float64,D=Int)` instead reads all stored fields
  named by the struct, including derived transforms. It does not promote
  mixed precision automatically or infer missing fields.
- **`get_geometry(raw; T=Float64,D=Int64)`** takes `RawAcquisitionData` and
  returns a vector of geometries. It requires `systemVendor`, `reconFOV`,
  `reconSize` and `patientPosition`, plus profile centres/directions. Header
  FOV and positions are converted from mm to m, and slice labels from zero
  to one based. Profiles are grouped by coordinates rounded to four decimal
  places in the original metadata units, preserving first occurrence. It
  does not return a profile-index map or guarantee physical sorting.
- **`gen_RPS_grid(fov::Vector{T}, matrix_size::Vector{D})`** and
  **`gen_RPS_grid(geo)`** return a dense `(nR,nP,nS,3)` array of T using the
  voxel-centre formula above. Both vectors need three positive components;
  only lengths are explicitly checked.
- **`RPS2PCS(geo, coordinates)`** applies `R_RPS_PCS*x+T_PCS` on each column
  coordinate. **`PCS2RPS(geo, coordinates)`** applies its transpose inverse.
  **`RPS2DCS(geo, coordinates)`** and **`DCS2RPS(geo, coordinates)`** use the
  corresponding DCS fields. All preserve the coordinate-array shape and
  convert geometry parameters to its element type; the last axis must have
  length three. PCS/DCS values are absolute positions; RPS values are local.

## Gradient coordinate conversion {#gradient-coordinate-conversion}

**`grad_Nominal2DCS(G, geo; dim=N==2 ? 1 : N,
R_polarity=geo.R_Nominal_RPS)`** applies
`geo.R_RPS_DCS*R_polarity` along an axis of length three. **`grad_DCS2Nominal`**
has the same keyword defaults and applies the transposes in reverse order.
Both preserve shape and units; the reverse is a true inverse only for
orthogonal matrices. They neither apply a GIRF nor turn nominal gradients into
measured waveforms. Explicit polarity matrices still need a compatible
physical convention.

Use [`plt_grad`](/reference/plotting#api-plt-grad) to inspect nominal values in
these frames. Its Geometry overload is a rotation display. The root package
also extends the same function with a `GIRFModel` overload for response prediction.

## Resampling {#resampling}

**`resample(source, source_geometry, target_geometry;
oversample=[1,1,1], space=:PCS, is_mask=false)`** accepts a 3D source matching
its geometry and returns the target spatial shape. `:DCS` is also available.
Continuous data use linear interpolation and zero extrapolation. Oversampling
averages equally weighted interior points of each target voxel; the default
uses only its centre. It is not exact volume-overlap integration.

`is_mask=true` uses nearest interpolation, disables oversampling and returns
integer labels. Otherwise floating/complex data retain their type, while
integer data are temporarily Float32 and rounded back. Counts must be three
positive integers; zero/negative counts are not safely checked.

A 4D source is processed channel by channel. Its extra keyword
`normalize_csm=false` can enable local RSS division with a 1e-8 denominator
offset; use it for floating-point sensitivity maps, not integer labels.
This changes the sensitivity normalization convention and is not a proof of
energy preservation. Passing a vector of target geometries returns a vector
of independently resampled arrays in that order, without assembling a volume.

::: warning Single-slice source arrays
The current interpolation setup can reject singleton spatial axes, including
an `(nX,nY,1)` source. A later z-index shortcut does not repair that setup or
define physical support outside the plane. Do not treat this as validated
single-slice-to-volume resampling.
:::

## Slice and slab assembly {#slice-and-slab-assembly}

**`merge(geometries)`** returns one Geometry, not reordered image data. It sorts
centres along the first normal, uses their midpoint, and makes one output
slice per entry. Its current FOV formula is centre-span plus the first slice
thickness. Gapped slices therefore acquire incorrect implied centres. Restrict
this shortcut to aligned single-plane slices with uniform spacing equal to
their thickness. Nonuniform stacks and multi-plane slabs require explicit
geometry handling; even a single input slab is reduced to a third count of one.

**`merge_multislabs(slabs, geometries, overlap_slices; dim_z=3)`** accepts a
vector of complex arrays. It physically sorts slab centres, removes half of
each even overlap at adjoining edges, concatenates, and returns
`(merged_data,global_geometry)`. **`merge_motsa(...; dim_z=3,method=:mip)`** also
accepts real arrays: `:discard` crops, `:blend` linearly cross-fades, and `:mip`
keeps the larger-magnitude sample with its original value/phase. It is not a
separate 2D projection.

For both mergers, require nonempty matched data/geometries, aligned orientations,
equal in-plane shapes and slice spacing, and a nonnegative even overlap smaller
than each slab. `merge_motsa` additionally assumes equal slab sizes. Only some
conditions are asserted; unknown method symbols are not safely rejected.
Returned geometry is inferred from the first slab spacing and outer bounds.
These functions do not register or resample heterogeneous volumes.

## NIfTI export {#nifti-export}

**`export_nifti(image, geometry, outpath::String, name::String)`** writes
`name.nii.gz` for a real 3D/4D array. Create the folder first, and ensure the
first three array dimensions equal `geometry.MatrixSize`. These preconditions
are not all checked. The affine uses voxel centres, converts PCS LPS to NIfTI
RAS and metres to mm, and is returned as a Float32 3×4 matrix. Fourth-axis
values are retained, but time-step metadata is zero. See the
[NIfTI header field definitions](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields.html).

Complex input writes separate `name_mag.nii.gz` and `name_pha.nii.gz` files;
phase is the principal angle in radians. The return is the magnitude affine.

::: warning Vector-of-geometries overload
The overload accepting a vector sorts by `Idx_Slice` and uses those indices
directly on the third image dimension. A geometry representing a multi-plane
slab can silently cause planes to be dropped. Descending physical centres can
produce an affine with the wrong slice direction. Nonuniform spacing and
in-plane offsets are also not validated.

Prefer a correctly ordered complete image and an explicitly validated single
regular Geometry. The vector overload is restricted in practice to one plane
per geometry, with Idx_Slice order matching increasing positions on a common
normal, uniform spacing and matching in-plane geometry. Verify exported
coordinates independently; these implementation defects remain unfixed.
:::

## Geometry plotting {#geometry-plotting}

**`plt_geometry(geo_or_vector; space=:DCS, ...)`** returns a PyPlot figure of
FOV boxes. `:PCS` and `:RPS`/`:rps` are also accepted; RPS displays local boxes,
not relative patient positions. The single-geometry form wraps a vector.
Width/height are centimetres, displayed positions are millimetres, and optional
logical spans can be drawn with `plot_RPS_vectors=true`.

**`plt_grid(grid; step=4, space=:DCS, ...)`** subsamples a dense coordinate
array and colours logical indices. `space` selects labels, not a transform.
Use a positive step and already transformed metre coordinates. `:PCS` and
`:DCS` label paths work; the current `:RPS` branch throws, and `:XYZ`/`:xyz`
lacks an implemented labelling branch. Width/height are centimetres. Neither
plot is an independent validation of physical metadata.

Complete source signatures remain in the module docstrings; this page states
the supported contracts and known limits for all geometry re-exports. For
encoding coordinate restrictions, return to [the encoding model](/theory/encoding-model#coordinate-scope).
