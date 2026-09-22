# Extend `write_mesh_files_only` beyond `SphericalShellInternalBoundary`

## Summary

#592 added `write_mesh_files_only=True`, which returns the `.msh.h5` path and stops before
constructing a `Mesh`. We measured it and it does what the commit message claims. It is
currently implemented on **1 of the 20 public mesh constructors**, and not on either of the two
most likely to be generating meshes in bulk — `UnstructuredSimplexBox` and `SphericalShell`.

Could it be extended? The mechanism is already merged and exported; what is missing is the
early return in the other generators.

## Measurement

`SphericalShellInternalBoundary`, `cellSize=0.05` (136,947 cells), serial:

| | wall time | returns |
|---|---|---|
| `write_mesh_files_only=False` | 14.73 s | `Mesh` |
| `write_mesh_files_only=True` | **5.15 s** | `str` (h5 path) |

**2.86×.** We loaded both resulting `.msh.h5` files back and compared every DM label and
stratum size — identical (`All_Boundaries`, `UW_Boundaries`, the face labels, `Elements`,
`celltype`, `depth`, cell count). The files differ byte-wise, which we assume is HDF5-internal
and not meaningful.

This is at 137k cells; the commit message motivates it at "millions", so the gap should widen.

## Current coverage

`write_mesh_files_only` occurs at four lines in `src/underworld3/meshing/`, all inside
`SphericalShellInternalBoundary`. `_gmsh_to_h5` is imported and called in exactly one place
(`spherical.py`, in that same branch).

The other 19 constructors — including `UnstructuredSimplexBox`, `SphericalShell`,
`StructuredQuadBox`, `Annulus`, `CubedSphere`, `SegmentedSphericalShell` — build a full `Mesh`
even when the caller only wants reusable files on disk.

## Why this matters for pre-generation

The pattern the flag exists for is exactly what a bulk mesh-generation step does: produce
reusable mesh files for a set of resolutions, once, so that later jobs load them. Our own
generator does this for six models across a resolution sweep, and for every resolution it
constructs a `Mesh` and immediately discards it.

There is a second reason to want the files rather than the `Mesh`. For large resolutions the
full construction can fail where the file writing succeeds, so a generator ends up wrapping the
call in `try/except` and checking whether the file appeared — which works only because
`_gmsh_to_h5` happens to run before the part that fails. `write_mesh_files_only=True` makes that
intent explicit instead of incidental.

## The change

The template is the existing branch in `SphericalShellInternalBoundary`, immediately after the
Gmsh write:

```python
    if write_mesh_files_only:
        return gmsh2h5(
            uw_filename,
            markVertices=True,
            useRegions=True,
            useMultipleTags=True,
        )
```

Per constructor this is a keyword argument, a docstring entry, and that return — with the
`markVertices` / `useRegions` / `useMultipleTags` values matching whatever the constructor
already passes to `_from_gmsh` on its normal path, since those affect the conversion.

The return type becoming `Mesh | str` is the one API wrinkle, and #592 has already set the
precedent for how that is documented.

## Priority

`UnstructuredSimplexBox` and `SphericalShell` would cover the common cases. The rest could
follow the same pattern if wanted.

## Offer

Happy to send this as a PR for those two rather than leave it as a request — the change is
mechanical and we have the harness set up to report before/after generation times per
constructor. Say which you'd prefer.

We have not measured the saving for `UnstructuredSimplexBox` directly, because there is no way
to skip the `Mesh` build there yet — that being the point of the request.
