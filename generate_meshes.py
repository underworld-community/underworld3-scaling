"""
Pre-generate mesh files for all scaling models and resolutions.

Run this once before submitting a scaling campaign to populate the mesh cache.
All scaling jobs will then skip Gmsh and load the cached .msh.h5 in place. A .msh that
has lost its .h5 is converted, not regenerated, so the campaign's mesh is preserved.

Mesh types by model:
  Box  (UnstructuredSimplexBox): poisson-scaling, ns-scaling, vep-scaling,
                                  advdiff-scaling, checkpoint-scaling
  Spherical (SphericalShell):    stokes-scaling

Cache filenames (qdeg not encoded — the .msh file stores geometry only):
  Box:       UnstructuredSimplexBox_res<R>.msh
  Spherical: SphericalShell_ri<ri>_ro<ro>_res<R>.msh

Usage (matches params.sh conventions):
    python generate_meshes.py \\
        --models "poisson-scaling stokes-scaling" \\
        --base 24 --jobs "1 2 3 4 5 7 10 17" \\
        --cache /scratch/el06/jg0883/mesh_cache

On Gadi — submit as a serial PBS job (hugemem queue for stokes at large N):
    qsub generate_meshes_pbs.sh
"""

import os
import sys
import argparse
from enum import Enum

import underworld3 as uw
from underworld3.coordinates import CoordinateSystemType

# --------------------------------------------------------------------------- #
# Mesh spec registry — one entry per model                                     #
# --------------------------------------------------------------------------- #
MESH_SPECS = {
    "poisson-scaling": {
        "type": "box",
        "qdeg": 3,
    },
    "ns-scaling": {
        "type": "box",
        "qdeg": 2,
    },
    "vep-scaling": {
        "type": "box",
        "qdeg": 2,
    },
    "advdiff-scaling": {
        "type": "box",
        "qdeg": 3,
    },
    "checkpoint-scaling": {
        "type": "box",
        "qdeg": 2,
    },
    "stokes-scaling": {
        "type": "spherical",
        "qdeg": 2,
        "r_i": 1.22,
        "r_o": 2.22,
    },
}


def cache_filename_box(res):
    return f"UnstructuredSimplexBox_res{res}.msh"


def cache_filename_spherical(r_i, r_o, res):
    return f"SphericalShell_ri{r_i}_ro{r_o}_res{res}.msh"


def generate_box(res, qdeg, cache_file):
    uw.pprint(f"  Generating box mesh res={res} → {cache_file}")
    try:
        uw.meshing.UnstructuredSimplexBox(
            minCoords=(0., 0., 0.),
            maxCoords=(1., 1., 1.),
            cellSize=1.0 / res,
            qdegree=qdeg,
            filename=cache_file,
        )
    except Exception:
        if os.path.exists(cache_file):
            uw.pprint(f"  .msh written successfully; DMPlex load skipped for large mesh.")
        else:
            raise


def generate_spherical(r_i, r_o, res, qdeg, cache_file):
    uw.pprint(f"  Generating spherical mesh ri={r_i} ro={r_o} res={res} → {cache_file}")
    try:
        uw.meshing.SphericalShell(
            radiusInner=r_i,
            radiusOuter=r_o,
            cellSize=1.0 / res,
            qdegree=qdeg,
            filename=cache_file,
        )
    except Exception:
        if os.path.exists(cache_file):
            uw.pprint(f"  .msh written successfully; DMPlex load skipped for large mesh.")
        else:
            raise


class _Box3D(Enum):
    Bottom = 11; Top = 12; Right = 13; Left = 14; Front = 15; Back = 16


class _Spherical(Enum):
    Lower = 11; Upper = 12


def ensure_h5(cache_file, mesh_type):
    """Write the .msh.h5 beside an existing .msh that lacks one.

    Mesh() on a .msh runs the serial gmsh->h5 conversion next to its input. Done here, not
    through the constructors: SphericalShell/UnstructuredSimplexBox run gmsh unconditionally
    and would replace the campaign's mesh. The gmsh reader flags must match the constructors'
    since they shape the labels written to the file; qdegree and normals do not.
    """
    if os.path.exists(cache_file + ".h5"):
        return
    uw.pprint(f"  converting {os.path.basename(cache_file)} -> .msh.h5")
    boundaries, csys = ((_Box3D, CoordinateSystemType.CARTESIAN) if mesh_type == "box"
                        else (_Spherical, CoordinateSystemType.SPHERICAL))
    uw.discretisation.Mesh(cache_file, boundaries=boundaries, boundary_normals=None,
                           coordinate_system_type=csys,
                           useMultipleTags=True, useRegions=True, markVertices=True)


def generate_meshes(models, base, jobs, cache_dir):
    if uw.mpi.rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
    uw.mpi.barrier()

    # Track which cache files have already been generated this run
    # to avoid regenerating when multiple models share the same mesh type.
    done = set()

    for model in models:
        spec = MESH_SPECS.get(model)
        if spec is None:
            print(f"WARNING: no mesh spec for '{model}' — skipping", file=sys.stderr)
            continue

        uw.pprint(f"\n=== {model} ({spec['type']}) ===")

        for i in jobs:
            res = base * i

            if spec["type"] == "box":
                cache_file = os.path.join(cache_dir, cache_filename_box(res))
                if cache_file in done or os.path.exists(cache_file):
                    uw.pprint(f"  res={res}: .msh cached")
                else:
                    generate_box(res, spec["qdeg"], cache_file)

            elif spec["type"] == "spherical":
                r_i, r_o = spec["r_i"], spec["r_o"]
                cache_file = os.path.join(cache_dir, cache_filename_spherical(r_i, r_o, res))
                if cache_file in done or os.path.exists(cache_file):
                    uw.pprint(f"  res={res}: .msh cached")
                else:
                    generate_spherical(r_i, r_o, res, spec["qdeg"], cache_file)

            done.add(cache_file)
            ensure_h5(cache_file, spec["type"])

    uw.pprint("\nMesh generation complete.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=os.environ.get("UW_MODELS", "poisson-scaling"),
                   help="Space-separated model names (default: $UW_MODELS)")
    p.add_argument("--base",   type=int, default=int(os.environ.get("SCALING_BASE", 12)),
                   help="SCALING_BASE (default: $SCALING_BASE or 12)")
    p.add_argument("--jobs",   default=os.environ.get("JOBS", "1 2 3 4 5"),
                   help="Space-separated JOBS indices (default: $JOBS)")
    p.add_argument("--cache",  default=os.environ.get("MESH_CACHE",
                   os.path.join(os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883"), "mesh_cache")),
                   help="Mesh cache directory (default: $MESH_CACHE or $OUTPUT_BASE/mesh_cache)")
    args = p.parse_args()

    models = args.models.split()
    jobs   = [int(j) for j in args.jobs.split()]

    uw.pprint(f"Models:    {models}")
    uw.pprint(f"BASE:      {args.base}")
    uw.pprint(f"Jobs:      {jobs}")
    uw.pprint(f"Cache dir: {args.cache}")
    uw.pprint(f"Resolutions: {[args.base * i for i in jobs]}")

    generate_meshes(models, args.base, jobs, args.cache)


if __name__ == "__main__":
    main()
    # Force exit to bypass PETSc/MPI finalization hang in single-process Singularity runs.
    import os as _os; _os._exit(0)
