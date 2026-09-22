"""Recover .msh.h5 mesh files from completed campaign outputs into the mesh cache.

WHY THIS EXISTS. Every scaling run copies the cached .msh into its own output
directory and then calls Mesh(...), which converts Gmsh -> PETSc HDF5 on rank 0
(_gmsh_to_h5) before any parallel read. That conversion is SERIAL, and under our
weak-scaling protocol (res = BASE*i, ranks = i^3) the mesh grows with rank count,
so the serial term grows with rank count while per-rank work stays fixed. Measured
locally at the 8-rank point: 9.14 s -> 5.81 s by loading the .h5 directly, moving
weak-scaling efficiency for the load from 0.45 to 0.65.

Every run therefore leaves behind exactly the file we want -- mesh.msh.h5 -- and
throws it away. This harvests one per resolution back into the cache so future
campaigns load it instead of re-deriving it.

SAFE TO SHARE ACROSS RANK COUNTS. _gmsh_to_h5 writes the plex built on COMM_SELF,
i.e. an undistributed serial DMPlex; the partitioning happens later, on load. One
file per resolution therefore serves every job in the sweep.

VERIFIED: a harvested file loads with byte-identical DM labels to its .msh
(All_Boundaries, UW_Boundaries, the six faces, Elements, celltype, depth).

Usage:
    python harvest_mesh_cache.py --scan /scratch/el06/jg0883 --cache /scratch/el06/jg0883/mesh_cache --dry-run
    python harvest_mesh_cache.py --scan /scratch/el06/jg0883 --cache /scratch/el06/jg0883/mesh_cache
"""

import argparse
import os
import re
import shutil
import sys

# Mesh geometry per output directory, mirroring MESH_SPECS in generate_meshes.py.
# Keep in step with that file -- the cache NAME must match what the scaling
# scripts look for, or the harvested file is silently ignored.
BOX_DIRS = {"poisson_out", "ns_out", "vep_out", "advdiff_out", "checkpoint_out"}
SPHERICAL_DIRS = {"stokes_out": (1.22, 2.22)}

RES_RE = re.compile(r"_res(\d+)_")


def cache_name(out_dir, res):
    if out_dir in BOX_DIRS:
        return f"UnstructuredSimplexBox_res{res}.msh.h5"
    if out_dir in SPHERICAL_DIRS:
        r_i, r_o = SPHERICAL_DIRS[out_dir]
        return f"SphericalShell_ri{r_i}_ro{r_o}_res{res}.msh.h5"
    return None


def find_candidates(scan_roots):
    """Map canonical cache filename -> best source file (largest wins)."""
    found = {}
    for root in scan_roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if "mesh.msh.h5" not in filenames:
                continue
            src = os.path.join(dirpath, "mesh.msh.h5")
            run_dir = os.path.basename(dirpath)
            out_dir = os.path.basename(os.path.dirname(dirpath))
            m = RES_RE.search(run_dir)
            if not m:
                continue
            name = cache_name(out_dir, int(m.group(1)))
            if name is None:
                continue
            size = os.path.getsize(src)
            if size == 0:
                continue
            # A truncated file from a killed job would be smaller than a complete
            # one for the same resolution, so prefer the largest.
            if name not in found or size > found[name][1]:
                found[name] = (src, size)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", nargs="+", required=True,
                    help="directories to search for completed run outputs")
    ap.add_argument("--cache", required=True, help="mesh cache directory to populate")
    ap.add_argument("--dry-run", action="store_true", help="report only, copy nothing")
    ap.add_argument("--force", action="store_true", help="overwrite existing cache entries")
    args = ap.parse_args()

    found = find_candidates(args.scan)
    if not found:
        print("no mesh.msh.h5 files found under: " + " ".join(args.scan), file=sys.stderr)
        return 1

    if not args.dry_run:
        os.makedirs(args.cache, exist_ok=True)

    copied = skipped = 0
    for name in sorted(found):
        src, size = found[name]
        dst = os.path.join(args.cache, name)
        exists = os.path.exists(dst)
        if exists and not args.force:
            print(f"  SKIP  {name:<52} (already in cache)")
            skipped += 1
            continue
        verb = "WOULD" if args.dry_run else ("REPLACE" if exists else "COPY")
        print(f"  {verb:<7}{name:<52} {size/1e6:>8.1f} MB  <- {src}")
        if not args.dry_run:
            shutil.copy2(src, dst)
            copied += 1

    print(f"\n{len(found)} resolution(s) found; "
          f"{copied} copied, {skipped} already present"
          + ("  [dry run — nothing written]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
