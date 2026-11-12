"""Caching the PETSc HDF5 mesh instead of the Gmsh .msh — measured on Gadi.

WHAT THIS SHOWS. mesh_setup was the worst-scaling stage in the checkpoint campaign by a
wide margin: 670 s at 343 ranks against 10.4 s serial, an efficiency of 0.016, and 76% of
the whole run. Most of that was a SERIAL term.

Mesh(<file>) dispatches on the extension. A .msh goes through _from_gmsh, which first runs
_gmsh_to_h5 -- a rank-0, COMM_SELF Gmsh->HDF5 conversion -- and only then reads in parallel.
A .msh.h5 goes straight to _from_plexh5, a collective read. The campaign cached the .msh, so
every run re-derived the same .h5 and threw it away, and at res 168 also paid a 3.7 GB rank-0
copy into the run directory first.

Under this protocol (res = BASE*i, ranks = i^3) total mesh size grows with rank count while
per-rank work is fixed, so that serial term is an Amdahl term that WORSENS with scale. The
fix is entirely on our side: cache the .h5 and read it in place. No UW3 change -- the .h5
dispatch predates v3.1.0, and both campaigns ran the same container.

Panel 2 is the honest part. The stage is 2.4x faster and still scales badly (0.016 -> 0.035),
so it remains the thing to fix; removing the dominant serial term exposed the next one.

Usage:  python analysis/fig_meshcache.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

BEFORE = "weak-scaling-2026-evalbarrier"   # .msh cache
AFTER  = "weak-scaling-2026-h5mesh"        # .msh.h5 cache, same container, same meshes

# Every stage that exists in both campaigns, so panel 2 can show what mesh_setup is now
# being compared against rather than just asserting that it still dominates.
STAGES = ["mesh_setup", "field_setup", "io_write", "io_read"]


def stage_series(d, stage):
    ns = sorted(d)
    ys, lo, hi = [], [], []
    for n in ns:
        s = d[n]["stages"].get(stage)
        ys.append(s["time"] if s else 0.0)
    _, lo, hi = figdata.stage_bounds(d, stage)
    return ns, ys, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="investigations/figures")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    before = figdata.load(os.path.join(repo, BEFORE))
    after = figdata.load(os.path.join(repo, AFTER))
    if not after:
        raise SystemExit(f"no runs under {AFTER} — has the rsync finished?")

    # A PARTIAL download is the dangerous case, not an empty one: efficiency is
    # y[0]/y[-1], so a single rank count silently reports a perfect 1.000. This campaign
    # is 3 rank counts x 3 replicates; refuse anything less rather than plot it.
    expected = {1: 3, 8: 3, 343: 3}
    missing = {n: expected[n] - len(after.get(n, {}).get("reps", []))
               for n in expected
               if len(after.get(n, {}).get("reps", [])) < expected[n]}
    if missing:
        raise SystemExit(
            "incomplete data for " + AFTER + " — still rsyncing?\n"
            + "\n".join(f"  {n} ranks: have {expected[n] - m} of {expected[n]} replicates"
                         for n, m in sorted(missing.items())))

    fig, ax = fs.new_figure(2)

    # Panel 1 -- the result. Both curves are mesh_setup; only the cached file format differs.
    bn, by, blo, bhi = stage_series(before, "mesh_setup")
    an, ay, alo, ahi = stage_series(after, "mesh_setup")
    fs.line(ax[0], bn, by, "C3", "cached .msh (Gmsh)", dashed=True, marker="x",
            lo=blo, hi=bhi)
    fs.line(ax[0], an, ay, "C1", "cached .msh.h5 (PETSc)", lo=alo, hi=ahi)
    fs.style(ax[0], "wall time (s)", "mesh_setup — cost by cached mesh format",
             xs=sorted(set(bn) | set(an)), logy=True, legend_loc="upper left")

    # Panel 2 -- where the time is NOW. The point of the figure is not that mesh_setup got
    # faster but that it is still the stage that fails to scale, so the next target is
    # inside it rather than elsewhere.
    for s in STAGES:
        if not any(s in after[n]["stages"] for n in after):
            continue
        ns, ys, lo, hi = stage_series(after, s)
        fs.line(ax[1], ns, ys, fs.STAGE.get(s, "C7"), fs.STAGE_LABEL.get(s, s),
                lo=lo, hi=hi)
    fs.style(ax[1], "wall time (s)", "Cost by stage, after the change",
             xs=sorted(after), logy=True, legend_loc="upper left")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "meshcache.png")

    # Efficiency against serial, quoted in the caption rather than drawn: with only three
    # rank counts in the AFTER sweep a curve would imply more resolution than we measured.
    def eff(d):
        ns = sorted(d)
        y = [d[n]["stages"]["mesh_setup"]["time"] for n in ns]
        return y[0] / y[-1], ns[-1]

    be, bmax = eff(before)
    ae, amax = eff(after)

    fs.caption(fig,
               "Same container (v3.1.0), same mesh geometry, same rank counts — the only "
               "change is which cached file the run loads. Mesh() sends a .msh through a "
               "SERIAL rank-0 Gmsh-to-HDF5 conversion before reading; a .msh.h5 goes "
               "straight to a collective read. Under weak scaling the mesh grows with rank "
               "count, so that serial step is an Amdahl term that worsens with scale — which "
               "is why the gap widens to the right. No UW3 change was needed: the .h5 "
               f"dispatch predates v3.1.0. mesh_setup efficiency T(1)/T(N) at {bmax} ranks "
               f"was {be:.3f} and is now {ae:.3f} at {amax} — better by 2.3x, still the "
               "worst-scaling stage in the campaign. Error bars span replicates.")
    fs.finish(fig, "Underworld3 — caching the PETSc HDF5 mesh instead of the Gmsh mesh", out)

    print(f"{'ranks':>7}{'before (s)':>13}{'after (s)':>12}{'speedup':>10}")
    for n in sorted(set(bn) & set(an)):
        b = before[n]["stages"]["mesh_setup"]["time"]
        a = after[n]["stages"]["mesh_setup"]["time"]
        print(f"{n:>7}{b:>13.2f}{a:>12.2f}{b/a:>10.2f}x")
    print(f"\nmesh_setup efficiency: {be:.3f} (before, {bmax} ranks) "
          f"-> {ae:.3f} (after, {amax} ranks)")


if __name__ == "__main__":
    main()
