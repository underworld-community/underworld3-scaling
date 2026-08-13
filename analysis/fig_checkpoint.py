"""Checkpoint I/O weak scaling — write scales, read reverses.

What this figure has to prove:
  1. write and read cost diverge: they cross at 64 ranks, and by 1000 ranks read costs 3.8x
     what write does on the identical file
  2. neither scales, but they fail differently — write degrades gracefully, read collapses
  3. that the two stages are NOT symmetric, and io_read is not an I/O measurement

The read curve is the finding, but read WHAT. `mesh.write_timestep()` writes the mesh and all
three fields in one call. `MeshVariable.read_timestep()` is the COORDINATE-REMAP reader: rank
0 reads the file, saved (coord, value) pairs migrate to the ranks that own them, each rank
runs a local KDTree against what it received, and interpolated values migrate back. It exists
for reloading onto a DIFFERENT mesh.

So io_read times point location and swarm migration, not a field read — which is why it
collapses, and why it likely shares a mechanism with the evaluate result. For same-mesh
reload UW3 provides `read_checkpoint()`, a native PETSc DMPlex section/vector path; this
campaign did not measure it.

An earlier version of this figure carried an "achieved bandwidth" panel. It was removed: it
divided the bytes the WRITE produced (mesh included, which the read never touches) by a time
that is mostly interpolation. Both the numerator and the meaning were wrong.

Data volume per rank is held constant by the weak-scaling protocol (30.2 -> 27.1 MB/rank),
so the ideal is FLAT time and bandwidth growing linearly with rank count. Both curves are
therefore measured against a fixed per-rank workload, exactly like the solver figures.

Usage:  python analysis/fig_checkpoint.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

CAMPAIGN = "weak-scaling-2026-checkpoint"

# io_verify is omitted: it is 0.01 s at every rank count (a local comparison, no I/O), so it
# would sit on the floor of a log axis and add nothing. field_setup is omitted for the same
# reason it is omitted from the Poisson figure — small and nearly flat.
STAGES = ["io_write", "io_read"]
LABEL = {"io_write": "io_write (mesh.write_timestep)",
         "io_read": "io_read (read_timestep — coordinate remap)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--campaign", default=CAMPAIGN)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = figdata.load(os.path.join(repo, args.campaign))
    if not data:
        raise SystemExit(f"no runs found under {args.campaign}")

    ns = sorted(data)
    times = {s: [data[n]["stages"][s]["time"] for n in ns] for s in STAGES}
    rng = {s: figdata.stage_bounds(data, s)[1:] for s in STAGES}

    fig, ax = fs.new_figure(2)

    for s in STAGES:
        lo, hi = rng[s]
        fs.line(ax[0], ns, times[s], fs.STAGE[s], LABEL[s], lo=lo, hi=hi)
    fs.style(ax[0], "wall time (s)", "Cost by stage", xs=ns, logy=True,
             legend_loc="upper left")

    # Serial baseline only. A log axis is still needed here, unlike the solver figures:
    # read falls to 0.004, so 2.5 decades have to be shown and a linear axis would pin
    # everything past 125 ranks to the floor, hiding the difference between poor and
    # catastrophic — which is the finding.
    for s in STAGES:
        e_lo, e_hi = fs.efficiency_bounds(times[s], *rng[s])
        fs.line(ax[1], ns, fs.efficiency(times[s]), fs.STAGE[s], LABEL[s],
                lo=e_lo, hi=e_hi)
    fs.ideal_line(ax[1], logy=True)
    fs.style(ax[1], "weak scaling efficiency  T(1)/T(N)", "Efficiency",
             xs=ns, logy=True, legend_loc="lower left")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "checkpoint.png")

    reps = sorted({data[n]["n_replicates"] for n in ns})
    reps_txt = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"
    mb = data[ns[0]]["info"]["checkpoint_bytes"] / 1e6
    fs.caption(fig,
               f"Weak scaling holds data per rank constant ({mb:.0f} MB at 1 rank, "
               f"27 MB/rank at 1000), so the ideal is FLAT time. BASE=24 box mesh, "
               f"{reps_txt} replicates. meshUpdates=True throughout, so every checkpoint "
               f"writes the mesh as well as the fields — ~90% of the bytes at res=24, 66% "
               f"at res=240. Round-trip is bit-exact on all three fields. Note the "
               f"asymmetry in the bars: writes contend with other filesystem users and "
               f"vary by up to 79% between replicates, reads by at most 6%. io_read uses "
               f"the coordinate-remap reader (KDTree point location), not a native reload.")
    fs.finish(fig, "Underworld3 checkpoint I/O — weak scaling to 1000 ranks", out)

    # No MB/s column: checkpoint_bytes is what the WRITE produced, and the read moves a
    # different (smaller) set of bytes through a remap path, so a shared rate is meaningless.
    print(f"{'ranks':>6}{'GB written':>12}{'write s':>9}{'read s':>10}{'read/write':>12}")
    for i, n in enumerate(ns):
        gb = data[n]["info"]["checkpoint_bytes"] / 1e9
        w, r = times["io_write"][i], times["io_read"][i]
        print(f"{n:>6}{gb:>12.2f}{w:>9.2f}{r:>10.2f}{r/w:>12.2f}")


if __name__ == "__main__":
    main()
