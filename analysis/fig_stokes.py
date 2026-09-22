"""Stokes weak scaling — per-stage cost and efficiency, and how configurations compare.

Layout (3x2):
  row 1   stokes.tolerance = 1e-5 (recommended)  — cost by stage | efficiency by stage
  row 2   stokes.tolerance = 1e-8 (UW3 default)  — cost by stage | efficiency by stage
  row 3   four configurations overlaid    — first_solve efficiency | steady_solves efficiency

Rows 1 and 2 are the same axes for two tolerance settings, so the comparison is read
vertically: the tighter curves sit ~1.8x higher in cost while the efficiency panels look
nearly identical. That is the point. Tolerance is a CONSTANT-FACTOR tax, not a scalability
defect, and a plot of cost alone would only show the unsurprising fact that a tighter
tolerance takes longer.

WHAT WAS ACTUALLY VARIED. One knob: `stokes.tolerance`. Its setter derives everything else in
fixed proportion — snes_rtol directly, ksp_atol x1e-6, fieldsplit pressure x0.1, velocity
x0.033 — and enables Eisenstat-Walker, so the outer KSP has no fixed rtol at all. The
campaign directories are named after the resulting inner pressure value (default / 1e-6 /
1e-3) for historical reasons; that names an intention, not a mechanism.

This is therefore a ONE-PARAMETER SWEEP, not a study of the inner solve. Isolating the inner
tolerance needs fieldsplit_*_ksp_rtol set independently, which #477 prevents in v3.1.0:
solve() re-derives them from tolerance immediately before PETSc reads them. Achieved
residuals are 2.6e-11 (tol 1e-8) and 2.2e-8 (tol 1e-5).

Row 3 asks the separate question of whether any configuration scales differently: the two
tolerances, plus BASE=10 packed against BASE=10 spread (`--map-by node`), for the initial
solve and the hot path.

`solver_setup` is omitted throughout — it is 0.0 s at every rank count for Stokes.

Efficiency is against SERIAL throughout. The node-occupancy artefact — which moves the
headline Stokes number from 0.32 to 0.87 depending on the baseline — is visible directly in
these curves rather than needing a second one: they fall steeply while ranks-per-node is
still climbing and flatten from 64 ranks, where a node first fills. Gadi nodes are 48 cores,
so the sweep is 27 on one node, then 48+16 at 64 ranks (average 32/node), reaching 47.6/node
only at 1000 — occupancy is not constant past the knee either. Quote the 64-rank-baseline
number only with that explanation attached.

Usage:  python analysis/fig_stokes.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

# Labelled by the knob that was actually turned. `stokes.tolerance` is the only thing set;
# snes_rtol, ksp_atol and both fieldsplit rtols all follow from it in fixed proportion, so
# naming a row after its inner value would name an intention rather than a mechanism. The
# campaign directories keep their original names (default / 1e-6 / 1e-3, after the inner
# pressure value) — those are data identifiers, not claims.
ROWS = [
    ("weak-scaling-2026-stokes-BASE5-3way/1e-6",    "tolerance 1e-5, recommended (inner 1e-6)"),
    ("weak-scaling-2026-stokes-BASE5-3way/default", "tolerance 1e-8, UW3 default (inner 1e-9)"),
]

# Four configurations for the bottom row. BASE=10 runs are at tolerance 1e-5, so the
# tolerance lever is held fixed across the placement pair.
CONFIGS = [
    ("weak-scaling-2026-stokes-BASE5-3way/1e-6",    "C0", "BASE=5, tol 1e-5"),
    ("weak-scaling-2026-stokes-BASE5-3way/default", "C3", "BASE=5, tol 1e-8"),
    ("weak-scaling-2026-stokes-BASE10",             "C2", "BASE=10 packed"),
    ("weak-scaling-2026-stokes-BASE10-spread",      "C4", "BASE=10 spread"),
]

STAGES = ["mesh_setup", "first_solve", "steady_solves"]


def efficiency_series(ax, d, stage, colour, label):
    """Efficiency against SERIAL.

    Only one baseline is drawn. The packed-baseline curve carries the same information in a
    form that has to be explained, whereas serial-normalised curves show it directly: they
    fall while ranks-per-node is still rising (1, 8, 27) and FLATTEN from 64 ranks on, once
    every job is equally packed. The visible knee is the node-occupancy effect.
    """
    ns = sorted(d)
    ys = [d[n]["stages"][stage]["time"] for n in ns]
    lo, hi = figdata.stage_bounds(d, stage)[1:]
    e_lo, e_hi = fs.efficiency_bounds(ys, lo, hi)
    fs.line(ax, ns, fs.efficiency(ys), colour, label, lo=e_lo, hi=e_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache = {}

    def load(path):
        if path not in cache:
            d = figdata.load(os.path.join(repo, path))
            if not d:
                raise SystemExit(f"no runs found under {path}")
            cache[path] = d
        return cache[path]

    fig, ax = fs.new_grid(3, 2)

    # Rows 1-2. A SHARED y range on the cost panels is what makes the 1.8x gap between the
    # tolerances readable at a glance; independent axes would rescale it away.
    cost_max = max(load(p)[n]["stages"][s]["time"]
                   for p, _ in ROWS for n in load(p) for s in STAGES)
    cost_min = min(load(p)[n]["stages"][s]["time"]
                   for p, _ in ROWS for n in load(p) for s in STAGES)

    for r, (path, title) in enumerate(ROWS):
        d = load(path)
        ns = sorted(d)
        for s in STAGES:
            ys = [d[n]["stages"][s]["time"] for n in ns]
            lo, hi = figdata.stage_bounds(d, s)[1:]
            fs.line(ax[r][0], ns, ys, fs.STAGE[s], fs.STAGE_LABEL.get(s, s), lo=lo, hi=hi)
        ax[r][0].set_ylim(cost_min * 0.6, cost_max * 1.6)
        # Lower right: the curves climb to the right, and the panel's shared y range leaves
        # the region below steady_solves empty at high rank counts.
        fs.style(ax[r][0], "wall time (s)", f"Cost by stage — {title}",
                 xs=ns, logy=True, legend_loc="lower right")

        for s in STAGES:
            efficiency_series(ax[r][1], d, s, fs.STAGE[s], fs.STAGE_LABEL.get(s, s))
        # Linear, not log. mesh_setup collapses to 0.005 and is pinned to the axis floor
        # here, which is acceptable: these panels exist to compare the two SOLVE stages
        # between the tolerances, and a linear axis shows the shallow difference between
        # them far better than a log axis spanning two decades to accommodate mesh_setup.
        fs.ideal_line(ax[r][1])
        fs.style(ax[r][1], "weak scaling efficiency  T(1)/T(N)",
                 f"Efficiency by stage — {title}", xs=ns, legend_loc="upper right")

    # Row 3: the same two stages, but comparing configurations rather than stages.
    for c, stage in enumerate(["first_solve", "steady_solves"]):
        all_ns = set()
        for path, colour, label in CONFIGS:
            d = load(path)
            all_ns.update(d)
            efficiency_series(ax[2][c], d, stage, colour, label)
        # Same linear 0-1.05 range as the row 1 and 2 efficiency panels, so a reader can
        # carry a vertical position down the figure and have it mean the same thing.
        fs.ideal_line(ax[2][c])
        fs.style(ax[2][c], "weak scaling efficiency  T(1)/T(N)",
                 f"{stage} — all configurations", xs=sorted(all_ns),
                 legend_loc="upper right")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "stokes.png")

    # Campaigns are held in `cache`, keyed by path, not in a list of tuples.
    reps = sorted({d[n]["n_replicates"] for d in cache.values() for n in d})
    reps_txt = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"

    fs.caption(fig,
               "Fixed-tolerance protocol: every point converges in 1 outer Krylov iteration, "
               "so cost is what varies. ONE knob was turned — stokes.tolerance, 1e-8 vs 1e-5. "
               "snes_rtol, ksp_atol (x1e-6) and the fieldsplit block rtols (x0.1 pressure, "
               "x0.033 velocity) all follow from it in fixed proportion; the outer KSP has no "
               "fixed rtol, since the setter enables Eisenstat-Walker. BASE=5 spherical "
               f"shell, {reps_txt} replicates; BASE=10 at tolerance 1e-5. Efficiency is "
               "normalised to serial. A Gadi node is 48 cores, so the sweep runs 27 on one "
               "node, then 48+16 at 64 ranks (avg 32/node), rising to 47.6/node at 1000 — "
               "the knee at 64 is where a node first fills. Rows 1 and 2 share a y range so "
               "the 1.8x cost difference is visible; note the efficiency panels barely "
               "differ. solver_setup omitted (0.0 s throughout).")
    fs.finish(fig, "Underworld3 Stokes — weak scaling by solve tolerance, "
                   "work per rank and rank placement", out)

    print(f"{'config':<24}{'stage':<15}{'ranks':>7}{'vs serial':>11}{'vs packed':>11}")
    for path, _, label in CONFIGS:
        d = load(path)
        ns = sorted(d)
        p = fs.baseline_index(ns, figdata.ranks_per_node(d))
        for stage in ("first_solve", "steady_solves"):
            ys = [d[n]["stages"][stage]["time"] for n in ns]
            print(f"{label:<24}{stage:<15}{ns[-1]:>7}"
                  f"{fs.efficiency(ys)[-1]:>11.3f}{fs.efficiency(ys, p)[-1]:>11.3f}")


if __name__ == "__main__":
    main()
