"""All four cost classes on one axis — what scales, what does not, and what dominates.

What this figure has to prove:
  1. the four cost classes span two orders of magnitude in weak-scaling efficiency, from
     Poisson's solver at 0.79 to checkpoint read at 0.004
  2. the baseline does not change the ORDER but it changes the GAPS enormously, so a single
     efficiency number is never a complete statement. Poisson and Stokes look 1.9x apart
     against serial (0.795 vs 0.418) and indistinguishable against the packed job
     (0.904 vs 0.895) — the apparent difference was node occupancy, not the solvers.
  3. none of it matters as much as panel 3: for a cheap solver, mesh distribution costs
     620x the solve by 2197 ranks, so the solver's good scaling is invisible in practice

COLOUR CONVENTION DIFFERS HERE, deliberately. Elsewhere colour follows the STAGE
(io_write is C0 in the checkpoint figure). This figure compares MODELS, so colour follows
the model — Poisson C0, Stokes C1, advdiff C2, checkpoint C3 — and checkpoint's two
directions are separated by linestyle instead. The legend is explicit either way.

THE CURVES ARE NOT LIKE FOR LIKE, and the caption says so. Poisson and advdiff are
fixed-WORK (unreachable tolerance, capped iterations); Stokes is fixed-TOLERANCE (every
point converges); checkpoint is fixed data per rank. They also run different meshes at
different sizes. What is comparable is the SHAPE of each curve against its own ideal, not
one model's efficiency against another's.

Usage:  python analysis/fig_overview.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

# (campaign, stage, colour, dashed, label)
SERIES = [
    ("weak-scaling-2026-BASE24-container2",       "steady_solves", "C0", False, "Poisson (solve)"),
    ("weak-scaling-2026-stokes-BASE5-3way/1e-6",  "steady_solves", "C1", False, "Stokes (solve, inner 1e-6)"),
    ("weak-scaling-2026-advdiff-BASE24",          "steady_solves", "C2", False, "advection-diffusion (solve)"),
    ("weak-scaling-2026-checkpoint",              "io_write",      "C3", False, "checkpoint write"),
    ("weak-scaling-2026-checkpoint",              "io_read",       "C3", True,  "checkpoint read"),
]

# Panel 3: how the one-off mesh cost compares with the per-solve cost it enables.
DOMINANCE = [
    ("weak-scaling-2026-BASE24-container2",      "C0", "Poisson"),
    ("weak-scaling-2026-stokes-BASE5-3way/1e-6", "C1", "Stokes"),
    ("weak-scaling-2026-advdiff-BASE24",         "C2", "advection-diffusion"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache = {}

    def campaign(path):
        if path not in cache:
            d = figdata.load(os.path.join(repo, path))
            if not d:
                raise SystemExit(f"no runs found under {path}")
            cache[path] = d
        return cache[path]

    fig, ax = fs.new_figure(3)
    all_ranks = set()

    for panel, use_packed in ((0, False), (1, True)):
        for path, stage, colour, dashed, label in SERIES:
            d = campaign(path)
            ns = sorted(d)
            all_ranks.update(ns)
            ys = [d[n]["stages"][stage]["time"] for n in ns]
            lo, hi = figdata.stage_bounds(d, stage)[1:]
            base = fs.baseline_index(ns, figdata.ranks_per_node(d)) if use_packed else 0
            e = fs.efficiency(ys, base)
            e_lo, e_hi = fs.efficiency_bounds(ys, lo, hi, baseline_index=base)
            fs.line(ax[panel], ns, e, colour, label, dashed=dashed,
                    marker="s" if dashed else "o", lo=e_lo, hi=e_hi)
        fs.ideal_line(ax[panel], logy=True)

    xs = sorted(all_ranks)
    fs.style(ax[0], "efficiency  T(1)/T(N)", "Normalised to serial",
             xs=xs, logy=True, legend_loc="lower left")
    fs.style(ax[1], "efficiency  T(64)/T(N)", "Normalised to the 64-rank job",
             xs=xs, logy=True, legend_loc="lower left")

    # Panel 3. mesh_setup is paid once per run and steady_solves is the per-solve cost, so
    # the ratio answers the question a user actually has: does the solver's scaling matter?
    for path, colour, label in DOMINANCE:
        d = campaign(path)
        ns = sorted(d)
        ratio = [d[n]["stages"]["mesh_setup"]["time"] / d[n]["stages"]["steady_solves"]["time"]
                 for n in ns]
        fs.line(ax[2], ns, ratio, colour, label)
    ax[2].axhline(1.0, color="k", linestyle="--", linewidth=1,
                  label="mesh cost = solve cost")
    fs.style(ax[2], "mesh_setup / steady_solves", "What actually dominates a run",
             xs=xs, logy=True, legend_loc="upper left")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "overview.png")

    fs.caption(fig,
               "Curves are NOT like for like and should be read against their own ideal, "
               "not against each other: Poisson and advdiff are fixed-work (capped "
               "iterations), Stokes is fixed-tolerance (every point converges), checkpoint "
               "holds data per rank constant. Meshes and problem sizes differ too. Panel 3 "
               "compares the one-off mesh_setup cost with the per-solve cost — above the "
               "dashed line, mesh distribution costs more than the solve it enables.")
    fs.finish(fig, "Underworld3 weak scaling — four cost classes", out)

    print(f"{'series':<30}{'ranks':>12}{'vs serial':>11}{'vs packed':>11}")
    for path, stage, _, _, label in SERIES:
        d = campaign(path)
        ns = sorted(d)
        ys = [d[n]["stages"][stage]["time"] for n in ns]
        b = fs.baseline_index(ns, figdata.ranks_per_node(d))
        print(f"{label:<30}{f'{ns[0]}-{ns[-1]}':>12}"
              f"{fs.efficiency(ys)[-1]:>11.3f}{fs.efficiency(ys, b)[-1]:>11.3f}")
    print(f"\n{'mesh_setup / steady_solves':<30}")
    for path, _, label in DOMINANCE:
        d = campaign(path)
        ns = sorted(d)
        r = [d[n]["stages"]["mesh_setup"]["time"] / d[n]["stages"]["steady_solves"]["time"]
             for n in ns]
        print(f"  {label:<26} 1 rank {r[0]:>8.2f}   {ns[-1]} ranks {r[-1]:>8.1f}")


if __name__ == "__main__":
    main()
