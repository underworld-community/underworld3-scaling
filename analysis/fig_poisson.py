"""Poisson weak scaling — the reference case.

What this figure has to prove:
  1. every stage scales differently, and the solver is not the expensive one — mesh_setup
     grows 540x across the sweep while steady_solves grows 1.26x
  2. the fixed-work protocol held (10 KSP iterations at every rank count)

A third panel showing AMG coarse-grid consolidation was dropped. The MatMult imbalance is
real (13,965 calls on the busiest rank against 138 on the least busy, within one run at 2197
ranks) but it is PETSc's engineering decision, not UW3 behaviour: coarse levels hold fewer
unknowns than ranks, so GAMG redistributes them onto a subset rather than running collectives
across 2197 ranks for a 50-unknown system. It also explains only the ONSET of the solver's
drift — imbalance saturates from 64 ranks while efficiency keeps falling 0.88 -> 0.79 — and
is dwarfed by mesh_setup at 620x the solve. The numbers are still printed below for anyone
investigating.

Usage:  python analysis/fig_poisson.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

CAMPAIGN = "weak-scaling-2026-BASE24-container2"
HOT = "steady_solves"

# solver_setup is omitted: it is ~1 s and flat at every rank count, so it would sit on the
# floor of a log axis and add nothing.
STAGES = ["mesh_setup", "first_solve", "steady_solves", "error_analysis"]


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
    stage_times = {
        s: [data[n]["stages"].get(s, {}).get("time", 0.0) for n in ns]
        for s in STAGES
        if any(s in data[n]["stages"] for n in ns)
    }

    _, mm_max = figdata.event(data, HOT, "MatMult", "max")
    _, mm_min = figdata.event(data, HOT, "MatMult", "min")
    _, pcapply = figdata.event(data, HOT, "PCApply", "max")

    # PCApply runs once per Krylov iteration plus once for the initial residual, so
    # iterations = count - 1. Valid here because Poisson is a single-level solve; it reads 0
    # for nested solvers like Stokes, where PETSc folds inner KSPSolve calls into the outer
    # counter.
    iters = sorted({c - 1 for c in pcapply if c})

    fig, ax = fs.new_figure(2)

    # Replicate range per stage, reused by both the cost and the efficiency panel.
    rng = {s: figdata.stage_bounds(data, s)[1:] for s in stage_times}

    for s, ys in stage_times.items():
        lo, hi = rng[s]
        fs.line(ax[0], ns, ys, fs.STAGE[s], fs.STAGE_LABEL.get(s, s), lo=lo, hi=hi)
    fs.style(ax[0], "wall time (s)", "Cost by stage", xs=ns, logy=True,
             legend_loc="upper left")

    for s, ys in stage_times.items():
        e_lo, e_hi = fs.efficiency_bounds(ys, *rng[s])
        fs.line(ax[1], ns, fs.efficiency(ys), fs.STAGE[s], fs.STAGE_LABEL.get(s, s),
                lo=e_lo, hi=e_hi)
    fs.ideal_line(ax[1])
    # Lower left is the clear region: curves start high on the left and descend rightwards,
    # so the bottom-left corner stays empty. Upper right would sit on steady_solves.
    fs.style(ax[1], "weak scaling efficiency  T(1)/T(N)", "Efficiency by stage",
             xs=ns, legend_loc="lower left")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "poisson.png")

    # Replicates vary across the sweep, so state the range rather than one number — the
    # error bars are only as trustworthy as the count behind them.
    reps = sorted({data[n]["n_replicates"] for n in ns})
    reps_txt = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"

    # The bars are drawn but are sub-pixel on the log panel — a 1% spread across four
    # decades is smaller than the marker sitting on top of it. Stating the worst case
    # numerically is the only form a reader can actually read there.
    worst = max(data[n]["stages"][s]["spread"] for s in stage_times for n in ns
                if s in data[n]["stages"])
    fs.caption(fig,
               f"Fixed-work protocol: unreachable tolerance with ksp_max_it, giving "
               f"{iters[0] if len(iters) == 1 else iters} KSP iterations at every rank count. "
               f"BASE=24 box mesh, {reps_txt} replicates per point; bars span the full "
               f"min-max range, at most {worst:.0f}% of the mean. "
               f"solver_setup omitted (~1 s, flat).")
    fs.finish(fig, "Underworld3 Poisson — weak scaling to 2197 ranks", out)

    hdr = f"{'ranks':>6}" + "".join(f"{s[:12]:>14}" for s in stage_times)
    print(hdr)
    for i, n in enumerate(ns):
        print(f"{n:>6}" + "".join(f"{ys[i]:>14.2f}" for ys in stage_times.values()))
    print(f"\n{'ranks':>6} {'eff(hot)':>9} {'MM max':>9} {'MM min':>8} {'imbalance':>10}")
    eff_hot = fs.efficiency(stage_times[HOT])
    for i, n in enumerate(ns):
        imb = mm_max[i] / mm_min[i] if mm_min[i] else 1.0
        print(f"{n:>6} {eff_hot[i]:>9.2f} {mm_max[i]:>9} {mm_min[i]:>8} {imb:>10.1f}")


if __name__ == "__main__":
    main()
