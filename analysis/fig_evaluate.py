"""uw.function.evaluate weak scaling — and a retraction.

What this figure has to prove:
  1. evaluate at a mesh variable's OWN coordinates does NOT scale badly. It holds efficiency
     0.80 to 343 ranks and is CHEAPER per point than probing cell interiors at every parallel
     rank count — which is what you expect when no search is needed.
  2. the 2026-08 result that said otherwise (efficiency 0.05, a 10x nodal penalty) was a
     measurement artefact, shown here as the dashed curve
  3. what actually grows is rank IMBALANCE, plotted as the sum of the sync stages

THE ARTEFACT, because it is worth understanding rather than just correcting. The eval stages
had no barrier. eval_nodal ran first, so the ranks entered it skewed by the three
uw.function.evaluate calls in field_setup, and the first collective inside evaluate
(Mesh.get_max_radius, within points_in_domain) absorbed that wait. eval_offnode ran second
against already-synchronised ranks and was never affected — its numbers are unchanged between
the two campaigns (0.97-1.00x), which is the control that proves the mechanism.

Every stage is now followed by a TIMED barrier in its own <name>_sync stage, which
separates the two: <name> is the work, <name>_sync is the waiting. At 343 ranks the wait is 470 us/point against 26 us/point of work.

This is the second time this round that a first timed region absorbed a cost belonging to
what preceded it — the JIT warm-up was the first, and it was already in the README's pitfalls
list when we walked into the skew version of it.

Usage:  python analysis/fig_evaluate.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

CAMPAIGN = "weak-scaling-2026-evalbarrier"          # barriered, benchmark v2
SUPERSEDED = "weak-scaling-2026-checkpoint-evalfix"  # 2026-08, no barrier

STAGES = ["eval_nodal", "eval_offnode"]
COLOUR = {"eval_nodal": fs.ENTITY["nodal"], "eval_offnode": fs.ENTITY["offnode"]}
LABEL = {
    "eval_nodal":   "eval_nodal (own coords)",
    "eval_offnode": "eval_offnode (interiors)",
}


# Everything that ran before eval_nodal and could leave the ranks out of step. The
# barriered campaign splits this across one sync stage per phase; the first barriered run
# had a single eval_sync_1 covering all of them. Summing whichever exist keeps the grey
# curve meaning the same thing in both — total skew entering eval_nodal.
SKEW_STAGES = ["mesh_setup_sync", "field_setup_sync", "jit_warmup_sync", "eval_sync_1"]


def per_point(data, stage):
    """Microseconds per point PER RANK. n_eval_points is the GLOBAL count."""
    ns = sorted(data)
    return ns, [data[n]["stages"][stage]["time"] /
                (data[n]["info"]["n_eval_points"] / n) * 1e6 for n in ns]


def skew_per_point(data):
    """Sum of every sync stage preceding eval_nodal, in us per point per rank."""
    ns = sorted(data)
    present = [s for s in SKEW_STAGES if all(s in data[n]["stages"] for n in ns)]
    if not present:
        raise SystemExit("no sync stage found; is this a barriered campaign?")
    return ns, [sum(data[n]["stages"][s]["time"] for s in present) /
                (data[n]["info"]["n_eval_points"] / n) * 1e6 for n in ns], present


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="investigations/figures")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = figdata.load(os.path.join(repo, CAMPAIGN))
    old = figdata.load(os.path.join(repo, SUPERSEDED))
    if not d:
        raise SystemExit(f"no runs under {CAMPAIGN}")

    ns = sorted(d)
    fig, ax = fs.new_figure(2)

    # Panel 1 — cost per point. The skew is plotted alongside the work, in grey, because it
    # is context rather than a result: it is what the old measurement mistook for work.
    for s in STAGES:
        _, y = per_point(d, s)
        lo, hi = figdata.stage_bounds(d, s)[1:]
        ppr = [d[n]["info"]["n_eval_points"] / n for n in ns]
        fs.line(ax[0], ns, y, COLOUR[s], LABEL[s],
                lo=[a / p * 1e6 for a, p in zip(lo, ppr)],
                hi=[b / p * 1e6 for b, p in zip(hi, ppr)])
    # Skew is plotted from 8 ranks: a serial barrier waits for nobody, so its ~0 would drag
    # a log axis down three decades and squash the curves this figure is about.
    _, skew, skew_used = skew_per_point(d)
    par = [i for i, n in enumerate(ns) if n > 1]
    fs.line(ax[0], [ns[i] for i in par], [skew[i] for i in par], "C7",
            "rank imbalance (sync stages)", marker="s")
    if old:
        on = sorted(old)
        _, oy = per_point(old, "eval_nodal")
        fs.line(ax[0], on, oy, COLOUR["eval_nodal"],
                "eval_nodal as published — SUPERSEDED", dashed=True, marker="x")
    fs.style(ax[0], "µs per point per rank", "Cost per point", xs=ns, logy=True,
             legend_loc="upper left")

    # Panel 2 — efficiency, with the superseded curve for contrast.
    for s in STAGES:
        ys = [d[n]["stages"][s]["time"] for n in ns]
        lo, hi = figdata.stage_bounds(d, s)[1:]
        e_lo, e_hi = fs.efficiency_bounds(ys, lo, hi)
        fs.line(ax[1], ns, fs.efficiency(ys), COLOUR[s], LABEL[s], lo=e_lo, hi=e_hi)
    if old:
        on = sorted(old)
        oys = [old[n]["stages"]["eval_nodal"]["time"] for n in on]
        fs.line(ax[1], on, fs.efficiency(oys), COLOUR["eval_nodal"],
                "eval_nodal as published — SUPERSEDED", dashed=True, marker="x")
    fs.ideal_line(ax[1])
    fs.style(ax[1], "weak scaling efficiency  T(1)/T(N)", "Efficiency",
             xs=ns, legend_loc="upper right")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "evaluate.png")

    reps = sorted({d[n]["n_replicates"] for n in ns})
    reps_txt = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"
    fs.caption(fig,
               f"Same expression, mesh, rank count and point count (~11.4-12.3k per rank); "
               f"the stages differ only in whether the target points are locally owned, so "
               f"the ideal is FLAT. BASE=24, {reps_txt} replicates. An MPI barrier now "
               f"precedes each stage AND IS TIMED (eval_sync_1), so waiting is separated from "
               f"work. The dashed curve is the 2026-08 measurement, which had no barrier: "
               f"eval_nodal ran first and absorbed the imbalance left by field_setup. "
               f"eval_offnode was unaffected then and is unchanged now, which is the control. "
               f"Note in panel 1 that the superseded curve lies on the imbalance curve — that "
               f"is what it was measuring.")
    fs.finish(fig, "Underworld3 uw.function.evaluate — weak scaling to 343 ranks", out)

    print(f"{'ranks':>6}{'nodal':>9}{'offnode':>9}{'skew':>9}{'nodal OLD':>11}")
    _, n_y = per_point(d, "eval_nodal"); _, o_y = per_point(d, "eval_offnode")
    on = sorted(old); _, old_y = per_point(old, "eval_nodal")
    for i, n in enumerate(ns):
        j = on.index(n) if n in on else None
        print(f"{n:>6}{n_y[i]:>9.1f}{o_y[i]:>9.1f}{skew[i]:>9.1f}"
              f"{old_y[j] if j is not None else float('nan'):>11.1f}")
    ys = [d[n]["stages"]["eval_nodal"]["time"] for n in ns]
    os_ = [d[n]["stages"]["eval_offnode"]["time"] for n in ns]
    print(f"\n  efficiency at {ns[-1]} ranks: nodal {fs.efficiency(ys)[-1]:.2f}, "
          f"offnode {fs.efficiency(os_)[-1]:.2f}")


if __name__ == "__main__":
    main()
