"""uw.function.evaluate weak scaling — the easy case is the slow one.

What this figure has to prove:
  1. evaluating at a mesh variable's OWN coordinates collapses (efficiency 0.05 by 125
     ranks) while evaluating at cell interiors holds (0.56 and flat from 64 ranks)
  2. the gap is large in absolute terms: 470 us/point vs 46 us/point at 343 ranks, a 10x
     penalty for the case that requires NO search at all
  3. the ratio inverts with rank count — 0.8 at 1 rank, 10.3 at 343

This is the counter-intuitive one. Nodal points sit exactly on cell vertices, shared by
every cell touching them, and that degeneracy — not a missing fast path — is what costs.

READ BEFORE REUSING THIS DATA. Only `weak-scaling-2026-checkpoint-evalfix` measures point
location. The earlier `weak-scaling-2026-checkpoint` eval stages are flat at 3 ms because
the probe was a COORDINATE-ONLY expression, which `is_pure_sympy_expression()` routes to a
lambdify fast path (functions_unit_system.py:157) that never touches the mesh. The probe
must reference a MeshVariable.

Usage:  python analysis/fig_evaluate.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

CAMPAIGN = "weak-scaling-2026-checkpoint-evalfix"

STAGES = ["eval_nodal", "eval_offnode"]
COLOUR = {"eval_nodal": fs.ENTITY["nodal"], "eval_offnode": fs.ENTITY["offnode"]}
LABEL = {
    "eval_nodal":   "eval_nodal (at the variable's own coords)",
    "eval_offnode": "eval_offnode (cell interiors)",
}


def per_point_us(data, stage):
    """Microseconds per point PER RANK.

    n_eval_points in run_info.json is the GLOBAL count, so it must be divided by the rank
    count before it means anything per-rank. The weak-scaling protocol holds points per rank
    roughly constant (~11.4-12.3k), which is what makes a flat line the ideal here.
    """
    ns = sorted(data)
    return ns, [data[n]["stages"][stage]["time"] /
                (data[n]["info"]["n_eval_points"] / n) * 1e6 for n in ns]


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
    res = [data[n]["info"]["res"] for n in ns]
    times = {s: [data[n]["stages"][s]["time"] for n in ns] for s in STAGES}
    rng = {s: figdata.stage_bounds(data, s)[1:] for s in STAGES}

    fig, ax = fs.new_figure(2)

    for s in STAGES:
        lo, hi = rng[s]
        fs.line(ax[0], ns, times[s], COLOUR[s], LABEL[s], lo=lo, hi=hi)
    fs.style(ax[0], "wall time (s)", "Cost by stage", xs=ns, logy=True,
             legend_loc="upper left")

    # Serial baseline, linear axis: the span is 1.0 down to 0.045 and the point is the
    # CONTRAST between a curve holding near 0.56 and one falling to the floor. Linear shows
    # that directly; a log axis would compress it into a gap.
    for st in STAGES:
        e_lo, e_hi = fs.efficiency_bounds(times[st], *rng[st])
        fs.line(ax[1], ns, fs.efficiency(times[st]), COLOUR[st], LABEL[st],
                lo=e_lo, hi=e_hi)
    fs.ideal_line(ax[1])
    fs.style(ax[1], "weak scaling efficiency  T(1)/T(N)", "Efficiency",
             xs=ns, legend_loc="upper right")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "evaluate.png")

    reps = sorted({data[n]["n_replicates"] for n in ns})
    reps_txt = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"
    pts = data[ns[0]]["info"]["n_eval_points"]
    fs.caption(fig,
               f"Same expression, mesh, rank count and point count (~{pts//1000}k per rank) "
               f"in both stages — they differ ONLY in whether the target points are locally "
               f"owned, so the ideal is FLAT time. Probe is a MeshVariable "
               f"(T_soln.sym[0]); a coordinate-only expression takes a lambdify fast path "
               f"and measures nothing. mode='default' is DMInterp+RBF, approximate by "
               f"design and identical in both stages, so accuracy is not what separates "
               f"them. BASE=24, {reps_txt} replicate, so bars are zero-length.")
    fs.finish(fig, "Underworld3 uw.function.evaluate — weak scaling to 343 ranks", out)

    print(f"{'ranks':>6}{'res':>6}{'pts/rank':>10}{'nodal s':>9}{'offnode s':>11}"
          f"{'us/pt nodal':>13}{'us/pt off':>11}{'ratio':>7}")
    _, un = per_point_us(data, "eval_nodal")
    _, uo = per_point_us(data, "eval_offnode")
    for i, n in enumerate(ns):
        ppr = data[n]["info"]["n_eval_points"] / n
        print(f"{n:>6}{res[i]:>6}{ppr:>10.0f}{times['eval_nodal'][i]:>9.3f}"
              f"{times['eval_offnode'][i]:>11.3f}{un[i]:>13.1f}{uo[i]:>11.1f}"
              f"{un[i]/uo[i]:>7.1f}")


if __name__ == "__main__":
    main()
