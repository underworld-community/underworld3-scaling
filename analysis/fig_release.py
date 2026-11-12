"""Release over release — the front-page figure, one per model.

What this figure has to answer: did the release change how the code scales? Three stages
(mesh_setup, first_solve, steady_solves); the current release solid, the previous dashed in
the same stage colour; and a third panel with the ratio current/previous per stage. The
ratio panel is the only form in which a 10% change is readable — on the log cost axis it is
sub-pixel, smaller than the marker.

Efficiency is T(first point)/T(N) for both releases. The packed-node baseline the
investigations page insists on cancels here: both releases ran the same placement, so the
ratio is immune to it and the efficiency panel compares like with like. Stokes starts at 8
ranks (see PAIRS). The previous sweep may extend further
than the current one (v3.1.0 ran to 2197 ranks, the v3.2.0 check to 343): cost and
efficiency show every point each release has, the ratio only the ranks both have.

Prints the per-stage ratio table in markdown for the README.

Usage:  python analysis/fig_release.py [--outdir figures] [--models poisson stokes advdiff]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs
from matplotlib.lines import Line2D

# One figure per entry: previous and current campaign (same model script, same parameters),
# the first rank count shown, the stages drawn, and the caption's protocol line.
#
# Stokes starts at 8 ranks. Its serial-to-parallel step is a different regime, not a scaling
# effect: the 1-rank job owns a node's memory bandwidth and pays no communication, and the
# investigations page shows the whole 1 -> 27 loss falls inside one node. Normalising to it
# would report node occupancy as if it were the release's scaling.
#
# solver_setup (~1 s, flat) and error_analysis (Poisson only) are left to the investigations
# page. checkpoint and evaluate come from the same campaign pair (one script, two
# questions); their previous round is the barriered 2026-09 campaign, which read the .msh.
SOLVER_STAGES = ["mesh_setup", "first_solve", "steady_solves"]
PAIRS = {
    "poisson": dict(prev="weak-scaling-2026-BASE24-container2", curr="weak-scaling-2026-11-poisson",
                    start=1, stages=SOLVER_STAGES,
                    caption="Poisson, BASE=24 box, fixed work (unreachable tolerance, 10 KSP iterations)"),
    "stokes": dict(prev="weak-scaling-2026-stokes-BASE5-3way/default", curr="weak-scaling-2026-11-stokes",
                   start=8, stages=SOLVER_STAGES,
                   caption="Stokes, BASE=5 spherical shell, stokes.tolerance 1e-8, every point converged; "
                           "from 8 ranks, since the serial point measures node occupancy, not scaling"),
    "advdiff": dict(prev="weak-scaling-2026-advdiff-BASE24", curr="weak-scaling-2026-11-advdiff",
                    start=1, stages=SOLVER_STAGES,
                    caption="advection-diffusion, BASE=24 box, 10 timesteps, fixed work"),
    "checkpoint": dict(prev="weak-scaling-2026-evalbarrier", curr="weak-scaling-2026-11-checkpoint",
                       start=1, stages=["io_write", "io_read", "io_write_native", "io_read_native"],
                       caption="checkpoint I/O, BASE=24 box, three fields (v, p, T): the remap format "
                               "(write_timestep / read_timestep) and the native reload "
                               "(petsc_reload / read_checkpoint); no solve"),
    "evaluate": dict(prev="weak-scaling-2026-evalbarrier", curr="weak-scaling-2026-11-checkpoint",
                     start=1, stages=["eval_nodal", "eval_offnode"],
                     caption="uw.function.evaluate of a P1 field at ~12k points per rank: at the "
                             "variable's own nodes, and half a cell into the interiors"),
    "swarm": dict(prev=None, curr="weak-scaling-2026-11-swarm",
                  start=1, stages=["mesh_setup", "swarm_setup", "first_advect", "advect_steps"],
                  caption="dense swarm advection, BASE=24 box, 20 particles per cell (fill 3), "
                          "10 RK2 steps of half a cell through a 3D cellular flow; no solve"),
}
TITLE = {"poisson": "Poisson", "stokes": "Stokes", "advdiff": "Advection-diffusion",
         "checkpoint": "Checkpoint I/O", "evaluate": "uw.function.evaluate",
         "swarm": "Swarm advection"}

# Stage = colour + marker, release = line style. One figure-level legend in two rows:
# the line styles, then the stages. Entries are listed column-major (that is how
# matplotlib fills ncol=3), so the pairs below are (row 1, row 2) of one column each.
STAGE_MARKER = {"mesh_setup": "o", "first_solve": "s", "steady_solves": "^",
                "io_write": "o", "io_read": "s", "io_write_native": "^", "io_read_native": "D",
                "eval_nodal": "o", "eval_offnode": "s",
                "swarm_setup": "D", "first_advect": "s", "advect_steps": "^"}
RELEASE_GREY = "0.35"


def figure_legend(fig, cv, pv, stages):
    row1 = [Line2D([], [], color="k", linestyle="--", linewidth=1, label="ideal / no change"),
            Line2D([], [], color=RELEASE_GREY, linestyle="-", label=f"{cv} (solid)")]
    if pv:
        row1.append(Line2D([], [], color=RELEASE_GREY, linestyle="--", label=f"{pv} (dashed)"))
    row2 = [_stage_proxy(s) for s in stages]
    ncol = max(len(row1), len(row2))
    blank = lambda: Line2D([], [], linestyle="none", label=" ")
    row1 += [blank() for _ in range(ncol - len(row1))]
    row2 += [blank() for _ in range(ncol - len(row2))]
    proxies = [h for pair in zip(row1, row2) for h in pair]
    fig.legend(handles=proxies, loc="upper center", ncol=ncol, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.935), handlelength=3.0, columnspacing=2.5)


def _stage_proxy(stage):
    return Line2D([], [], color=fs.STAGE[stage], marker=STAGE_MARKER[stage], markersize=5,
                  label=fs.STAGE_LABEL.get(stage, stage))

# Applies to every v3.1.0 -> v3.2.0 pair; drop once both rounds of a pair read the .h5.
HARNESS_NOTE = ("mesh_setup is not a release comparison: the later campaign read a cached "
                ".msh.h5 where the earlier one converted a .msh on rank 0 first, so its drop "
                "is the harness fix (see investigations).")


def version(campaign):
    return figdata.round_info(campaign).get("uw3_version") or campaign


def stage_series(d, stage):
    """(ranks, mean, lo, hi) over the ranks that have the stage."""
    sub = {n: d[n] for n in sorted(d) if stage in d[n]["stages"]}
    ns = sorted(sub)
    ys = [sub[n]["stages"][stage]["time"] for n in ns]
    _, lo, hi = figdata.bounds(sub, lambda r: r["stages"].get(stage, {}).get("time"))
    return ns, ys, lo, hi


def ratio_series(prev, curr, stage):
    """current/previous at the ranks both have; bars combine both replicate ranges."""
    pn, py, plo, phi = stage_series(prev, stage)
    cn, cy, clo, chi = stage_series(curr, stage)
    P = {n: v for n, v in zip(pn, zip(py, plo, phi))}
    C = {n: v for n, v in zip(cn, zip(cy, clo, chi))}
    ns = [n for n in cn if n in P]
    r = [C[n][0] / P[n][0] for n in ns]
    # Worst cases in each direction: fastest current over slowest previous, and vice versa.
    lo = [C[n][1] / P[n][2] for n in ns]
    hi = [C[n][2] / P[n][1] for n in ns]
    return ns, r, lo, hi


def reps_text(d):
    reps = sorted({d[n]["n_replicates"] for n in d})
    return f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--models", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    md = []   # README table, one row per model/stage, one column per common rank count
    for model in args.models:
        pair = PAIRS[model]
        prev_c, curr_c, start, STAGES = pair["prev"], pair["curr"], pair["start"], pair["stages"]
        curr = figdata.load(os.path.join(repo, curr_c))
        if not curr:
            print(f"{TITLE[model]}: no runs under {curr_c} yet — skipped")
            continue
        # A model with no previous round (first appearance) gets the two left panels only.
        prev = figdata.load(os.path.join(repo, prev_c)) if prev_c else {}
        prev = {n: v for n, v in prev.items() if n >= start}
        curr = {n: v for n, v in curr.items() if n >= start}
        pv, cv = (version(prev_c) if prev else None), version(curr_c)
        all_ranks = sorted(set(prev) | set(curr))

        fig, ax = fs.new_figure(3 if prev else 2)

        for s in STAGES:
            for d, dashed in ((curr, False), (prev, True)):
                ns, ys, lo, hi = stage_series(d, s)
                if not ns:
                    continue
                fs.line(ax[0], ns, ys, fs.STAGE[s], s, dashed=dashed,
                        marker=STAGE_MARKER[s], lo=lo, hi=hi)
                e_lo, e_hi = fs.efficiency_bounds(ys, lo, hi)
                fs.line(ax[1], ns, fs.efficiency(ys), fs.STAGE[s], s, dashed=dashed,
                        marker=STAGE_MARKER[s], lo=e_lo, hi=e_hi)
        # No per-panel legends: one figure-level legend (figure_legend) serves all three
        # panels, and six entries inside a panel cover the low-rank points of whichever
        # curves start high.
        fs.style(ax[0], "wall time (s)", "Cost by stage", xs=all_ranks, logy=True,
                 legend=False)
        fs.ideal_line(ax[1])
        fs.style(ax[1], f"weak scaling efficiency  T({start})/T(N)", "Efficiency by stage",
                 xs=all_ranks, legend=False)

        rmax, common = 0.0, []
        for s in (STAGES if prev else []):
            ns, r, lo, hi = ratio_series(prev, curr, s)
            if not ns:
                continue
            common = ns
            fs.line(ax[2], ns, r, fs.STAGE[s], s, marker=STAGE_MARKER[s], lo=lo, hi=hi)
            rmax = max(rmax, max(hi))
            md.append((model, s, ns, r))
            if s == "mesh_setup":
                # The one curve that is not a release comparison, said where it is read.
                ax[2].annotate("harness (.msh.h5 read), not UW3", xy=(ns[-1], r[-1]),
                               xytext=(-6, -14), textcoords="offset points", ha="right",
                               fontsize=8, color=fs.STAGE[s])
        if prev:
            ax[2].axhline(1.0, color="k", linestyle="--", linewidth=1)
            ax[2].set_ylim(0, max(1.3, rmax * 1.1))
            fs.style(ax[2], f"T({cv}) / T({pv})", f"{cv} relative to {pv}", xs=common,
                     legend=False)
        figure_legend(fig, cv, pv, STAGES)

        ncold = sum(d[n]["n_cold"] for d in (prev, curr) for n in d)
        cold_txt = (f" {ncold} cold-JIT replicate{'s' if ncold != 1 else ''} excluded."
                    if ncold else "")
        prev_txt = (f"dashed: {pv}, {reps_text(prev)}. Bars span the replicate min-max "
                    f"range; in the ratio panel they combine both releases' ranges."
                    if prev else "Bars span the replicate min-max range.")
        harness = f" {HARNESS_NOTE}" if prev and "mesh_setup" in STAGES else ""
        fs.caption(fig,
                   f"{pair['caption']}. Solid: {cv}, {reps_text(curr)} replicates per point; "
                   f"{prev_txt}{cold_txt}{harness}")
        out = os.path.join(outdir, f"release_{model}.png")
        title = (f"Underworld3 {TITLE[model]} — weak scaling, {cv} against {pv}" if prev
                 else f"Underworld3 {TITLE[model]} — weak scaling, {cv}")
        fs.finish(fig, title, out, top=0.83)

        if prev:
            print(f"\n{TITLE[model]}: {cv} / {pv}")
            print(f"{'ranks':>6}" + "".join(f"{s[:13]:>15}" for s in STAGES))
            rows = {s: dict(zip(ns_, r_)) for m, s, ns_, r_ in md if m == model}
            for n in common:
                print(f"{n:>6}" + "".join(f"{rows[s].get(n, float('nan')):>15.3f}"
                                          for s in STAGES))

    if md:
        ranks = sorted({n for _, _, ns, _ in md for n in ns})
        print("\nREADME table — change in wall time, % (negative = faster):\n")
        print("| model | stage | " + " | ".join(str(n) for n in ranks) + " |")
        print("|---|---|" + "---|" * len(ranks))
        for model, s, ns, r in md:
            cells = dict(zip(ns, r))
            print(f"| {TITLE[model]} | `{s}` | " + " | ".join(
                f"{(cells[n] - 1) * 100:+.0f}" if n in cells else "" for n in ranks) + " |")


if __name__ == "__main__":
    main()
