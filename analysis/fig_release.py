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

Two outputs per model. `<model>.png` is the front page: this release alone, cost and
efficiency, two panels so it reads large. `release_<model>.png` is the comparison page:
the previous release dashed and a third panel with the ratio current/previous per stage,
the only form in which a 10% change is readable (on the log cost axis it is sub-pixel).
Prints the per-stage tables for both pages.

Usage:  python analysis/fig_release.py [--mode current|compare|both] [--models ...]
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
                    caption=("Poisson equation with a manufactured solution (sin πx sin πy sin "
                             "πz), Dirichlet walls, degree-2 elements on a BASE=24 unstructured "
                             "tetrahedral box, Krylov + algebraic multigrid (GAMG); fixed work: "
                             "unreachable tolerance, 10 KSP iterations per solve")),
    "stokes": dict(prev="weak-scaling-2026-stokes-BASE5-3way/default", curr="weak-scaling-2026-11-stokes",
                   start=8, stages=SOLVER_STAGES,
                   caption=("Isoviscous incompressible Stokes in a spherical shell (r = "
                            "1.22–2.22), no-slip walls, radial body force, Q2–P1 elements on a "
                            "BASE=5 unstructured mesh, Newton with a Schur-complement "
                            "fieldsplit and GAMG on the velocity block; fixed tolerance 1e-8, "
                            "every point converged; from 8 ranks, since the serial point "
                            "measures node occupancy, not scaling")),
    "advdiff": dict(prev="weak-scaling-2026-advdiff-BASE24", curr="weak-scaling-2026-11-advdiff",
                    start=1, stages=SOLVER_STAGES,
                    caption=("Advection–diffusion of a scalar (initial field sin πx sin πy sin "
                             "πz, unit diffusivity) in a prescribed divergence-free solid-body "
                             "rotation, semi-Lagrangian characteristics with nodal projection "
                             "(SLCN), degree-2 elements, Dirichlet walls, BASE=24 box; 10 "
                             "timesteps at CFL 0.5, fixed work")),
    # legend_loc / legend_panel pin the front-page legend where "best" collides: four long
    # I/O labels only fit in the cost panel's upper left, above curves still near the floor.
    "checkpoint": dict(prev="weak-scaling-2026-evalbarrier", curr="weak-scaling-2026-11-checkpoint",
                       start=1, stages=["io_write", "io_read", "io_write_native", "io_read_native"],
                       legend_loc="upper left", legend_panel="cost",
                       caption=("Three fields set from expressions on a BASE=24 box (velocity "
                                "Q2, pressure and temperature P1), written and read back two "
                                "ways: the visualisation/remap format (write_timestep / "
                                "read_timestep) and the PETSc-native reload (petsc_reload / "
                                "read_checkpoint); no solve")),
    "evaluate": dict(prev="weak-scaling-2026-evalbarrier", curr="weak-scaling-2026-11-checkpoint",
                     start=1, stages=["eval_nodal", "eval_offnode"],
                     caption=("uw.function.evaluate of a P1 field (T = 1 − z) at ~12k points "
                              "per rank on a BASE=24 box: at the variable's own nodes, where no "
                              "search is needed, and half a cell into the cell interiors, where "
                              "the owning cell must be found first; same runs as the checkpoint "
                              "campaign")),
    "swarm": dict(prev=None, curr="weak-scaling-2026-11-swarm",
                  start=1, stages=["mesh_setup", "swarm_setup", "first_advect", "advect_steps"],
                  caption=("Dense particle swarm (20 per tetrahedron, 1.3 M per rank) on a "
                           "BASE=24 box, advected with UW3's own Swarm.advection (RK2 midpoint, "
                           "global_evaluate for the velocity, migrate after each step) through "
                           "a steady divergence-free 3D cellular flow carried on a degree-2 "
                           "velocity field, zero normal flow at the walls; 10 steps of half a "
                           "cell; no solve")),
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


def figure_legend(fig, rows, anchor=0.935):
    """One figure-level legend for every panel, laid out as the given rows.

    matplotlib fills a legend column-major, so the rows are interleaved into one list
    (padded with blanks) and ncol set to the longest row. Inside a panel, six entries would
    cover the low-rank points of whichever curves start high.
    """
    ncol = max(len(r) for r in rows)
    blank = lambda: Line2D([], [], linestyle="none", label=" ")
    rows = [r + [blank() for _ in range(ncol - len(r))] for r in rows]
    proxies = [h for col in zip(*rows) for h in col]
    fig.legend(handles=proxies, loc="upper center", ncol=ncol, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, anchor), handlelength=3.0, columnspacing=2.5)


def _stage_proxy(stage):
    return Line2D([], [], color=fs.STAGE[stage], marker=STAGE_MARKER[stage], markersize=5,
                  label=fs.STAGE_LABEL.get(stage, stage))


def _ideal_proxy(label="ideal"):
    return Line2D([], [], color="k", linestyle="--", linewidth=1, label=label)


def _release_proxy(version, dashed):
    return Line2D([], [], color=RELEASE_GREY, linestyle="--" if dashed else "-",
                  label=f"{version} ({'dashed' if dashed else 'solid'})")


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


def exclusion_notes(*rounds):
    ncold = sum(d[n]["n_cold"] for d in rounds for n in d)
    nexcl = sum(d[n].get("n_excluded", 0) for d in rounds for n in d)
    txt = f" {ncold} cold-JIT replicate{'s' if ncold != 1 else ''} excluded." if ncold else ""
    if nexcl:
        txt += (f" {nexcl} replicate{'s' if nexcl != 1 else ''} excluded as a slow-node "
                f"outlier (reason in runs.csv).")
    return txt


def cost_and_efficiency(ax, curr, prev, stages, start, labels=None, legend_loc=None,
                        legend_panel="efficiency"):
    """The two panels every figure has: current release solid, previous (if any) dashed.

    With `legend_loc` one panel carries the legend (the front-page figure: one release, few
    entries); without it the caller draws a figure-level legend (the comparison figure,
    where six entries would cover the low-rank points). `legend_panel` picks which panel:
    the efficiency panel's lower left is usually empty, but four long I/O labels only fit
    in the cost panel's upper left, where every curve is still near the floor.
    """
    labels = labels or {}
    all_ranks = sorted(set(prev) | set(curr))
    for s in stages:
        for d, dashed in ((curr, False), (prev, True)):
            ns, ys, lo, hi = stage_series(d, s)
            if not ns:
                continue
            fs.line(ax[0], ns, ys, fs.STAGE[s], labels.get(s, s), dashed=dashed,
                    marker=STAGE_MARKER[s], lo=lo, hi=hi)
            e_lo, e_hi = fs.efficiency_bounds(ys, lo, hi)
            fs.line(ax[1], ns, fs.efficiency(ys), fs.STAGE[s], labels.get(s, s),
                    dashed=dashed, marker=STAGE_MARKER[s], lo=e_lo, hi=e_hi)
    on_cost = legend_loc is not None and legend_panel == "cost"
    on_eff = legend_loc is not None and not on_cost
    fs.style(ax[0], "wall time (s)", "Cost by stage", xs=all_ranks, logy=True, legend=False)
    fs.ideal_line(ax[1])
    fs.style(ax[1], f"weak scaling efficiency  T({start})/T(N)", "Efficiency by stage",
             xs=all_ranks, legend=on_eff, legend_loc=legend_loc or "best")
    if on_cost:
        # The ideal line lives in the other panel, so it joins this legend as a proxy.
        handles, _ = ax[0].get_legend_handles_labels()
        ax[0].legend(handles=handles + [_ideal_proxy()], loc=legend_loc, fontsize=9)


def draw_current(model, pair, curr, cv, outdir):
    """The front-page figure: this release alone, two panels, legend inside the efficiency
    panel and no caption in the image — the README carries the caption text, printed below
    so it can be regenerated with the figure."""
    stages = pair["stages"]
    fig, ax = fs.new_figure(2)
    cost_and_efficiency(ax, curr, {}, stages, pair["start"], labels=fs.STAGE_LABEL,
                        legend_loc=pair.get("legend_loc", "best"),
                        legend_panel=pair.get("legend_panel", "efficiency"))
    out = os.path.join(outdir, f"{model}.png")
    # No legend row and no caption, so the panels can reach up to the title.
    fs.finish(fig, f"Underworld3 {TITLE[model]} — weak scaling, {cv}", out, top=0.965)
    print(f"CAPTION {model}: {pair['caption']}. {cv}, {reps_text(curr)} replicates per "
          f"point; bars span the replicate min-max range.{exclusion_notes(curr)}")

    ns = sorted(curr)
    print(f"\n{TITLE[model]} ({cv}): wall time s / efficiency T({pair['start']})/T(N)")
    print(f"{'ranks':>6}" + "".join(f"{s[:14]:>22}" for s in stages))
    for n in ns:
        cells = []
        for s in stages:
            ys = [curr[m]["stages"][s]["time"] for m in ns if s in curr[m]["stages"]]
            t = curr[n]["stages"].get(s, {}).get("time")
            cells.append(f"{t:>12.1f} / {ys[0] / t:5.2f}" if t else f"{'':>20}")
        print(f"{n:>6}" + "".join(f"{c:>22}" for c in cells))


def draw_compare(model, pair, prev, curr, pv, cv, outdir, md):
    """The comparison figure: both releases plus the ratio panel."""
    stages = pair["stages"]
    fig, ax = fs.new_figure(3)
    cost_and_efficiency(ax, curr, prev, stages, pair["start"])

    rmax, common = 0.0, []
    for s in stages:
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
    ax[2].axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax[2].set_ylim(0, max(1.3, rmax * 1.1))
    fs.style(ax[2], f"T({cv}) / T({pv})", f"{cv} relative to {pv}", xs=common, legend=False)

    figure_legend(fig, [[_ideal_proxy("ideal / no change"), _release_proxy(cv, False),
                         _release_proxy(pv, True)],
                        [_stage_proxy(s) for s in stages]])
    harness = f" {HARNESS_NOTE}" if "mesh_setup" in stages else ""
    fs.caption(fig, f"{pair['caption']}. Solid: {cv}, {reps_text(curr)} replicates per point; "
                    f"dashed: {pv}, {reps_text(prev)}. Bars span the replicate min-max range; "
                    f"in the ratio panel they combine both releases' ranges."
                    f"{exclusion_notes(prev, curr)}{harness}")
    out = os.path.join(outdir, f"release_{model}.png")
    fs.finish(fig, f"Underworld3 {TITLE[model]} — weak scaling, {cv} against {pv}", out,
              top=0.83)

    print(f"\n{TITLE[model]}: {cv} / {pv}")
    print(f"{'ranks':>6}" + "".join(f"{s[:13]:>15}" for s in stages))
    rows = {s: dict(zip(ns_, r_)) for m, s, ns_, r_ in md if m == model}
    for n in common:
        print(f"{n:>6}" + "".join(f"{rows[s].get(n, float('nan')):>15.3f}" for s in stages))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--models", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    ap.add_argument("--mode", default="both", choices=["current", "compare", "both"],
                    help="current: <model>.png, this release only (front page); compare: "
                         "release_<model>.png against the previous round")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    md = []   # comparison table, one row per model/stage, one column per common rank count
    for model in args.models:
        pair = PAIRS[model]
        start = pair["start"]
        curr = figdata.load(os.path.join(repo, pair["curr"]))
        if not curr:
            print(f"{TITLE[model]}: no runs under {pair['curr']} yet — skipped")
            continue
        curr = {n: v for n, v in curr.items() if n >= start}
        cv = version(pair["curr"])
        if args.mode in ("current", "both"):
            draw_current(model, pair, curr, cv, outdir)
        if args.mode in ("compare", "both"):
            prev = figdata.load(os.path.join(repo, pair["prev"])) if pair["prev"] else {}
            prev = {n: v for n, v in prev.items() if n >= start}
            if prev:
                draw_compare(model, pair, prev, curr, version(pair["prev"]), cv, outdir, md)
            else:
                print(f"{TITLE[model]}: no previous round to compare against")

    if md:
        ranks = sorted({n for _, _, ns, _ in md for n in ns})
        print("\nChange in wall time, % (negative = faster):\n")
        print("| model | stage | " + " | ".join(str(n) for n in ranks) + " |")
        print("|---|---|" + "---|" * len(ranks))
        for model, s, ns, r in md:
            cells = dict(zip(ns, r))
            print(f"| {TITLE[model]} | `{s}` | " + " | ".join(
                f"{(cells[n] - 1) * 100:+.0f}" if n in cells else "" for n in ranks) + " |")


if __name__ == "__main__":
    main()
