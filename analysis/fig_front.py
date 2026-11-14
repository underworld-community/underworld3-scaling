"""The front page — four figures for the current release, one question each.

solvers.png     How do the solvers scale? Colour and marker = model (Poisson, Stokes,
                advection-diffusion, swarm advection); dashed = the first solve or step,
                which carries JIT and setup; solid = a later one, the hot path. Costs are
                per solve or per step, so the two line styles mean the same thing for every
                model and the gap between them is the one-off cost. Efficiency is relative to
                each model's own first point (1 rank; Stokes 8, see fig_release.PAIRS).
evaluate.png    uw.function.evaluate at nodes and off nodes.
mesh.png        mesh_setup. The four campaigns on the box mesh agree within 5% at every rank
                count (same .msh.h5, same DMPlexDistribute), so one line stands for all of
                them, pooled; the Stokes spherical shell is the other line. Dash-dot and
                dotted: the .h5 read and DMPlexDistribute, the parts of the box cost that
                grow.
checkpoint.png  The four I/O stages.

The comparison with the previous release is fig_release.py. Each figure prints a CAPTION
line for the README. Usage: python analysis/fig_front.py [--figures solvers mesh ...]
"""

import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from fig_release import (PAIRS, cost_and_efficiency, exclusion_notes, reps_text, stage_series,
                         version, _ideal_proxy)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Laid out for a page: 160 mm wide (A4 less 25 mm margins) by 80 mm, saved uncropped at
# that size, so two near-square panels sit side by side. Fonts shrink to match; the figure
# title sits just above the panel titles; legends go inside a panel where it is empty
# (placed by hand per figure — check the PNG after a change).
FIGSIZE = (160 / 25.4, 80 / 25.4)
RC = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
      "ytick.labelsize": 7, "legend.fontsize": 7}
PANELS_TOP = 0.92
# The bare stage names where figstyle.STAGE_LABEL's parentheticals would not fit the
# page-width panels; the caption carries the UW3 call behind each.
LABELS = dict(fs.STAGE_LABEL, **{s: s for s in ("io_write", "io_read", "io_write_native",
                                                "io_read_native", "eval_nodal", "eval_offnode")})

# `steps` names the run_info field holding how many solves or steps the steady stage ran
# (absent = one). Colours are fixed per model so the mapping survives a model being dropped.
SOLVERS = {
    "poisson": dict(first="first_solve", steady="steady_solves",
                    colour="C0", marker="o", label="Poisson"),
    "stokes": dict(first="first_solve", steady="steady_solves",
                   colour="C1", marker="s", label="Stokes (from 8 ranks)"),
    "advdiff": dict(first="first_solve", steady="steady_solves", steps="n_steady_steps",
                    colour="C2", marker="^", label="Advection-diffusion"),
    "swarm": dict(first="first_advect", steady="advect_steps", steps="n_steps",
                  colour="C4", marker="D", label="Swarm advection"),
}
MESHES = {
    "box": dict(models=["poisson", "advdiff", "checkpoint", "swarm"], colour="C1", marker="o",
                label="box, BASE=24"),
    "shell": dict(models=["stokes"], colour="C0", marker="s",
                  label="shell, BASE=5"),
}
# The two parts of mesh_setup that grow, drawn as "of which" lines: the parallel HDF5 read
# of the cached mesh and the redistribution. They are disjoint; the remainder (interpolation,
# stratify, the MeshVariables) is small at every rank count.
PARTS = [("_from_plexh5", "-.", "box: .h5 read"),
         ("DMPlexDistribute", ":", "box: DMPlexDistribute")]
GREY = "0.35"
# Every efficiency panel is linear with the same top; the floor is raised where nothing
# falls lower, so the curves use the height.
EFF_YMAX = 1.05
EFF_YMIN = {"solvers": 0.3, "evaluate": 0.3, "mesh": 0.0, "checkpoint": 0.0}


def finish(fig, title, out):
    fs.finish(fig, title, out, top=PANELS_TOP, title_pad=0.015, title_size=10, dpi=200,
              crop=False)


def load(model):
    pair = PAIRS[model]
    d = figdata.load(os.path.join(REPO, pair["curr"]))
    return {n: v for n, v in d.items() if n >= pair["start"]}


def per_unit(d, n, steps):
    return float(d[n]["info"][steps]) if steps else 1.0


def pooled(models, stage):
    """Warm replicates of several campaigns merged per rank count, in the shape
    `figdata.load` returns, so `stage_series` and `bounds` read it unchanged."""
    out = {}
    for m in models:
        for n, v in load(m).items():
            o = out.setdefault(n, {"reps": [], "n_cold": 0, "n_excluded": 0})
            o["reps"] += [r for r in v["reps"] if stage in r["stages"]]
            o["n_cold"] += v["n_cold"]
            o["n_excluded"] += v.get("n_excluded", 0)
    for o in out.values():
        ts = [r["stages"][stage]["time"] for r in o["reps"]]
        o["stages"] = {stage: {"time": st.mean(ts)}}
        o["n_replicates"] = len(ts)
    return out


def cv_of(models):
    vs = {version(PAIRS[m]["curr"]) for m in models}
    assert len(vs) == 1, f"one release per front page, got {vs}"
    return vs.pop()


def efficiency_lines(ax, ns, ys, lo, hi, colour, marker, dashed):
    e_lo, e_hi = fs.efficiency_bounds(ys, lo, hi)
    fs.line(ax, ns, fs.efficiency(ys), colour, "", dashed=dashed, marker=marker,
            lo=e_lo, hi=e_hi)


def draw_solvers(outdir):
    data = {m: load(m) for m in SOLVERS}
    cv = cv_of(SOLVERS)
    fig, ax = fs.new_figure(2, width_per=FIGSIZE[0] / 2, height=FIGSIZE[1])
    all_ranks = sorted(set().union(*(set(d) for d in data.values())))
    for m, spec in SOLVERS.items():
        d = data[m]
        for stage, dashed in ((spec["first"], True), (spec["steady"], False)):
            ns, ys, lo, hi = stage_series(d, stage)
            k = [per_unit(d, n, spec.get("steps")) if not dashed else 1.0 for n in ns]
            ys, lo, hi = ([y / q for y, q in zip(v, k)] for v in (ys, lo, hi))
            fs.line(ax[0], ns, ys, spec["colour"], "", dashed=dashed, marker=spec["marker"],
                    lo=lo, hi=hi)
            efficiency_lines(ax[1], ns, ys, lo, hi, spec["colour"], spec["marker"], dashed)
    fs.style(ax[0], "wall time (s)", "Cost per solve / step", xs=all_ranks, logy=True,
             legend=False)
    fs.ideal_line(ax[1])
    fs.style(ax[1], "efficiency", "Efficiency T(N₀)/T(N)", xs=all_ranks, legend=False)
    # Two small legends, each where the panel is empty: the models in the lower left, under
    # every curve at the ranks the box spans; the line styles in the upper right, where
    # only the ideal line runs. One seven-entry box would reach the 1000-rank points.
    models = [Line2D([], [], color=s["colour"], marker=s["marker"], markersize=fs.MARKERSIZE,
                     label=s["label"]) for s in SOLVERS.values()]
    styles = [Line2D([], [], color=GREY, linestyle="--", label="first solve / step"),
              Line2D([], [], color=GREY, linestyle="-", label="later solve / step"),
              _ideal_proxy()]
    ax[1].set_ylim(EFF_YMIN["solvers"], EFF_YMAX)
    ax[1].add_artist(ax[1].legend(handles=models, loc="lower left", fontsize=fs.LEGEND_FONTSIZE))
    ax[1].legend(handles=styles, loc="upper right", fontsize=fs.LEGEND_FONTSIZE)
    out = os.path.join(outdir, "solvers.png")
    finish(fig, f"Underworld3 solvers — weak scaling, {cv}", out)

    notes = exclusion_notes(*data.values())
    print(f"CAPTION solvers: Wall time of one solve (Poisson, Stokes) or one timestep "
          f"(advection-diffusion, swarm advection), dashed for the first, which includes JIT "
          f"compilation and solver setup, solid for the steady ones; efficiency is relative to "
          f"each model's own first point (1 rank; Stokes 8). {cv}, "
          f"{reps_text(data['poisson'])} replicates per point; bars span the replicate "
          f"min-max range.{notes}")
    print(f"\nSolvers ({cv}): wall time s per solve/step, first (dashed) | steady (solid), "
          f"and efficiency")
    for m, spec in SOLVERS.items():
        d = data[m]
        ns = sorted(d)
        ff = [d[n]["stages"][spec["first"]]["time"] for n in ns]
        ss = [d[n]["stages"][spec["steady"]]["time"] / per_unit(d, n, spec.get("steps")) for n in ns]
        print(f"  {spec['label']}")
        for n, f, s in zip(ns, ff, ss):
            print(f"  {n:>6}  {f:>10.1f} ({ff[0] / f:4.2f})  {s:>10.1f} ({ss[0] / s:4.2f})")


def draw_mesh(outdir):
    stage = "mesh_setup"
    data = {k: pooled(spec["models"], stage) for k, spec in MESHES.items()}
    cv = cv_of([m for spec in MESHES.values() for m in spec["models"]])
    fig, ax = fs.new_figure(2, width_per=FIGSIZE[0] / 2, height=FIGSIZE[1])
    all_ranks = sorted(set().union(*(set(d) for d in data.values())))
    for k, spec in MESHES.items():
        ns, ys, lo, hi = stage_series(data[k], stage)
        fs.line(ax[0], ns, ys, spec["colour"], "", marker=spec["marker"], lo=lo, hi=hi)
        efficiency_lines(ax[1], ns, ys, lo, hi, spec["colour"], spec["marker"], False)
    # From one box campaign (events are recorded per campaign, and the four agree); max over
    # ranks, as the stage total is the slowest rank's time too. Serial runs do not
    # distribute, so that line starts at 8.
    one_box = load(MESHES["box"]["models"][0])
    parts = {}
    for name, ls, _ in PARTS:
        pn, pt = figdata.event(one_box, stage, name, "time_max")
        pn, pt = zip(*[(n, t) for n, t in zip(pn, pt) if t > 0])
        parts[name] = dict(zip(pn, pt))
        ax[0].plot(pn, pt, color=MESHES["box"]["colour"], linestyle=ls, marker="None")
    fs.style(ax[0], "wall time (s)", "Cost", xs=all_ranks, logy=True, legend=False)
    fs.ideal_line(ax[1])
    fs.style(ax[1], "efficiency", "Efficiency T(1)/T(N)", xs=all_ranks, legend=False)
    ax[1].set_ylim(EFF_YMIN["mesh"], EFF_YMAX)
    # Two small legends where the panels are empty: the meshes and the ideal line in the
    # efficiency panel's upper right (both curves are below 0.4 from 64 ranks on, only
    # the ideal line runs under it); the "of which" lines in the cost panel's lower
    # right, under the 0.7 s floor the box reaches by 8 ranks.
    meshes = [Line2D([], [], color=m["colour"], marker=m["marker"], markersize=fs.MARKERSIZE,
                     label=m["label"]) for m in MESHES.values()] + [_ideal_proxy()]
    parts_ = [Line2D([], [], color=MESHES["box"]["colour"], linestyle=ls, label=label)
              for _, ls, label in PARTS]
    ax[1].legend(handles=meshes, loc="upper right", fontsize=fs.LEGEND_FONTSIZE)
    ax[0].legend(handles=parts_, loc="lower right", fontsize=fs.LEGEND_FONTSIZE,
                 handlelength=2.6)
    out = os.path.join(outdir, "mesh.png")
    finish(fig, f"Underworld3 mesh set-up — weak scaling, {cv}", out)

    box, shell = data["box"], data["shell"]
    print(f"CAPTION mesh: mesh_setup, the uw.discretisation.Mesh constructor on the cached "
          f".msh.h5: the parallel HDF5 read, DMPlexDistribute, interpolation and the mesh "
          f"variables. Box, BASE=24: the Poisson, advection-diffusion, checkpoint and swarm "
          f"campaigns pooled ({reps_text(box)} runs per point; the four agree within 5%). "
          f"Shell, BASE=5: the Stokes campaign ({reps_text(shell)} replicates). Dash-dot and "
          f"dotted: the .h5 read and the redistribution within the box cost. {cv}; bars span "
          f"the min-max range of the pooled runs.{exclusion_notes(box, shell)}")
    print(f"\nMesh set-up ({cv}): wall time s (efficiency)")
    print(f"{'ranks':>6}{'box':>18}{'shell':>18}" + "".join(f"{p[0]:>18}" for p in PARTS))
    bn, by, _, _ = stage_series(box, stage)
    sn, sy, _, _ = stage_series(shell, stage)
    S = dict(zip(sn, sy))
    for n, b in zip(bn, by):
        s = S.get(n)
        print(f"{n:>6}{b:>10.1f} ({by[0] / b:5.3f})" +
              (f"{s:>10.1f} ({sy[0] / s:5.3f})" if s else f"{'':>18}") +
              "".join(f"{parts[p[0]].get(n, 0):>18.1f}" for p in PARTS))


def draw_stages(model, outdir):
    """evaluate and checkpoint: the stage-per-line figure, as before."""
    pair = PAIRS[model]
    d = load(model)
    cv = version(pair["curr"])
    fig, ax = fs.new_figure(2, width_per=FIGSIZE[0] / 2, height=FIGSIZE[1])
    cost_and_efficiency(ax, d, {}, pair["stages"], pair["start"], labels=LABELS,
                        legend_loc=pair.get("legend_loc", "best"),
                        legend_panel=pair.get("legend_panel", "efficiency"))
    ax[0].set_title("Cost")
    ax[1].set_ylabel("efficiency")
    ax[1].set_title(f"Efficiency T({pair['start']})/T(N)")
    ax[1].set_ylim(EFF_YMIN[model], EFF_YMAX)
    title = {"evaluate": "uw.function.evaluate", "checkpoint": "Checkpoint I/O"}[model]
    out = os.path.join(outdir, f"{model}.png")
    finish(fig, f"Underworld3 {title} — weak scaling, {cv}", out)
    print(f"CAPTION {model}: {pair['caption']}. {cv}, {reps_text(d)} replicates per point; "
          f"bars span the replicate min-max range.{exclusion_notes(d)}")
    ns = sorted(d)
    print(f"\n{title} ({cv}): wall time s (efficiency)")
    print(f"{'ranks':>6}" + "".join(f"{s[:14]:>20}" for s in pair["stages"]))
    for n in ns:
        cells = []
        for s in pair["stages"]:
            ys = [d[m]["stages"][s]["time"] for m in ns if s in d[m]["stages"]]
            t = d[n]["stages"].get(s, {}).get("time")
            cells.append(f"{t:>10.2f} ({ys[0] / t:5.3f})" if t else "")
        print(f"{n:>6}" + "".join(f"{c:>20}" for c in cells))


FIGURES = {"solvers": draw_solvers, "evaluate": lambda o: draw_stages("evaluate", o),
           "mesh": draw_mesh, "checkpoint": lambda o: draw_stages("checkpoint", o)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--figures", nargs="+", default=list(FIGURES), choices=list(FIGURES))
    args = ap.parse_args()
    outdir = os.path.join(REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    fs.MARKERSIZE, fs.LEGEND_FONTSIZE, fs.MINOR_TICK_FONTSIZE = 4, 7, 6
    with plt.rc_context(RC):
        for f in args.figures:
            FIGURES[f](outdir)
            print()


if __name__ == "__main__":
    main()
