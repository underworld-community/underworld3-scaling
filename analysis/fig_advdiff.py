"""Advection-diffusion weak scaling — the solver scales, the advection does not.

What this figure has to prove:
  1. how each stage costs and scales (same layout as the Poisson figure)
  2. that within steady_solves the SNES stack is nearly flat while the semi-Lagrangian
     advection grows 7.7x to 1000 ranks and is still climbing
  3. that this does not depend on the solve protocol — the converged control lands on the
     same advection curve, with only the SNES half shifted

Panel 3 carries the headline. Solver work is pinned identically at every rank count
(20 SNES solves, 20 Jacobians, ~200 preconditioner applications), so any growth there is
parallel overhead rather than extra work.

Usage:  python analysis/fig_advdiff.py [--outdir figures]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
import figstyle as fs

FIXED = "weak-scaling-2026-advdiff-BASE24"        # tol=1e-50, ksp_max_it=10
CONVERGED = "weak-scaling-2026-advdiff-meshindep"  # tol=1e-6, no cap
HOT = "steady_solves"
STAGES = ["mesh_setup", "first_solve", "steady_solves"]


def decompose(data):
    """steady_solves split into the semi-Lagrangian step and the SNES stack.

    SLCN is what is left after SNESSolve: the characteristic tracing and interpolation that
    happen before each solve. Both terms are rank-0 times so the remainder is not biased by
    mixing a max-across-ranks total with a single-rank event.
    """
    ns = sorted(data)
    snes = [data[n]["events"].get((HOT, "SNESSolve"), {}).get("time", 0.0) for n in ns]
    total = [data[n]["stages"][HOT]["time_rank0"] for n in ns]
    return ns, [t - s for t, s in zip(total, snes)], snes


def _snes_of(r):
    return r["events"].get((HOT, "SNESSolve"), {}).get("time", 0.0)


def decompose_bounds(data):
    """Replicate ranges for the two components, recomputed per run.

    The subtraction has to happen INSIDE each replicate. Differencing the averaged total and
    the averaged SNES time would give a mean that is right but no spread at all, since the
    averaging has already discarded which total went with which SNES time.
    """
    _, slcn_lo, slcn_hi = figdata.bounds(
        data, lambda r: r["stages"][HOT]["time_rank0"] - _snes_of(r))
    _, snes_lo, snes_hi = figdata.bounds(data, _snes_of)
    return (slcn_lo, slcn_hi), (snes_lo, snes_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixed = figdata.load(os.path.join(repo, FIXED))
    conv = figdata.load(os.path.join(repo, CONVERGED))
    if not fixed:
        raise SystemExit(f"no runs found under {FIXED}")

    ns = sorted(fixed)
    stage_times = {
        s: [fixed[n]["stages"].get(s, {}).get("time", 0.0) for n in ns]
        for s in STAGES if any(s in fixed[n]["stages"] for n in ns)
    }

    fx_n, fx_slcn, fx_snes = decompose(fixed)
    cv_n, cv_slcn, cv_snes = decompose(conv) if conv else ([], [], [])

    fig, ax = fs.new_figure(3)

    rng = {s: figdata.stage_bounds(fixed, s)[1:] for s in stage_times}

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
    fs.style(ax[1], "weak scaling efficiency  T(1)/T(N)", "Efficiency by stage",
             xs=ns, legend_loc="lower left")

    # Solid = fixed-work campaign, dashed = converged control. Colour still follows the
    # component, so the two advection curves lying on top of each other is the visible proof
    # that the protocol does not matter here.
    (sl_lo, sl_hi), (sn_lo, sn_hi) = decompose_bounds(fixed)
    sl_elo, sl_ehi = fs.efficiency_bounds(fx_slcn, sl_lo, sl_hi)
    sn_elo, sn_ehi = fs.efficiency_bounds(fx_snes, sn_lo, sn_hi)

    fs.line(ax[2], fx_n, fs.efficiency(fx_slcn), fs.ENTITY["advection"],
            "semi-Lagrangian advection", lo=sl_elo, hi=sl_ehi)
    fs.line(ax[2], fx_n, fs.efficiency(fx_snes), fs.ENTITY["snes"],
            "SNES (residual, Jacobian, Krylov)", lo=sn_elo, hi=sn_ehi)
    if cv_n:
        # The control gets bars too, even where it ran once. Beyond honesty this keeps every
        # curve in the panel an ErrorbarContainer: matplotlib lists plain Line2D handles
        # before containers, so mixing the two silently reorders the legend.
        (cl_lo, cl_hi), (cn_lo, cn_hi) = decompose_bounds(conv)
        cl_elo, cl_ehi = fs.efficiency_bounds(cv_slcn, cl_lo, cl_hi)
        cn_elo, cn_ehi = fs.efficiency_bounds(cv_snes, cn_lo, cn_hi)
        fs.line(ax[2], cv_n, fs.efficiency(cv_slcn), fs.ENTITY["advection"],
                "advection — converged control", dashed=True, marker="s",
                lo=cl_elo, hi=cl_ehi)
        fs.line(ax[2], cv_n, fs.efficiency(cv_snes), fs.ENTITY["snes"],
                "SNES — converged control", dashed=True, marker="s",
                lo=cn_elo, hi=cn_ehi)
    fs.ideal_line(ax[2])
    # Upper right: the advection curve descends into the lower left, and SNES stays high on
    # the right only above 0.85, leaving the 0.6-0.85 band to the right of 27 ranks clear.
    fs.style(ax[2], "weak scaling efficiency  T(1)/T(N)",
             "Inside steady_solves", xs=ns, legend_loc="center right")

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "advdiff.png")

    # Replicates vary across the sweep; state the range so the bars are read with the right
    # weight — a bar from two runs is a much weaker claim than one from three.
    reps = sorted({fixed[n]["n_replicates"] for n in ns})
    reps_txt = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"

    # Stated numerically as well as drawn: on the log cost panel the bars are narrower than
    # the markers, because the campaign really is this reproducible.
    worst = max(fixed[n]["stages"][s]["spread"] for s in stage_times for n in ns
                if s in fixed[n]["stages"])

    fs.caption(fig,
               "Fixed-work protocol: 20 SNES solves, 20 Jacobian evaluations and ~200 "
               "preconditioner applications at every rank count. BASE=24 box mesh, "
               f"10 timesteps, {reps_txt} replicates per point; bars span the full min-max "
               f"range, at most {worst:.0f}% of the mean. "
               "Dashed: converged control at rtol 1e-6 with no iteration cap.")
    fs.finish(fig, "Underworld3 advection-diffusion — weak scaling to 1000 ranks", out)

    print(f"{'ranks':>6} {'total':>9} {'SLCN':>9} {'SNES':>8} {'SLCN%':>7} "
          f"{'SLCN eff':>9} {'SNES eff':>9}")
    e_slcn, e_snes = fs.efficiency(fx_slcn), fs.efficiency(fx_snes)
    for i, n in enumerate(fx_n):
        tot = fx_slcn[i] + fx_snes[i]
        print(f"{n:>6} {tot:>9.1f} {fx_slcn[i]:>9.1f} {fx_snes[i]:>8.1f} "
              f"{fx_slcn[i] / tot * 100:>6.1f}% {e_slcn[i]:>9.2f} {e_snes[i]:>9.2f}")


if __name__ == "__main__":
    main()
