"""
Generate scaling efficiency plots from results.csv produced by collect_results.py.

Plots produced (saved as PNG alongside the CSV):
  strong_speedup.png     — T(N_ref)/T(N) vs N, log-log, per stage
  strong_efficiency.png  — speedup/N vs N, per stage
  weak_efficiency.png        — T(N_ref)/T(N) vs N for hot stage (weak scaling)
  weak_efficiency_stages.png — weak efficiency for all stages overlaid, annotated by category
  stage_breakdown.png        — stacked bar of wall time per stage vs nprocs (annotated)
  etd_vs_bdf.png         — per-step steady_solves time for ETD vs BDF (vep-scaling only)
  memory_growth.png      — RSS delta per stage (if memprobe data present)
  jit_overhead.png       — first_solve / (steady_solves / n_steps) ratio vs nprocs

Usage:
    python analysis/plot_scaling.py --csv results.csv --model vep-scaling
    python analysis/plot_scaling.py --csv results.csv --model stokes-scaling --stage first_solve
    python analysis/plot_scaling.py --csv results.csv --model all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

STAGE_ORDER = [
    "mesh_setup", "analytical_setup", "solver_setup",
    "first_solve", "steady_solves", "io", "io_write", "io_read", "io_verify",
    "error_analysis",
]

STAGE_COLOURS = {
    "mesh_setup":       "#4e79a7",
    "analytical_setup": "#76b7b2",
    "solver_setup":     "#f28e2b",
    "first_solve":      "#e15759",
    "steady_solves":    "#59a14f",
    "io":               "#b07aa1",
    "io_write":         "#b07aa1",
    "io_read":          "#edc948",
    "io_verify":        "#9c755f",
    "error_analysis":   "#bab0ac",
}

# Stages grouped by role — used for annotations and line styles.
STAGE_CATEGORIES = {
    "mesh_setup":       "infrastructure",
    "analytical_setup": "infrastructure",
    "error_analysis":   "infrastructure",
    "solver_setup":     "solver_core",
    "first_solve":      "solver_core",
    "steady_solves":    "solver_core",
    "io":               "io",
    "io_write":         "io",
    "io_read":          "io",
    "io_verify":        "io",
}

_CATEGORY_NOTE = (
    "Solver core (solid):      solver_setup, first_solve, steady_solves\n"
    "Infrastructure (dashed):  mesh_setup, analytical_setup, error_analysis\n"
    "I/O (dotted):             io_write, io_read, io_verify"
)


def _load(csv_path, model):
    df = pd.read_csv(csv_path)
    df["nprocs"]     = pd.to_numeric(df["nprocs"],     errors="coerce")
    df["res"]        = pd.to_numeric(df["res"],        errors="coerce")
    df["wall_time_s"]= pd.to_numeric(df["wall_time_s"],errors="coerce")
    df["run_idx"]    = pd.to_numeric(df["run_idx"],    errors="coerce")
    if model != "all":
        df = df[df["model"] == model]
    return df.dropna(subset=["nprocs", "wall_time_s"])


def _median_time(df, stage, groupby=("nprocs", "res")):
    """Median wall time across run_idx replicates."""
    sub = df[df["stage"] == stage].copy()
    return sub.groupby(list(groupby))["wall_time_s"].median().reset_index()


def _save(fig, outdir, name):
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# Plot functions                                                               #
# --------------------------------------------------------------------------- #

def plot_strong_speedup(df, outdir, hot_stage="steady_solves"):
    """Strong scaling: speedup = T(N_min) / T(N) vs N."""
    data = _median_time(df[df["scaling"].str.lower() == "strong"], hot_stage)
    if data.empty:
        print(f"  No strong-scaling data for stage '{hot_stage}' — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for res_val, grp in data.groupby("res"):
        grp = grp.sort_values("nprocs")
        n   = grp["nprocs"].values
        t   = grp["wall_time_s"].values
        if len(n) < 2:
            continue
        speedup = t[0] / t
        ax.loglog(n, speedup, "o-", label=f"res={res_val}")

    # Ideal line
    n_all = data["nprocs"].dropna().values
    if len(n_all):
        n_range = np.array([n_all.min(), n_all.max()])
        ax.loglog(n_range, n_range / n_range[0], "k--", lw=1, label="ideal")

    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel(f"Speedup  T(N_min)/T(N)")
    ax.set_title(f"Strong scaling speedup — {hot_stage}")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    _save(fig, outdir, "strong_speedup.png")


def plot_strong_efficiency(df, outdir, hot_stage="steady_solves"):
    """Strong scaling efficiency = speedup / N."""
    data = _median_time(df[df["scaling"].str.lower() == "strong"], hot_stage)
    if data.empty:
        print(f"  No strong-scaling data for stage '{hot_stage}' — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for res_val, grp in data.groupby("res"):
        grp = grp.sort_values("nprocs")
        n   = grp["nprocs"].values
        t   = grp["wall_time_s"].values
        if len(n) < 2:
            continue
        speedup    = t[0] / t
        efficiency = speedup / (n / n[0])
        ax.semilogx(n, efficiency, "o-", label=f"res={res_val}")

    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal")
    ax.axhline(0.7, color="r", lw=1, ls=":", label="70% threshold")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Parallel efficiency")
    ax.set_title(f"Strong scaling efficiency — {hot_stage}")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    _save(fig, outdir, "strong_efficiency.png")


def plot_weak_efficiency(df, outdir, hot_stage="steady_solves"):
    """Weak scaling efficiency = T(N_min) / T(N)."""
    data = _median_time(df[df["scaling"].str.lower() == "weak"], hot_stage)
    if data.empty:
        print(f"  No weak-scaling data for stage '{hot_stage}' — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    grp = data.sort_values("nprocs")
    n   = grp["nprocs"].values
    t   = grp["wall_time_s"].values
    if len(n) >= 2:
        efficiency = t[0] / t
        ax.semilogx(n, efficiency, "o-", color="#59a14f")

    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal")
    ax.axhline(0.7, color="r", lw=1, ls=":", label="70% threshold")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Weak scaling efficiency  T(N_min)/T(N)")
    ax.set_title(f"Weak scaling efficiency — {hot_stage}")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    _save(fig, outdir, "weak_efficiency.png")


def plot_weak_efficiency_all_stages(df, outdir):
    """Weak scaling efficiency for every stage overlaid on one figure."""
    weak_df = df[df["scaling"].str.lower() == "weak"]
    stages_present = [s for s in STAGE_ORDER if s in weak_df["stage"].unique()]
    if not stages_present:
        print("  No weak-scaling data — skipping per-stage efficiency plot.")
        return

    # Line styles encode stage category so identity isn't color-alone.
    _ls = {"solver_core": "-", "infrastructure": "--", "io": ":"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in stages_present:
        data = _median_time(weak_df, stage)
        if data.empty or len(data) < 2:
            continue
        grp = data.sort_values("nprocs")
        n = grp["nprocs"].values
        t = grp["wall_time_s"].values
        if t[0] == 0:
            continue
        efficiency = t[0] / t
        cat = STAGE_CATEGORIES.get(stage, "solver_core")
        ax.semilogx(n, efficiency, "o",
                    color=STAGE_COLOURS.get(stage, "#aaaaaa"),
                    linestyle=_ls[cat],
                    linewidth=1.5, markersize=5, label=stage)

    ax.axhline(1.0, color="#222222", lw=1, ls="-",  label="ideal")
    ax.axhline(0.7, color="#cc3333", lw=1, ls=":",  label="70% threshold")
    ax.set_ylim(0, 1.4)
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Weak scaling efficiency  T(N=1) / T(N)")
    ax.set_title("Weak scaling efficiency — all stages")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.text(0.02, 0.03, _CATEGORY_NOTE, transform=ax.transAxes, fontsize=6.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f2", alpha=0.85,
                      edgecolor="#cccccc", linewidth=0.8))

    _save(fig, outdir, "weak_efficiency_stages.png")


def plot_stage_breakdown(df, outdir):
    """Stacked bar: wall time per stage at each process count."""
    stages_present = [s for s in STAGE_ORDER if s in df["stage"].unique()]
    if not stages_present:
        print("  No recognisable stages found — skipping stage breakdown.")
        return

    nprocs_vals = sorted(df["nprocs"].dropna().unique())
    bottom = np.zeros(len(nprocs_vals))
    fig, ax = plt.subplots(figsize=(max(6, len(nprocs_vals) * 0.8 + 2), 5))

    for stage in stages_present:
        times = []
        for n in nprocs_vals:
            sub = df[(df["stage"] == stage) & (df["nprocs"] == n)]["wall_time_s"]
            times.append(sub.median() if not sub.empty else 0.0)
        times = np.array(times, dtype=float)
        times = np.nan_to_num(times)
        colour = STAGE_COLOURS.get(stage, "#aaaaaa")
        ax.bar(range(len(nprocs_vals)), times, bottom=bottom, color=colour, label=stage)
        bottom += times

    ax.set_xticks(range(len(nprocs_vals)))
    ax.set_xticklabels([str(int(n)) for n in nprocs_vals])
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Stage breakdown by process count")
    ax.legend(loc="upper left", fontsize=8)

    ax.text(0.99, 0.98, _CATEGORY_NOTE, transform=ax.transAxes, fontsize=6.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f2", alpha=0.85,
                      edgecolor="#cccccc", linewidth=0.8))

    _save(fig, outdir, "stage_breakdown.png")


def plot_etd_vs_bdf(df, outdir):
    """Compare ETD-1 vs BDF-1 steady_solves wall time (vep-scaling only)."""
    sub = df[df["stage"] == "steady_solves"].copy()
    if "integrator" not in sub.columns or sub["integrator"].isna().all():
        print("  No integrator column — skipping ETD vs BDF plot.")
        return
    sub = sub.dropna(subset=["integrator"])
    if sub.empty:
        print("  No integrator data for steady_solves — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colours = {"etd": "#e15759", "bdf": "#4e79a7"}
    for integrator, grp in sub.groupby("integrator"):
        med = grp.groupby("nprocs")["wall_time_s"].median().reset_index().sort_values("nprocs")
        ax.semilogx(med["nprocs"], med["wall_time_s"], "o-",
                    color=colours.get(integrator, "grey"),
                    label=integrator.upper())

    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Wall time — steady_solves (s)")
    ax.set_title("ETD-1 vs BDF-1 steady-solve cost")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    _save(fig, outdir, "etd_vs_bdf.png")


def plot_jit_overhead(df, outdir):
    """Ratio of first_solve time to mean-per-step steady_solves time."""
    first  = _median_time(df, "first_solve")
    steady = _median_time(df, "steady_solves")
    if first.empty or steady.empty:
        print("  Insufficient data for JIT overhead plot — skipping.")
        return

    merged = pd.merge(first, steady, on=["nprocs", "res"], suffixes=("_first", "_steady"))
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for res_val, grp in merged.groupby("res"):
        grp = grp.sort_values("nprocs")
        ratio = grp["wall_time_s_first"] / grp["wall_time_s_steady"]
        ax.semilogx(grp["nprocs"], ratio, "o-", label=f"res={res_val}")

    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("T(first_solve) / T(steady_solves)")
    ax.set_title("JIT overhead: first-solve vs hot-path cost")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    _save(fig, outdir, "jit_overhead.png")


def plot_memory_growth(df, outdir):
    """RSS delta per stage from memprobe data (column rss_delta_mib)."""
    if "rss_delta_mib" not in df.columns:
        return
    sub = df.dropna(subset=["rss_delta_mib"]).copy()
    sub["rss_delta_mib"] = pd.to_numeric(sub["rss_delta_mib"], errors="coerce")
    sub = sub.dropna(subset=["rss_delta_mib"])
    if sub.empty:
        return

    stages_present = [s for s in STAGE_ORDER if s in sub["stage"].unique()]
    nprocs_vals    = sorted(sub["nprocs"].dropna().unique())

    fig, ax = plt.subplots(figsize=(max(6, len(nprocs_vals) * 0.8 + 2), 5))
    x = np.arange(len(stages_present))
    width = 0.8 / max(len(nprocs_vals), 1)

    for i, n in enumerate(nprocs_vals):
        rss = []
        for stage in stages_present:
            vals = sub[(sub["stage"] == stage) & (sub["nprocs"] == n)]["rss_delta_mib"]
            rss.append(vals.median() if not vals.empty else 0.0)
        ax.bar(x + i * width - 0.4, rss, width, label=f"N={int(n)}")

    ax.set_xticks(x)
    ax.set_xticklabels(stages_present, rotation=20, ha="right")
    ax.set_ylabel("RSS delta (MiB)")
    ax.set_title("Memory growth per stage (memprobe)")
    ax.legend()
    _save(fig, outdir, "memory_growth.png")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot scaling results from results.csv"
    )
    parser.add_argument("--csv",   required=True, help="Path to results.csv")
    parser.add_argument("--model", default="all",
                        help="Model name to filter (e.g. vep-scaling, stokes-scaling, all)")
    parser.add_argument("--stage", default="steady_solves",
                        help="Hot-path stage for speedup/efficiency plots (default: steady_solves)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for PNGs (default: same directory as CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = _load(csv_path, args.model)
    if df.empty:
        print(f"No data for model='{args.model}' in {csv_path}", file=sys.stderr)
        sys.exit(1)

    hot_stage = args.stage
    print(f"Generating plots for model='{args.model}', hot_stage='{hot_stage}'")
    print(f"  Data: {len(df)} rows, nprocs={sorted(df['nprocs'].dropna().unique().tolist())}")

    plot_strong_speedup(df, outdir, hot_stage)
    plot_strong_efficiency(df, outdir, hot_stage)
    plot_weak_efficiency(df, outdir, hot_stage)
    plot_weak_efficiency_all_stages(df, outdir)
    plot_stage_breakdown(df, outdir)
    plot_etd_vs_bdf(df, outdir)
    plot_jit_overhead(df, outdir)
    plot_memory_growth(df, outdir)

    print(f"\nAll plots saved to {outdir}")


if __name__ == "__main__":
    main()
