"""
Compare weak scaling efficiency across multiple backends / configurations.

Usage (two or more datasets):
    python analysis/compare_backends.py \
        --csv results_a.csv --label "Container" \
        --csv results_b.csv --label "Baremetal BASE=6" \
        --csv results_c.csv --label "Baremetal BASE=12" \
        --model poisson-scaling \
        --outdir /path/to/output

Produces:
    compare_weak_efficiency.png  — efficiency curves per stage
    compare_wall_time.png        — raw median wall time per stage vs nprocs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

STAGE_ORDER = [
    "mesh_setup", "analytical_setup", "solver_setup",
    "first_solve", "steady_solves",
    "io_write", "io_read", "io_verify", "error_analysis",
]

STAGE_COLOURS = {
    "mesh_setup":       "#4e79a7",
    "analytical_setup": "#76b7b2",
    "solver_setup":     "#f28e2b",
    "first_solve":      "#e15759",
    "steady_solves":    "#59a14f",
    "io_write":         "#b07aa1",
    "io_read":          "#edc948",
    "io_verify":        "#9c755f",
    "error_analysis":   "#bab0ac",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]


def _load(csv_path, model, label):
    df = pd.read_csv(csv_path)
    df["nprocs"]      = pd.to_numeric(df["nprocs"],      errors="coerce")
    df["res"]         = pd.to_numeric(df["res"],         errors="coerce")
    df["wall_time_s"] = pd.to_numeric(df["wall_time_s"], errors="coerce")
    df["run_idx"]     = pd.to_numeric(df["run_idx"],     errors="coerce")
    if model != "all":
        df = df[df["model"] == model]
    df = df.dropna(subset=["nprocs", "wall_time_s"])
    df["backend"] = label
    return df


def _median(df, stage):
    sub = df[(df["stage"] == stage) & (df["scaling"].str.lower() == "weak")]
    return sub.groupby("nprocs")["wall_time_s"].median().reset_index().sort_values("nprocs")


def _save(fig, outdir, name):
    path = Path(outdir) / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_efficiency_comparison(datasets, stages, outdir):
    """One panel per stage: efficiency curves for all backends."""
    n_stages = len(stages)
    ncols = min(n_stages, 3)
    nrows = (n_stages + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, stage in enumerate(stages):
        ax = axes[idx // ncols][idx % ncols]
        colour = STAGE_COLOURS.get(stage, "#888888")

        for k, (df, label) in enumerate(datasets):
            marker = MARKERS[k % len(MARKERS)]
            ls = LINESTYLES[k % len(LINESTYLES)]
            med = _median(df, stage)
            if med.empty or len(med) < 2:
                continue
            n = med["nprocs"].values
            t = med["wall_time_s"].values
            eff = t[0] / t
            # Vary shade per dataset so they're distinguishable on same colour
            alpha = 1.0 - 0.2 * k
            ax.semilogx(n, eff, marker=marker, linestyle=ls,
                        color=colour, linewidth=1.8, markersize=6,
                        alpha=alpha, label=label)

        ax.axhline(1.0, color="#222222", lw=1, ls="-",  label="ideal")
        ax.axhline(0.7, color="#cc3333", lw=1, ls=":",  label="70%")
        ax.set_ylim(0, 1.4)
        ax.set_title(stage, fontsize=10)
        ax.set_xlabel("N processes")
        ax.set_ylabel("Efficiency  T(1)/T(N)")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=7)

    for idx in range(len(stages), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    labels = " vs ".join(label for _, label in datasets)
    fig.suptitle(f"Weak scaling efficiency: {labels}", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "compare_weak_efficiency.png")


def plot_walltime_comparison(datasets, stages, outdir):
    """One panel per stage: raw median wall time for all backends."""
    n_stages = len(stages)
    ncols = min(n_stages, 3)
    nrows = (n_stages + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, stage in enumerate(stages):
        ax = axes[idx // ncols][idx % ncols]
        colour = STAGE_COLOURS.get(stage, "#888888")

        for k, (df, label) in enumerate(datasets):
            marker = MARKERS[k % len(MARKERS)]
            ls = LINESTYLES[k % len(LINESTYLES)]
            med = _median(df, stage)
            if med.empty:
                continue
            alpha = 1.0 - 0.2 * k
            ax.semilogx(med["nprocs"], med["wall_time_s"],
                        marker=marker, linestyle=ls,
                        color=colour, linewidth=1.8, markersize=6,
                        alpha=alpha, label=label)

        ax.set_title(stage, fontsize=10)
        ax.set_xlabel("N processes")
        ax.set_ylabel("Wall time (s)")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=7)

    for idx in range(len(stages), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    labels = " vs ".join(label for _, label in datasets)
    fig.suptitle(f"Median wall time: {labels}", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "compare_wall_time.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    action="append", required=True,
                   help="Path to a results CSV (repeat for each dataset)")
    p.add_argument("--label",  action="append", required=True,
                   help="Label for each dataset (same order as --csv)")
    p.add_argument("--model",  default="poisson-scaling")
    p.add_argument("--outdir", default=".")
    p.add_argument("--stages", nargs="+",
                   default=["mesh_setup", "solver_setup", "first_solve",
                            "steady_solves", "error_analysis"])
    args = p.parse_args()

    if len(args.csv) != len(args.label):
        p.error("Number of --csv and --label arguments must match")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    datasets = [(_load(csv, args.model, label), label)
                for csv, label in zip(args.csv, args.label)]

    # Only include stages present in ALL datasets
    all_stages = [set(df["stage"].unique()) for df, _ in datasets]
    common = set(args.stages)
    for s in all_stages:
        common &= s
    stages = [s for s in args.stages if s in common]
    if not stages:
        print("No common stages found across all datasets.")
        return

    print(f"Comparing stages: {stages}")
    plot_efficiency_comparison(datasets, stages, args.outdir)
    plot_walltime_comparison(datasets, stages, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
