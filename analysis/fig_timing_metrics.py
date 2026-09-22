"""Two metrics that classify every timed event as work, waiting, or serial.

THE PROBLEM. A PETSc stage time is the max across ranks. That cannot distinguish
"all 2197 ranks worked for 2328 s" from "rank 0 worked for 2328 s while 2196 waited".
Both report 2328 s. Our reduced dataset keeps only the max, so the distinction was
invisible until this analysis went back to the raw per-rank CSV.

THE METRICS, both free from data already written (the CSV has a Rank column):

    idle%     = 100 * (1 - mean/max)   across ranks
    n_working = #ranks above 50% of max

n_working is the decisive one. n_working == 1 means one rank did the work and every
other rank was blocked. idle% alone is NOT sufficient: when ALL ranks wait equally
inside a collective, mean == max and idle% reads 0 -- see mesh_setup/SFBcastEnd, which
is 2196 ranks blocking, at idle 4%. Waiting is only visible as idle% when it is
ASYMMETRIC. The reliable signature of a serial phase is the PAIR: an event with
n_working == 1 on rank 0, and a matching SFBcast/barrier event where everyone else sits.

Usage:  python analysis/fig_timing_metrics.py [--outdir figures]
"""

import argparse
import csv
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import matplotlib.pyplot as plt

CASES = [
    ("Poisson",    "weak-scaling-2026-BASE24-container2"),
    ("Stokes",     "weak-scaling-2026-stokes-BASE5-3way"),
    ("AdvDiff",    "weak-scaling-2026-advdiff-BASE24"),
    ("Checkpoint", "weak-scaling-2026-h5mesh"),
]


def largest_run(campaign):
    """Path to the highest-rank timing.csv, and its rank count."""
    best, bestn = None, -1
    for f in glob.glob(f"{campaign}/**/timing.csv", recursive=True):
        n = None
        try:
            n = json.load(open(os.path.join(os.path.dirname(f), "run_info.json")))["nprocs"]
        except Exception:
            # Older campaigns predate run_info.json; the CSV header's last field
            # is the rank count.
            try:
                with open(f) as fh:
                    n = int(fh.readline().rstrip("\n").split(",")[-1])
            except Exception:
                continue
        if n and n > bestn:
            bestn, best = n, f
    return best, bestn


def analyse(path):
    stage_t = collections.defaultdict(dict)
    ev = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if len(row) < 9:
                continue
            try:
                rank, cnt, t = int(row[2]), int(float(row[3])), float(row[4])
            except ValueError:
                continue
            if row[1] == "summary":
                stage_t[row[0]][rank] = t
            elif cnt > 0 and t > 0:
                ev[row[0]][row[1]][rank] = t
    return stage_t, ev


def collect(campaign):
    path, nranks = largest_run(campaign)
    if not path:
        return None
    stage_t, ev = analyse(path)
    total = sum(max(v.values()) for k, v in stage_t.items() if k != "Main Stage")
    stage_max = {k: max(v.values()) for k, v in stage_t.items()}
    rows = []
    for st, evs in ev.items():
        if st == "Main Stage":
            continue
        for e, m in evs.items():
            mx = max(m.values())
            if mx < 0.01 * total:
                continue
            mean = sum(m.values()) / len(m)
            nw = sum(1 for v in m.values() if v > 0.5 * mx)
            rows.append({"stage": st, "event": e, "max": mx, "mean": mean,
                         "nwork": nw, "nranks": len(m)})
    return {"total": total, "nranks": nranks, "rows": rows, "stage_max": stage_max,
            "run": os.path.basename(os.path.dirname(path))}


def serial_seconds(d, stage_max):
    """Wall seconds where ONE rank works — a conservative LOWER BOUND.

    PETSc events NEST (DMPlexSymmetrize inside DMPlexInterp inside DMPlexCreateGmsh
    inside DMPlexCrFromFile), and the CSV carries no call tree, so summing serial
    events double-counts: doing that naively gave Poisson 126% of its own runtime.

    Rule: take only the LARGEST serial event per stage. A parent contains its
    children, so this can never exceed the true serial time, and it cannot
    double-count. It UNDERSTATES whenever a stage has two independent serial
    phases — Poisson's mesh_setup has both a serial Gmsh read (DMPlexCrFromFile,
    2328 s) and a serial HDF5 write (DMView, 1080 s), and only the read is counted.
    """
    per_stage = {}
    for r in d["rows"]:
        if r["nwork"] > max(1, r["nranks"] // 100):
            continue
        cur = per_stage.get(r["stage"])
        if cur is None or r["max"] > cur["max"]:
            per_stage[r["stage"]] = r
    kept = sorted(per_stage.values(), key=lambda r: -r["max"])
    total = 0.0
    for r in kept:
        # Never let a stage contribute more than its own wall time.
        total += min(r["max"], stage_max.get(r["stage"], r["max"]))
    return total, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)

    data = {}
    for label, camp in CASES:
        d = collect(camp)
        if d:
            data[label] = d
            print(f"{label:<12}{d['nranks']:>6} ranks  total {d['total']:>9.1f} s  "
                  f"({len(d['rows'])} events >1%)")

    # figstyle.style() is written for log-log rank-vs-time line panels: it forces
    # set_xscale("log") and rank tick labels. Neither panel here is that, so these are
    # laid out directly and only figstyle's colours, caption and finish are reused.
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    order = [l for l, _ in CASES if l in data]

    # ---- Panel 1: how much of each run is one rank working ----------------- #
    labels, ser_frac, notes = [], [], []
    for label in order:
        d = data[label]
        s_sec, kept = serial_seconds(d, d["stage_max"])
        labels.append(f"{label}\n{d['nranks']} ranks")
        ser_frac.append(100.0 * s_sec / d["total"])
        notes.append((label, s_sec, d["total"], kept))

    y = list(range(len(labels)))
    ax[0].barh(y, ser_frac, color="#c0392b", height=0.62,
               label="one rank works, the rest wait")
    ax[0].barh(y, [100.0 - v for v in ser_frac], left=ser_frac, color="#2980b9",
               height=0.62, label="all ranks working")
    for i, (v, (_, s_sec, tot, _)) in enumerate(zip(ser_frac, notes)):
        ax[0].text(v / 2, i, f"{v:.0f}%", va="center", ha="center",
                   color="white", fontsize=11, fontweight="bold")
        ax[0].text(99, i, f"{tot:,.0f} s total  ", va="center", ha="right",
                   color="white", fontsize=8.5)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=9)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("% of run wall time")
    ax[0].set_title("Share of the run that is serial\n(lower bound)", fontsize=11)
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9,
                 frameon=False)
    ax[0].grid(axis="x", alpha=0.3)
    ax[0].set_axisbelow(True)

    # ---- Panel 2: the classifier itself ------------------------------------ #
    # x is the FRACTION of ranks doing the work, so runs at different rank counts are
    # directly comparable: 1/N at the left is serial, 1.0 at the right is collective.
    for label, colour, marker in zip(order, ["C1", "C0", "C4", "C2"], "os^D"):
        d = data[label]
        xs = [r["nwork"] / r["nranks"] for r in d["rows"]]
        ys = [100.0 * r["max"] / d["total"] for r in d["rows"]]
        ax[1].scatter(xs, ys, s=40, c=colour, marker=marker, alpha=0.8,
                      edgecolors="white", linewidths=0.6,
                      label=f"{label} ({d['nranks']} ranks)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(3e-4, 2.0)
    ax[1].axvspan(3e-4, 0.02, color="#c0392b", alpha=0.08)
    ax[1].annotate("SERIAL\none rank works", xy=(0.02, 0.06), xycoords="axes fraction",
                   fontsize=9, color="#c0392b", ha="left", va="bottom")
    ax[1].annotate("COLLECTIVE\nall ranks work", xy=(0.98, 0.06), xycoords="axes fraction",
                   fontsize=9, color="#2980b9", ha="right", va="bottom")
    ax[1].set_xlabel("fraction of ranks above 50% of max time")
    ax[1].set_ylabel("% of run wall time")
    ax[1].set_title("Every event worth >1%, by how many ranks do the work", fontsize=11)
    ax[1].legend(loc="upper left", fontsize=8.5)
    ax[1].grid(alpha=0.3)
    ax[1].set_axisbelow(True)

    outdir = os.path.join(repo, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "timing_metrics.png")

    fs.caption(fig,
               "Both panels use the Rank column of timing.csv, which records every event on "
               "every rank. LEFT is a conservative lower bound — the largest serial event per "
               "stage, because PETSc events nest and the CSV carries no call tree; summing "
               "them naively gave Poisson 126% of its own runtime. UW3 builds the whole mesh "
               "on rank 0 before DMPlexDistribute, so reading a Gmsh .msh is serial by "
               "construction; Checkpoint is the only campaign already loading a PETSc .msh.h5, "
               "which skips it. RIGHT is the classifier. idle% is deliberately not the test: "
               "when all ranks block equally inside one collective, mean equals max and idle "
               "reads 0.")
    fs.finish(fig, "Underworld3 — classifying timed events as work, waiting, or serial", out)

    print(f"\n{'campaign':<12}{'serial (s)':>12}{'total (s)':>12}{'serial %':>10}")
    for label, s, tot, kept in notes:
        print(f"{label:<12}{s:>12.1f}{tot:>12.1f}{100*s/tot:>10.1f}")
        for k in kept[:4]:
            print(f"    {k['stage']}/{k['event']:<34}{k['max']:>9.1f} s"
                  f"  ({k['nwork']}/{k['nranks']} ranks)")


if __name__ == "__main__":
    main()
