"""Export the reduced dataset that the figures actually consume.

WHY THIS EXISTS. The raw campaign output is 5.8 GB and a single 2197-rank timing.csv is
423 MB, because PETSc writes one row per (stage, event, RANK). The figures use the max and
min across ranks, so ~99% of those rows are read once and discarded. `/scratch` is purged on
a timer, so the raw output will not exist in a year — but the reduction is ~16 MB and can
live in git alongside the figures it produces.

The export is built by calling `figdata.load`, NOT by re-parsing the CSVs. That guarantees
the archived numbers are exactly the ones the figures were drawn from, rather than a second
implementation that could drift.

Output, per round:
    data/<round>/manifest.json   provenance: versions, campaigns, when, by what
    data/<round>/runs.csv        one row per (campaign, ranks, replicate)
    data/<round>/stages.csv      one row per (run, stage)
    data/<round>/events.csv      one row per (run, stage, event)
    data/<round>/info.json       full run_info per (campaign, ranks) — solver_stats etc.

Long format throughout: one fact per row, so a new campaign is an append and `git diff`
shows exactly what a re-run changed.

Usage:
    python analysis/export_data.py --round 2026-08_uw3-v3.1.0 \
        --uw3-version v3.1.0 --petsc 3.25.0 \
        --container /g/data/m18/software/containers/underworld3-gadi_v3.1.0.sif
"""

import argparse
import csv
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata

# Every campaign any figure reads, plus the controls that justify a claim. Sub-directory
# campaigns (the Stokes scans) are named by their full relative path so the key is unique.
CAMPAIGNS = [
    "weak-scaling-2026-BASE24-container2",
    "weak-scaling-2026-poisson-analytic",
    "weak-scaling-2026-advdiff-BASE24",
    "weak-scaling-2026-advdiff-meshindep",
    "weak-scaling-2026-checkpoint",
    "weak-scaling-2026-checkpoint-evalfix",
    "weak-scaling-2026-stokes-BASE5-3way/default",
    "weak-scaling-2026-stokes-BASE5-3way/1e-6",
    "weak-scaling-2026-stokes-BASE5-3way/1e-3",
    "weak-scaling-2026-stokes-BASE5-innerscan/1e-4",
    "weak-scaling-2026-stokes-BASE5-innerscan/1e-5",
    "weak-scaling-2026-stokes-BASE5-innerscan/1e-6",
    "weak-scaling-2026-stokes-BASE10",
    "weak-scaling-2026-stokes-BASE10-spread",
]

RUN_COLS = ["campaign", "site", "model", "nprocs", "res", "replicate", "ranks_per_node"]
STAGE_COLS = ["campaign", "nprocs", "replicate", "stage",
              "time", "time_rank0", "messages", "reductions", "flop"]
EVENT_COLS = ["campaign", "nprocs", "replicate", "stage", "event",
              "count_max", "count_min", "time_rank0"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", required=True,
                    help="round label, e.g. 2026-08_uw3-v3.1.0")
    ap.add_argument("--uw3-version", default="unknown")
    ap.add_argument("--petsc", default="unknown")
    ap.add_argument("--container", default="unknown")
    ap.add_argument("--site", default="gadi", choices=["gadi", "setonix"],
                    help="which machine these campaigns ran on")
    ap.add_argument("--cores-per-node", type=int, default=None,
                    help="full node occupancy (Gadi 48, Setonix 128). Recorded so the "
                         "packed baseline is not mis-derived on a second machine.")
    ap.add_argument("--queue", default="normal")
    # A round characterises a RELEASE, so cross-release comparison is the point — and it is
    # only valid while the benchmark measures the same work. Bump this whenever a model
    # script changes what it measures, so a later comparison can refuse to mix them.
    ap.add_argument("--benchmark-version", type=int, default=1)
    ap.add_argument("--benchmark-note", default="")
    args = ap.parse_args()

    cores_per_node = args.cores_per_node or {"gadi": 48, "setonix": 128}[args.site]

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(repo, "data", args.round)
    os.makedirs(outdir, exist_ok=True)

    runs, stages, events, info = [], [], [], {}
    missing = []

    for campaign in CAMPAIGNS:
        data = figdata.load(os.path.join(repo, campaign))
        if not data:
            missing.append(campaign)
            continue
        info[campaign] = {}
        for n in sorted(data):
            entry = data[n]
            info[campaign][str(n)] = entry["info"]
            rpn = (entry["info"].get("placement") or {}).get("ranks_per_node_max")
            for i, rep in enumerate(entry["reps"], start=1):
                runs.append({
                    "campaign": campaign,
                    "site": args.site,
                    "model": rep["info"].get("model", ""),
                    "nprocs": n,
                    "res": rep["info"].get("res", ""),
                    "replicate": i,
                    "ranks_per_node": rpn if rpn else "",
                })
                for s, v in sorted(rep["stages"].items()):
                    stages.append({
                        "campaign": campaign, "nprocs": n, "replicate": i, "stage": s,
                        "time": v.get("time", 0.0),
                        "time_rank0": v.get("time_rank0", 0.0),
                        "messages": v.get("messages", 0.0),
                        "reductions": v.get("reductions", 0.0),
                        "flop": v.get("flop", 0.0),
                    })
                for (s, e), v in sorted(rep["events"].items()):
                    # PETSc registers every event it knows about, called or not, and 92% of
                    # the rows are zero on every rank. They carry no information, and
                    # `figdata.event()` already returns 0 for a missing key, so dropping
                    # them is behaviour-preserving and takes the export 34 MB -> ~3 MB.
                    if not v.get("max") and not v.get("min"):
                        continue
                    events.append({
                        "campaign": campaign, "nprocs": n, "replicate": i,
                        "stage": s, "event": e,
                        "count_max": v.get("max", 0),
                        "count_min": v.get("min", 0),
                        "time_rank0": v.get("time", 0.0),
                    })

    _write_csv(os.path.join(outdir, "runs.csv"), RUN_COLS, runs)
    _write_csv(os.path.join(outdir, "stages.csv"), STAGE_COLS, stages)
    _write_csv(os.path.join(outdir, "events.csv"), EVENT_COLS, events)

    with open(os.path.join(outdir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2, sort_keys=True)

    manifest = {
        "round": args.round,
        "generated": datetime.date.today().isoformat(),
        "uw3_version": args.uw3_version,
        # Bumped when a model script changes what it measures. Rounds with different
        # benchmark versions may not be compared on absolute numbers; curve shapes often
        # still are. See README.
        "benchmark_version": args.benchmark_version,
        "benchmark_note": args.benchmark_note,
        "sites": {
            args.site: {
                "petsc_version": args.petsc,
                "container": args.container,
                "queue": args.queue,
                # Full node occupancy. Recorded rather than assumed: it sets which job is
                # the packed baseline, and it differs per machine (Gadi 48, Setonix 128).
                "cores_per_node": cores_per_node,
            }
        },
        # Recorded so a later round can tell whether a difference is UW3's or the machine's.
        # "unknown" is deliberate where we never captured it — an explicit gap beats a
        # confident-looking guess.
        "campaigns": sorted(info),
        "missing_campaigns": missing,
        "n_runs": len(runs),
        "raw_output_note": (
            "Raw PETSc logs (5.8 GB, one 423 MB file at 2197 ranks) lived under "
            "/scratch/el06/jg0883 on Gadi and are subject to scratch purging. This "
            "reduction is the archival copy: it holds every number the figures use."
        ),
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(os.path.getsize(os.path.join(outdir, f)) for f in os.listdir(outdir))
    print(f"exported to {outdir}")
    print(f"  {len(runs)} runs, {len(stages)} stage rows, {len(events)} event rows")
    print(f"  {total/1e6:.1f} MB total")
    if missing:
        print(f"  MISSING (not on disk): {', '.join(missing)}")


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
