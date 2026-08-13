"""
Collect timing and metadata from a scaling test output tree into results.csv.

Usage:
    python analysis/collect_results.py --outdir /scratch/el06/jg0883/Weak_uw3_Jun2026_...

The script walks the directory tree, finds timing.csv and run_info.json files,
extracts per-stage wall times from the PETSc CSV, and writes a single
results.csv keyed by (model, stage, nprocs, res, run_idx, scaling).

IMPORTANT — PETSc CSV format note:
    The PETSc ASCII_CSV output format (PETSc.Viewer.Format.ASCII_CSV) varies
    slightly between PETSc versions and stage configurations.  This parser
    targets the format produced by PETSc 3.20+ where each section begins with
    a "Stage:" header line followed by comma-separated event rows.

    If the format differs, run:
        python analysis/collect_results.py --outdir <dir> --dump-raw timing.csv
    on one of your output files to inspect the raw text and adjust _parse_petsc_csv()
    accordingly before processing the full tree.
"""

import os
import re
import json
import csv
import argparse
import sys
from pathlib import Path
from collections import defaultdict


# --------------------------------------------------------------------------- #
# PETSc CSV parser                                                             #
# --------------------------------------------------------------------------- #

def _float_or(value, default):
    """Parse a PETSc CSV cell as float, returning `default` if it is blank or malformed."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_petsc_csv(filepath):
    """
    Parse a PETSc ASCII_CSV timing file.

    Returns a dict:
        { stage_name: { event_name: {"time_s": float, "count": int} },
          "__nprocs__": int_or_None }

    Handles two formats:

    PETSc 3.25.0 flat-table (single CSV with Name / Event Name columns):
        Stage: Name,Event Name,Rank,Count,Time,...,<nprocs>
        mesh_setup,MatMult,0,10,1.234,...
        analytical_setup,MatSolve,0,5,2.345,...

    Legacy PETSc 3.20 per-stage sections:
        Stage: mesh_setup
        Event,Count,Time (s),...
        MatCreate,1,0.002,...

    The "__nprocs__" sentinel key carries the process count from the flat
    header (last column); callers should pop it before iterating stages.
    """
    stages = {}
    current_stage = "main"
    header = None
    flat_format = False
    flat_header = None
    nprocs_from_header = None

    try:
        with open(filepath, "r") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                # Detect PETSc 3.25.0 flat-table header:
                # "Stage Name,Event Name,Rank,Count,Time,...,<nprocs>"
                if line.startswith("Stage Name,") and "Event Name" in line:
                    flat_format = True
                    flat_header = [h.strip() for h in line.split(",")]
                    try:
                        nprocs_from_header = int(flat_header[-1])
                    except (ValueError, IndexError):
                        pass
                    continue

                if flat_format:
                    if not flat_header or "," not in line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    row = dict(zip(flat_header, parts))
                    stage_name = row.get("Stage Name", "").strip()
                    event_name = row.get("Event Name", "unknown").strip()
                    if not stage_name or not event_name:
                        continue
                    time_val = _float_or(row.get("Time"), None)
                    if time_val is None:
                        continue
                    record = {
                        "time_s":     time_val,
                        "count":      int(_float_or(row.get("Count"), 0)),
                        "flop":       _float_or(row.get("FLOP"), 0.0),
                        "messages":   _float_or(row.get("Num Messages"), 0.0),
                        "reductions": _float_or(row.get("Num Reductions"), 0.0),
                    }
                    # For "summary" events: keep the MAX across all ranks — the slowest
                    # rank determines true parallel wall time (the next global barrier
                    # can't proceed until every rank finishes its local work).
                    # For all other events: use rank 0 only (consistent reference).
                    if event_name == "summary":
                        existing = stages.get(stage_name, {}).get("summary", {})
                        if time_val > existing.get("time_s", -1):
                            stages.setdefault(stage_name, {})["summary"] = record
                    else:
                        # Aggregate across ALL ranks, not just rank 0. With
                        # pc_gamg_repartition the coarse multigrid levels are
                        # consolidated onto a subset of ranks, so per-event counts
                        # differ by up to ~60x between ranks in a single run — rank 0
                        # is one arbitrary sample and misrepresents both the work done
                        # and the load balance.
                        #
                        # `count` stays the MAX so collective events (PCApply, KSPSolve)
                        # still read as the per-rank iteration count; `count_total` and
                        # `count_min` carry the distribution.
                        prev = stages.setdefault(stage_name, {}).get(event_name)
                        if prev is None:
                            record.update({
                                "count_total": record["count"],
                                "count_min":   record["count"],
                                "nranks":      1,
                            })
                            stages[stage_name][event_name] = record
                        else:
                            prev["time_s"]       = max(prev["time_s"], time_val)
                            prev["count"]        = max(prev["count"], record["count"])
                            prev["count_min"]    = min(prev["count_min"], record["count"])
                            prev["count_total"] += record["count"]
                            prev["flop"]        += record["flop"]
                            prev["messages"]    += record["messages"]
                            prev["reductions"]  += record["reductions"]
                            prev["nranks"]      += 1
                    continue

                # Legacy PETSc 3.20 per-stage format
                m = re.match(r"Stage[:\s]+(.+)", line, re.IGNORECASE)
                if m:
                    current_stage = m.group(1).strip().rstrip(":")
                    stages.setdefault(current_stage, {})
                    header = None
                    continue

                if line.lower().startswith("event"):
                    header = [h.strip() for h in line.split(",")]
                    continue

                if header and "," in line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < len(header):
                        continue
                    row = dict(zip(header, parts))
                    event = row.get("Event") or row.get("event", "unknown")
                    time_val = None
                    for key in ("Time (s)", "Time", "Wall Time", "time"):
                        if key in row:
                            try:
                                time_val = float(row[key])
                            except ValueError:
                                pass
                            break
                    count_val = None
                    for key in ("Count", "count", "Calls"):
                        if key in row:
                            try:
                                count_val = int(float(row[key]))
                            except (ValueError, KeyError):
                                pass
                            break
                    if event and time_val is not None:
                        stages.setdefault(current_stage, {})[event] = {
                            "time_s": time_val,
                            "count": count_val or 0,
                            "flop": 0.0,
                            "messages": 0.0,
                            "reductions": 0.0,
                        }
    except Exception as exc:
        print(f"  WARNING: could not parse {filepath}: {exc}", file=sys.stderr)
        return {}

    stages["__nprocs__"] = nprocs_from_header
    return stages


def _stage_total_time(stage_events):
    """
    Return total wall time for a stage.

    PETSc 3.25.0 flat CSV includes a special "summary" event for each stage
    that equals the stage's total wall time at rank 0.  Prefer it when present.
    Fall back to max of all event times for legacy formats.
    """
    if not stage_events:
        return None
    if "summary" in stage_events and stage_events["summary"]["time_s"] > 0:
        return stage_events["summary"]["time_s"]
    times = [v["time_s"] for v in stage_events.values() if v["time_s"] > 0]
    if not times:
        return None
    return max(times)


# --------------------------------------------------------------------------- #
# Solver work metrics                                                          #
# --------------------------------------------------------------------------- #

# Per-event counts worth reporting alongside wall time. PCApply is the useful one:
# its count is (KSP iterations + 1), because the preconditioner is applied once per
# Krylov iteration plus once for the initial residual. PCSetUp and SNESJacobianEval
# count the expensive setup operations — for a linear problem both should be 1, and
# a value of 10 means the nonlinear loop is spinning.
_SOLVER_EVENTS = (
    "SNESSolve",
    "SNESFunctionEval",
    "SNESJacobianEval",
    "KSPSolve",
    "PCSetUp",
    "PCApply",
    "MatMult",
)


def _solver_metrics(stage_events):
    """
    Per-stage solver work metrics extracted from the PETSc event table.

    Returns counts for the events in _SOLVER_EVENTS, plus stage-level FLOP,
    message and reduction totals taken from the "summary" row.

    `ksp_its` is derived as PCApply count - 1 (see _SOLVER_EVENTS note). For
    nested solvers (Stokes fieldsplit) PETSc accumulates inner and outer
    applications into the same counter, so this is total Krylov work across all
    levels rather than the outer iteration count alone — use the `ksp_its_outer`
    value from run_info.json when the outer count specifically is wanted.
    """
    out = {}
    for ev in _SOLVER_EVENTS:
        rec = stage_events.get(ev)
        out[f"n_{ev}"] = rec["count"] if rec else None

    pc_apply = out.get("n_PCApply")
    out["ksp_its"] = (pc_apply - 1) if pc_apply else None

    summary = stage_events.get("summary", {})
    out["flop"]         = summary.get("flop")
    out["n_messages"]   = summary.get("messages")
    out["n_reductions"] = summary.get("reductions")

    # Coarse-grid load imbalance. AMG consolidates coarse levels onto a subset of ranks,
    # so the ranks left out idle at barriers while the rest work. The max/min spread in
    # MatMult calls measures that directly, and it is the mechanism behind weak-scaling
    # efficiency loss at high core counts — 1.0 is balanced, large is not.
    mm = stage_events.get("MatMult")
    if mm and mm.get("count_min"):
        out["matmult_total"]     = mm.get("count_total")
        out["matmult_imbalance"] = mm["count"] / mm["count_min"]
    return out


# --------------------------------------------------------------------------- #
# Directory path metadata parser                                               #
# --------------------------------------------------------------------------- #

# Short output dir names (e.g. "poisson_out") to full script names.
# These are the subdirectory names written by each script's output_dir setting.
_MODEL_DIR_MAP = {
    "poisson":    "poisson-scaling",
    "stokes":     "stokes-scaling",
    "vep":        "vep-scaling",
    "advdiff":    "advdiff-scaling",
    "checkpoint": "checkpoint-scaling",
    "ns":         "ns-scaling",
}


def _metadata_from_path(timing_csv_path, root_dir):
    """
    Extract run metadata from the directory structure.

    Expected layout (set by the model scripts):
        <root>/<model>_out/<scaling>_<params>_res<R>_job<J>_iter<I>/timing.csv

    Falls back to run_info.json or errors.json in the same directory when available.
    """
    d = Path(timing_csv_path).parent

    meta = {
        "model":   "unknown",
        "scaling": "unknown",
        "res":     None,
        "nprocs":  None,
        "job":     None,
        "run_idx": None,
        "integrator": None,
        "theta":   None,
        "monotone": None,
        "solver_stats": {},
        "placement": {},
    }

    # Prefer run_info.json; fall back to errors.json (written by poisson/stokes scripts)
    for json_name in ("run_info.json", "errors.json"):
        info_file = d / json_name
        if info_file.exists():
            try:
                with open(info_file) as fh:
                    info = json.load(fh)
                if info.get("model"):
                    meta["model"] = info["model"]
                if info.get("scaling"):
                    meta["scaling"] = info["scaling"]
                if info.get("res") is not None:
                    meta["res"] = info["res"]
                if info.get("nprocs") is not None:
                    meta["nprocs"] = info["nprocs"]
                meta["integrator"] = info.get("integrator", meta["integrator"])
                meta["theta"]      = info.get("theta",      meta["theta"])
                meta["monotone"]   = info.get("monotone_mode", meta["monotone"])
                # Solver convergence stats, keyed by stage name (see the
                # solver_stats block written by the model scripts).
                meta["solver_stats"] = info.get("solver_stats", {}) or {}
                # Rank placement — ranks_per_node drives memory bandwidth per rank and so
                # is a first-order control on solve time, not metadata.
                meta["placement"] = info.get("placement", {}) or {}
            except Exception:
                pass
            break

    # Parse directory name for fields still missing
    dir_name = d.name
    patterns = [
        (r"res(\d+)",                            "res",     int),
        (r"job(\d+)",                            "job",     int),
        (r"iter(\d+)",                           "run_idx", int),
        # Scaling: "Weak_..." at start OR "_Weak_" anywhere
        (r"(?:^|_)(Weak|Strong|weak|strong)(?=_|$)", "scaling", str),
        (r"integrator(\w+)",                     "integrator", str),
        (r"theta([\d.]+)",                       "theta",   float),
        (r"mono(\w+)",                           "monotone",str),
    ]
    for pattern, key, cast in patterns:
        if meta.get(key) is None:
            m = re.search(pattern, dir_name)
            if m:
                try:
                    meta[key] = cast(m.group(1))
                except ValueError:
                    pass

    # Infer model from parent directory name: "poisson_out" → "poisson-scaling"
    if meta["model"] == "unknown":
        parent = d.parent.name
        m = re.match(r"(.+?)_out", parent)
        if m:
            short = m.group(1)
            meta["model"] = _MODEL_DIR_MAP.get(short, short)

    # Infer nprocs from "Nproc<N>" in any ancestor path component
    if meta["nprocs"] is None:
        for part in d.parts:
            m = re.search(r"[Nn]proc[s]?(\d+)", part)
            if m:
                meta["nprocs"] = int(m.group(1))
                break

    return meta


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate UW3 scaling test timing CSVs into results.csv"
    )
    parser.add_argument("--outdir",   required=True,
                        help="Root output directory from scaling_test_job_launcher.sh")
    parser.add_argument("--out-csv",  default="results.csv",
                        help="Path for the aggregated CSV output (default: results.csv)")
    parser.add_argument("--dump-raw", metavar="FILE",
                        help="If set, pretty-print the raw content of this single timing.csv "
                             "and exit (useful for inspecting PETSc CSV format)")
    args = parser.parse_args()

    # --dump-raw helper
    if args.dump_raw:
        stages = _parse_petsc_csv(args.dump_raw)
        nprocs = stages.pop("__nprocs__", None)
        if nprocs is not None:
            print(f"nprocs (from CSV header): {nprocs}")
        for stage, events in stages.items():
            print(f"\n=== Stage: {stage} ===")
            for ev, vals in events.items():
                print(f"  {ev:50s}  time={vals['time_s']:.4f}s  count={vals['count']}")
        return

    root = Path(args.outdir)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    timing_files = sorted(root.rglob("timing.csv"))
    if not timing_files:
        print(f"No timing.csv files found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(timing_files)} timing.csv files")

    rows = []
    for tf in timing_files:
        print(f"  Processing: {tf}")
        meta = _metadata_from_path(tf, root)
        stages = _parse_petsc_csv(tf)
        nprocs_csv = stages.pop("__nprocs__", None)
        if meta["nprocs"] is None and nprocs_csv is not None:
            meta["nprocs"] = nprocs_csv

        for stage_name, events in stages.items():
            wall_time = _stage_total_time(events)
            row = {
                "model":      meta["model"],
                "scaling":    meta["scaling"],
                "res":        meta["res"],
                "nprocs":     meta["nprocs"],
                "job":        meta["job"],
                "run_idx":    meta["run_idx"],
                "integrator": meta["integrator"],
                "theta":      meta["theta"],
                "monotone":   meta["monotone"],
                "stage":      stage_name,
                "wall_time_s": wall_time,
                "n_events":   len(events),
                "timing_file": str(tf.relative_to(root)),
            }
            row.update(_solver_metrics(events))
            row.update(meta["placement"])

            # Convergence stats recorded by the model script for this stage.
            row.update(meta["solver_stats"].get(stage_name, {}))

            # Derived: cost per Krylov iteration separates parallel efficiency from
            # algorithmic scalability when the solve runs to a tolerance rather than
            # a fixed iteration count.
            its = row.get("ksp_its_total") or row.get("ksp_its")
            if wall_time and its:
                row["time_per_ksp_it_s"] = wall_time / its

            # Cost per matrix-vector product. NOT a normalizer — n_MatMult varies
            # non-monotonically by ~10x across the BASE=24 Poisson campaign, because
            # GAMG coarsens algebraically from the matrix AS PARTITIONED, so each core
            # count gets a different hierarchy depth and different matvecs per V-cycle.
            #
            # Report it as a DIAGNOSTIC: it is how you see that a fixed KSP iteration
            # count does not imply fixed work per iteration. Useful for nested solvers
            # too, where PETSc rolls fieldsplit inner solves into the outer KSPSolve
            # counter (Stokes reads KSPSolve=1, PCApply=1) and MatMult is the only
            # event that reflects the inner work at all.
            n_matmult = row.get("n_MatMult")
            if wall_time and n_matmult:
                row["time_per_matmult_ms"] = wall_time / n_matmult * 1e3
            # FLOP in the PETSc CSV is per-rank, so this is already per-core — do NOT
            # divide by nprocs. Flat across scales means each core keeps doing the same
            # amount of arithmetic per second; a decline is lost compute time.
            if wall_time and row.get("flop"):
                row["gflops_per_core"] = row["flop"] / wall_time / 1e9

            rows.append(row)

        # Also merge memprobe.csv if present
        mp_file = tf.parent / "memprobe.csv"
        if mp_file.exists():
            try:
                with open(mp_file) as fh:
                    for mp_row in csv.DictReader(fh):
                        merged = {
                            "model":      meta["model"],
                            "scaling":    meta["scaling"],
                            "res":        meta["res"],
                            "nprocs":     meta["nprocs"],
                            "job":        meta["job"],
                            "run_idx":    meta["run_idx"],
                            "integrator": meta["integrator"],
                            "stage":      mp_row.get("stage", ""),
                            "rss_delta_mib": mp_row.get("rss_delta_mib", ""),
                            "wall_time_s": None,
                            "timing_file": str(mp_file.relative_to(root)),
                        }
                        rows.append(merged)
            except Exception as exc:
                print(f"    WARNING: could not read {mp_file}: {exc}", file=sys.stderr)

    if not rows:
        print("No data extracted. Check PETSc CSV format with --dump-raw.", file=sys.stderr)
        sys.exit(1)

    # Write results.csv
    fieldnames = [
        "model", "scaling", "res", "nprocs", "job", "run_idx",
        "integrator", "theta", "monotone",
        "stage", "wall_time_s", "n_events", "rss_delta_mib",
        "n_nodes", "ranks_per_node_max", "ranks_per_node_min",
        # Solver convergence (from run_info.json, where the script recorded it)
        "snes_its", "snes_reason", "ksp_its_outer", "ksp_its_total", "ksp_reason",
        "ksp_rnorm_final", "snes_fnorm_final",
        # Solver work (from the PETSc event table)
        "ksp_its", "time_per_ksp_it_s", "gflops_per_core",
        "matmult_total", "matmult_imbalance",
        "flop", "n_messages", "n_reductions",
        *(f"n_{ev}" for ev in _SOLVER_EVENTS),
        "timing_file",
    ]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
    print("Next step: python analysis/plot_scaling.py --csv results.csv --model <model>")


if __name__ == "__main__":
    main()
