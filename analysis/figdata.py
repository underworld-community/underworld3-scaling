"""Load campaign output for the figures.

Deliberately lighter than `collect_results.py`: that walks every rank row to build the full
results table, which takes minutes on a 2197-rank campaign. The figures need per-stage wall
times, a handful of event counts, and the rank-imbalance spread — all obtainable in one pass.

Two things the PETSc CSV will mislead you about if read naively:

- Stage wall time must be the MAX across ranks, not rank 0's. The slowest rank determines
  when the next barrier releases, so rank 0 can finish early and understate the stage.
- Event counts differ enormously BETWEEN ranks when AMG consolidates coarse levels onto a
  subset (MatMult varied 138 to 8658 within a single Poisson run). Rank 0 is one arbitrary
  sample, so both the max and the min are kept — their ratio is the load imbalance.
"""

import csv
import glob
import hashlib
import json
import os
import pickle
import statistics as st
from collections import defaultdict


def _read_run(run_dir):
    """One run: stage wall times (max over ranks) and per-event count spread."""
    stages, events = {}, defaultdict(lambda: {"max": 0, "min": None, "time": 0.0})
    path = os.path.join(run_dir, "timing.csv")
    if not os.path.exists(path):
        return None

    with open(path) as fh:
        for raw in fh:
            f = raw.split(",")
            if len(f) < 9 or f[0] in ("Stage Name", "Main Stage"):
                continue
            try:
                rank, count, tsec = int(f[2]), int(float(f[3])), float(f[4])
            except ValueError:
                continue
            stage, event = f[0], f[1]

            if event == "summary":
                prev = stages.get(stage)
                # Max across ranks: the slowest rank sets the pace.
                if prev is None or tsec > prev["time"]:
                    keep_rank0 = prev.get("time_rank0", 0.0) if prev else 0.0
                    stages[stage] = {
                        "time": tsec,
                        "time_rank0": keep_rank0,
                        "messages": float(f[5]),
                        "reductions": float(f[7]),
                        "flop": float(f[8]),
                    }
                # Kept separately because component decompositions subtract a rank-0 event
                # time from a stage total, and mixing rank 0 with the across-rank max would
                # bias the remainder.
                if rank == 0:
                    stages.setdefault(stage, {"time": tsec, "messages": 0.0,
                                              "reductions": 0.0, "flop": 0.0})
                    stages[stage]["time_rank0"] = tsec
            else:
                e = events[(stage, event)]
                e["max"] = max(e["max"], count)
                e["min"] = count if e["min"] is None else min(e["min"], count)
                if rank == 0:
                    e["time"] = tsec

    # poisson-scaling.py writes errors.json; the others write run_info.json. Both carry
    # model/res/nprocs, so either will do.
    info = {}
    for name in ("run_info.json", "errors.json"):
        ipath = os.path.join(run_dir, name)
        if os.path.exists(ipath):
            with open(ipath) as fh:
                info = json.load(fh)
            break

    return {"stages": stages, "events": dict(events), "info": info}


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _cache_key(paths):
    """Campaign identity: which runs exist and when they last changed.

    A new replicate, a re-downloaded run or an edited CSV all move the key, so a stale
    cache cannot outlive the data it came from. Cheap because stat() is cheap — reading
    the CSVs is what costs, and a 2197-rank campaign takes minutes.
    """
    h = hashlib.sha1()
    for p in paths:
        try:
            h.update(f"{p}:{os.path.getmtime(p)}".encode())
        except OSError:
            pass
    return h.hexdigest()


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("UW3_SCALING_DATA", os.path.join(REPO, "data"))


def _export_rounds():
    """Exported rounds, newest label first — the open round, then frozen ones.

    Archived rounds are searched as well as the open one, so a figure in reports/<round>/
    still regenerates after its data has moved there. The open round is listed first, so a
    campaign present in both resolves to the current numbers.

    Round labels sort usefully (YYYY-MM_...), hence reverse=True within each location.
    """
    return (sorted(glob.glob(os.path.join(DATA_DIR, "*", "runs.csv")), reverse=True) +
            sorted(glob.glob(os.path.join(REPO, "reports", "*", "data", "runs.csv")),
                   reverse=True))


def _load_from_export(campaign):
    """Rebuild the load() structure from an exported round, or None if not exported.

    The archived CSVs are the long-format reduction of the raw PETSc logs. Reading them
    lets every figure regenerate from what is committed, with no dependency on 5.8 GB of
    scratch output that will be purged.
    """
    for runs_csv in _export_rounds():
        root = os.path.dirname(runs_csv)
        with open(runs_csv) as fh:
            rows = [r for r in csv.DictReader(fh) if r["campaign"] == campaign]
        if not rows:
            continue

        with open(os.path.join(root, "info.json")) as fh:
            info = json.load(fh).get(campaign, {})

        # (nprocs, replicate) -> the run being rebuilt.
        runs = defaultdict(lambda: {"stages": {}, "events": {}, "info": {}})
        for r in rows:
            key = (int(r["nprocs"]), int(r["replicate"]))
            runs[key]["info"] = info.get(r["nprocs"], {})

        for name, target in (("stages.csv", "stages"), ("events.csv", "events")):
            with open(os.path.join(root, name)) as fh:
                for r in csv.DictReader(fh):
                    if r["campaign"] != campaign:
                        continue
                    key = (int(r["nprocs"]), int(r["replicate"]))
                    if key not in runs:
                        continue
                    if target == "stages":
                        runs[key]["stages"][r["stage"]] = {
                            "time": float(r["time"]),
                            "time_rank0": float(r["time_rank0"]),
                            "messages": float(r["messages"]),
                            "reductions": float(r["reductions"]),
                            "flop": float(r["flop"]),
                        }
                    else:
                        runs[key]["events"][(r["stage"], r["event"])] = {
                            "max": int(r["count_max"]),
                            "min": int(r["count_min"]),
                            "time": float(r["time_rank0"]),
                        }

        grouped = defaultdict(list)
        for (n, _), run in sorted(runs.items()):
            grouped[n].append(run)
        return _aggregate(grouped)
    return None


def load(campaign_dir, pattern="*_out/*"):
    """All runs in a campaign, grouped by rank count and averaged over replicates.

    Returns {nprocs: {...}} with stage times averaged, event counts taken from the first
    replicate (they are deterministic under a fixed-work protocol), and run_info carried
    through for placement, checkpoint bytes and solver stats.

    Prefers an exported round under `data/` — that is the archival copy and the only one
    that survives scratch purging. Falls back to parsing the raw campaign directories.
    """
    campaign = os.path.relpath(os.path.abspath(campaign_dir), REPO).rstrip("/")
    from_export = _load_from_export(campaign)
    if from_export:
        return from_export

    dirs = sorted(glob.glob(os.path.join(campaign_dir, pattern)))

    key = _cache_key([os.path.join(d, "timing.csv") for d in dirs])
    cache = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass   # A damaged cache must never be worse than no cache.

    runs = defaultdict(list)
    for d in dirs:
        r = _read_run(d)
        if r and r["info"].get("nprocs"):
            runs[r["info"]["nprocs"]].append(r)

    out = _aggregate(runs)

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    except OSError:
        pass   # Caching is an optimisation; failing to write one must not fail the figure.
    return out


def _aggregate(runs):
    """{nprocs: [run, ...]} -> the public structure. Shared by both load paths.

    Kept as one function so the exported CSVs and the raw logs cannot drift into producing
    subtly different numbers.
    """
    out = {}
    for n, reps in sorted(runs.items()):
        stage_names = set().union(*(set(r["stages"]) for r in reps))
        out[n] = {
            "n_replicates": len(reps),
            "info": reps[0]["info"],
            "events": reps[0]["events"],
            # The raw replicates are kept so error bars can be derived from the same runs
            # that produced the mean. A panel plotting a DERIVED quantity — a stage total
            # minus an event time, say — cannot recover its spread from an average that has
            # already collapsed it, so `bounds()` re-extracts per replicate instead.
            "reps": reps,
            "stages": {
                s: {
                    "time": st.mean([r["stages"][s]["time"] for r in reps if s in r["stages"]]),
                    "time_rank0": st.mean([r["stages"][s].get("time_rank0", 0.0)
                                           for r in reps if s in r["stages"]]),
                    "spread": _spread([r["stages"][s]["time"] for r in reps if s in r["stages"]]),
                    "messages": reps[0]["stages"].get(s, {}).get("messages", 0.0),
                    "reductions": reps[0]["stages"].get(s, {}).get("reductions", 0.0),
                    "flop": reps[0]["stages"].get(s, {}).get("flop", 0.0),
                }
                for s in stage_names
            },
        }
    return out


def _spread(values):
    """(max-min)/mean as a percentage — how much the replicates disagree."""
    if len(values) < 2 or not st.mean(values):
        return 0.0
    return (max(values) - min(values)) / st.mean(values) * 100.0


def bounds(data, extract):
    """(ranks, lo, hi) — the full min-to-max range across replicates.

    `extract` maps ONE raw replicate to a number, so this works for derived quantities as
    well as plain stage times: the advdiff decomposition subtracts an event time from a
    stage total, and passing that whole expression here gives its spread directly.

    The range is deliberately not a standard error. With two or three runs a standard error
    is not meaningfully estimated, whereas min-to-max states exactly what was observed.
    Single-replicate points return lo == hi == the value, which draws a zero-length bar —
    honest, since it marks "not repeated" rather than "no variation".
    """
    ns = sorted(data)
    lo, hi = [], []
    for n in ns:
        vals = [v for v in (extract(r) for r in data[n]["reps"]) if v is not None]
        lo.append(min(vals) if vals else 0.0)
        hi.append(max(vals) if vals else 0.0)
    return ns, lo, hi


def stage_bounds(data, stage, key="time"):
    """(ranks, lo, hi) for one stage — the common case of `bounds`."""
    return bounds(data, lambda r: r["stages"].get(stage, {}).get(key))


def series(data, stage, key="time"):
    """(ranks, values) for one stage across the sweep."""
    ns = sorted(data)
    return ns, [data[n]["stages"][stage][key] for n in ns]


def event(data, stage, name, field="max"):
    """(ranks, values) for one PETSc event. field: max | min | time."""
    ns = sorted(data)
    return ns, [data[n]["events"].get((stage, name), {}).get(field, 0) for n in ns]


def ranks_per_node(data):
    """Recorded placement, or None where the run predates the placement capture."""
    return [(data[n]["info"].get("placement") or {}).get("ranks_per_node_max")
            for n in sorted(data)]
