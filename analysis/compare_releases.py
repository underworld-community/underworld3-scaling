"""Release-over-release attribution tables, in markdown, from the exported rounds.

Two tables per model, both from the warm replicates figdata selects:

  1. every stage the two campaigns share — previous and current wall time (max over
     ranks, mean over replicates) and the change, at every rank count both ran
  2. the PETSc events behind a stage at one rank count — rank-0 time and call count in
     each release, the largest movers first. This is where a stage delta gets its cause:
     a PCApply that costs 2x per call under different GAMG defaults, a KDTree.query count
     that fell by two thirds, a DMPlexDistribute that did not move at all.

The campaign pairs are fig_release.PAIRS, so the tables describe exactly the figure.

Usage:
    python analysis/compare_releases.py                      # all models, largest common ranks
    python analysis/compare_releases.py --ranks 64 --top 15
    python analysis/compare_releases.py --events poisson:343:steady_solves advdiff:343:steady_solves
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata
from fig_release import PAIRS, TITLE, version

EVENT_STAGES = ["mesh_setup", "first_solve", "steady_solves"]


def pct(curr, prev):
    return f"{(curr / prev - 1) * 100:+.0f}%" if prev else "n/a"


def stage_table(model, prev, curr, pv, cv):
    ranks = [n for n in sorted(curr) if n in prev]
    stages = [s for s in ("mesh_setup", "solver_setup", "first_solve", "steady_solves",
                          "error_analysis")
              if all(s in prev[n]["stages"] and s in curr[n]["stages"] for n in ranks)]
    out = [f"### {TITLE[model]} — stage wall time, {pv} → {cv}", "",
           "| stage | " + " | ".join(str(n) for n in ranks) + " |",
           "|---|" + "---|" * len(ranks)]
    for s in stages:
        cells = []
        for n in ranks:
            p, c = prev[n]["stages"][s]["time"], curr[n]["stages"][s]["time"]
            cells.append(f"{p:.1f} → {c:.1f} ({pct(c, p)})")
        out.append(f"| `{s}` | " + " | ".join(cells) + " |")
    reps = lambda d: "/".join(str(d[n]["n_replicates"]) for n in ranks)
    cold = sum(d[n]["n_cold"] for d in (prev, curr) for n in d)
    out.append("")
    out.append(f"Seconds, max over ranks, mean over warm replicates ({pv}: {reps(prev)}; "
               f"{cv}: {reps(curr)} per point{f'; {cold} cold-JIT replicates excluded' if cold else ''}).")
    return out


def event_table(model, prev, curr, pv, cv, n, stage, top):
    pe = {e: v for (s, e), v in prev[n]["events"].items() if s == stage}
    ce = {e: v for (s, e), v in curr[n]["events"].items() if s == stage}
    names = set(pe) | set(ce)
    # Largest movers first: the absolute change in rank-0 time decides what mattered.
    order = sorted(names, key=lambda e: -abs(ce.get(e, {}).get("time", 0.0)
                                             - pe.get(e, {}).get("time", 0.0)))[:top]
    ps, cs = prev[n]["stages"][stage], curr[n]["stages"][stage]
    out = [f"### {TITLE[model]} — events in `{stage}` at {n} ranks "
           f"(stage {ps['time']:.1f} → {cs['time']:.1f} s, {pct(cs['time'], ps['time'])})", "",
           f"| event | calls {pv} | calls {cv} | rank-0 s {pv} | rank-0 s {cv} | change |",
           "|---|---|---|---|---|---|"]
    for e in order:
        p, c = pe.get(e, {}), ce.get(e, {})
        pt, ct = p.get("time", 0.0), c.get("time", 0.0)
        out.append(f"| `{e}` | {p.get('max', 0)} | {c.get('max', 0)} | {pt:.2f} | {ct:.2f} | "
                   f"{pct(ct, pt) if pt else ('new' if ct else '')} |")
    out.append("")
    out.append("Calls are the max over ranks; times are rank 0, first warm replicate. "
               "`PCApply` etc. are per-call costs × counts, so a count change and a per-call "
               "change read differently.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    ap.add_argument("--ranks", type=int, default=None,
                    help="rank count for the event tables (default: largest both ran)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--events", nargs="*", default=None,
                    help="model:ranks:stage triples; default every model at --ranks for "
                         + ", ".join(EVENT_STAGES))
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines = []
    for model in args.models:
        prev_c, curr_c = PAIRS[model]["prev"], PAIRS[model]["curr"]
        if not prev_c:
            continue   # first appearance: nothing to compare against
        prev = figdata.load(os.path.join(repo, prev_c))
        curr = figdata.load(os.path.join(repo, curr_c))
        if not prev or not curr:
            print(f"_{TITLE[model]}: no data for both rounds yet._\n")
            continue
        pv, cv = version(prev_c), version(curr_c)
        lines += stage_table(model, prev, curr, pv, cv) + [""]

        if args.events is None:
            n = args.ranks or max(n for n in curr if n in prev)
            triples = [(model, n, s) for s in EVENT_STAGES]
        else:
            triples = [(m, int(n), s) for m, n, s in (t.split(":") for t in args.events)
                       if m == model]
        for m, n, s in triples:
            if n not in prev or n not in curr:
                lines.append(f"_{TITLE[m]} at {n} ranks: not in both releases._\n")
                continue
            lines += event_table(m, prev, curr, pv, cv, n, s, args.top) + [""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
