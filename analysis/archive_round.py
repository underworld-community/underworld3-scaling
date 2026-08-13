"""Freeze the current round into reports/<round>/ so a new one can start.

A round is one environment: a UW3 version, a PETSc version, a container image. It is NOT a
calendar period. Campaigns run against the same environment append to the open round; the
round closes when the environment changes, because that is what makes older numbers
incomparable rather than merely older.

Freezing moves three things into a single self-contained directory:

    reports/<round>/report.md    the README exactly as it stood
    reports/<round>/figures/     the figures it refers to
    reports/<round>/data/        the reduced dataset they were built from

NOTHING IS DELETED. The dataset is MOVED rather than copied, so there is one authoritative
copy instead of two that can drift; git records it as a rename. Afterwards the top-level
README.md, figures/ and data/ belong to the new round alone.

The point of keeping data/ beside the report is that an archived figure stays regenerable
years later with no Gadi access and no raw logs — `figdata` searches archived rounds too.

Usage:
    python analysis/archive_round.py --round 2026-08_uw3-v3.1.0
    python analysis/archive_round.py --round 2026-08_uw3-v3.1.0 --keep-data   # copy instead
"""

import argparse
import datetime
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", required=True, help="round label, e.g. 2026-08_uw3-v3.1.0")
    ap.add_argument("--keep-data", action="store_true",
                    help="copy data/<round>/ instead of moving it (leaves two copies)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an already-frozen round (destructive; normally refused)")
    # Freezing MOVES the dataset, so there is no way to see what it will do by running it.
    # Learned by freezing a round that was not ready.
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be moved and copied, and change nothing")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(repo, "reports", args.round)
    data_src = os.path.join(repo, "data", args.round)
    readme = os.path.join(repo, "README.md")
    figures = os.path.join(repo, "figures")

    # Refuse rather than clobber: a frozen round is the record of what was published, and
    # silently overwriting one destroys the only copy of a superseded result.
    if os.path.exists(dest) and not args.force:
        sys.exit(f"refusing: {dest} already exists (pass --force to overwrite)")

    missing = [p for p in (data_src, readme, figures) if not os.path.exists(p)]
    if missing:
        sys.exit("refusing: missing " + ", ".join(os.path.relpath(m, repo) for m in missing))

    pngs = [f for f in sorted(os.listdir(figures)) if f.endswith(".png")]
    if not pngs:
        sys.exit(f"refusing: no figures in {os.path.relpath(figures, repo)}")

    if args.dry_run:
        rel = lambda p: os.path.relpath(p, repo)
        print(f"dry run — nothing changed. Freezing '{args.round}' would:")
        print(f"  copy  {rel(readme)} -> {rel(dest)}/report.md (with a frozen-round banner)")
        print(f"  copy  {len(pngs)} figures -> {rel(dest)}/figures/")
        print(f"  {'copy' if args.keep_data else 'MOVE'}  {rel(data_src)} -> {rel(dest)}/data/")
        if not args.keep_data:
            print(f"        (after which data/{args.round}/ no longer exists)")
        return

    os.makedirs(dest, exist_ok=True)

    # The report gets a banner: without it a frozen copy is indistinguishable from the live
    # README, and someone will quote a superseded number from it.
    with open(readme) as fh:
        body = fh.read()
    banner = (
        f"<!-- FROZEN ROUND — do not edit -->\n"
        f"> **Archived round `{args.round}`**, frozen {datetime.date.today().isoformat()}.\n"
        f"> These results describe one environment and are not updated. The current round is\n"
        f"> in the repository README. Figures here regenerate from `data/` in this same\n"
        f"> directory.\n\n"
    )
    with open(os.path.join(dest, "report.md"), "w") as fh:
        fh.write(banner + body)

    fig_dest = os.path.join(dest, "figures")
    os.makedirs(fig_dest, exist_ok=True)
    for f in pngs:
        shutil.copy2(os.path.join(figures, f), os.path.join(fig_dest, f))

    data_dest = os.path.join(dest, "data")
    if os.path.exists(data_dest):
        shutil.rmtree(data_dest)
    if args.keep_data:
        shutil.copytree(data_src, data_dest)
    else:
        shutil.move(data_src, data_dest)

    print(f"frozen round -> {os.path.relpath(dest, repo)}")
    print(f"  report.md   (from README.md)")
    print(f"  figures/    {len(pngs)} files")
    print(f"  data/       {'copied' if args.keep_data else 'moved'} from "
          f"{os.path.relpath(data_src, repo)}")
    print()
    print("Next, for the new round:")
    print("  1. run the new campaigns and download them")
    print("  2. python3 analysis/export_data.py --round <new-label> --uw3-version ...")
    print("  3. regenerate figures, rewrite README.md")
    print("  4. record in CHANGELOG.md anything the new round OVERTURNS, not just adds")


if __name__ == "__main__":
    main()
