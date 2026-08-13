"""Shared helpers for the scaling figures.

Standard matplotlib appearance — default colour cycle, default chrome. What this module adds
is consistency, not restyling:

- COLOUR FOLLOWS THE ENTITY. mesh_setup is the same colour in every figure, so a reader
  moving between them does not have to re-learn the legend.
- RANK TICKS ARE REAL RANK COUNTS. A log axis labelled 10^0, 10^1, 10^2 does not answer
  "was that run at 343 or 1000 ranks", which is a question people actually ask.
- EFFICIENCY HAS AN IDEAL LINE. Weak scaling is flat at 1.0; without the reference the eye
  has nothing to judge against.

`matplotlib.use("Agg")` must precede the pyplot import: it selects the non-interactive
backend that rasterises to file, which is what these scripts do and what works headless on
Gadi. Comment it out if you want to open a figure interactively.
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

# Matplotlib's default tab10 cycle. Slots are fixed so the entity mappings below stay stable.
PALETTE = [f"C{i}" for i in range(10)]

# PETSc log stages appear in every campaign, so they get a fixed mapping — mesh_setup is the
# same colour whether you are looking at Poisson or checkpoint I/O.
STAGE = {
    "mesh_setup":     "C1",
    "solver_setup":   "C7",
    "field_setup":    "C7",
    "first_solve":    "C3",
    "steady_solves":  "C0",
    "error_analysis": "C4",
    "io_write":       "C0",
    "io_read":        "C3",
}

# Underscored names kept deliberately: these are the literal PETSc log stage identifiers, so
# a label in a figure can be grepped straight out of timing.csv or the model scripts. The
# parenthetical is the explanation; the identifier stays exact.
STAGE_LABEL = {
    "mesh_setup":     "mesh_setup (load + distribute)",
    "solver_setup":   "solver_setup",
    "field_setup":    "field_setup",
    "first_solve":    "first_solve (incl. JIT)",
    "steady_solves":  "steady_solves (hot path)",
    "error_analysis": "error_analysis",
    "io_write":       "io_write (mesh.write_timestep)",
    "io_read":        "io_read (read_timestep — coordinate remap)",
}

# Entities that recur across figures, likewise fixed.
ENTITY = {
    "poisson":    "C0",
    "stokes":     "C1",
    "advdiff":    "C2",
    "checkpoint": "C3",
    # The steady_solves decomposition. Kept clear of mesh_setup (C1), first_solve (C3) and
    # steady_solves (C0), which share a panel with them in the advdiff figure.
    "advection":  "C2",   # the semi-Lagrangian step
    "snes":       "C4",   # residual + Jacobian + Krylov
    "total":      "C0",
    "write":      "C0",
    "read":       "C3",
    "nodal":      "C0",
    "offnode":    "C1",
}

# The i^3 progression every campaign uses.
RANKS = [1, 8, 27, 64, 125, 343, 1000, 2197]


def new_figure(ncols, width_per=5.8, height=5.0):
    """Figure plus a list of axes — a list even for ncols=1, so callers index uniformly."""
    fig, axes = plt.subplots(1, ncols, figsize=(width_per * ncols, height))
    return fig, ([axes] if ncols == 1 else list(axes))


def new_grid(nrows, ncols, width_per=5.8, height_per=5.0):
    """Figure plus a 2D list of axes, `ax[row][col]`.

    Panels keep the same physical size as `new_figure`, so a 3x2 grid is as legible per
    panel as a 1x3 row — the figure grows rather than the panels shrinking.
    """
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows))
    return fig, [list(r) for r in axes.reshape(nrows, ncols)]


def line(ax, xs, ys, colour, label, dashed=False, marker="o", lo=None, hi=None):
    """One series, with error bars when `lo`/`hi` (the replicate range) are supplied.

    Bars span min-to-max over replicates rather than +/- a standard error — see
    `figdata.bounds` for why. Points run only once get a zero-length bar, so a reader can
    see at a glance which parts of a sweep were repeated and which rest on a single run.
    """
    if lo is None or hi is None:
        ax.plot(xs, ys, color=colour, marker=marker, markersize=5,
                linestyle="--" if dashed else "-", label=label)
        return
    # Clamped because the mean of the replicates can land a float epsilon outside the
    # observed range, and matplotlib rejects a negative yerr outright.
    yerr = [[max(0.0, y - l) for y, l in zip(ys, lo)],
            [max(0.0, h - y) for y, h in zip(ys, hi)]]
    ax.errorbar(xs, ys, yerr=yerr, color=colour, marker=marker, markersize=5,
                linestyle="--" if dashed else "-", label=label,
                capsize=3, elinewidth=1, capthick=1)


def style(ax, ylabel, title, xs=None, logy=False, legend_loc="best", legend=True):
    ax.set_xscale("log")
    ticks = [n for n in RANKS if xs is None or n in xs]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    # X only. `ax.minorticks_off()` would also strip the Y minor ticks, and a log-y panel
    # spanning less than a decade then carries a single labelled tick (10^0) with nothing
    # between — unreadable. The rank axis is the one with explicit ticks to protect.
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
        # A log axis spanning less than two decades gets ONE labelled tick (10^0) and is
        # effectively unreadable — an efficiency panel running 0.3 to 3 needs to show 0.5
        # and 2. Label the 2/3/5 minors as plain numbers in that case; leave wide-range
        # panels on the default powers of ten, where plain numbers would be unwieldy.
        low, high = ax.get_ylim()
        if low > 0 and high / low < 100:
            plain = ticker.FuncFormatter(lambda v, _: f"{v:g}")
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 3, 5)))
            ax.yaxis.set_minor_formatter(plain)
            ax.yaxis.set_major_formatter(plain)
            ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    # A single series needs no legend — the title names it. Two or more always get one, so
    # identity is never carried by colour alone.
    if legend and len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(loc=legend_loc, fontsize=9)


def ideal_line(ax, ymax=None, logy=False):
    """Flat at 1.0 — the reference the eye needs on a weak-scaling efficiency panel.

    `ymax` leaves headroom when a curve exceeds 1.0, which happens whenever the baseline is
    a packed job: low-rank points genuinely ran faster per unit work because they had more
    of a node's memory bandwidth to themselves. That excursion is information, not an
    artefact, so it must not be clipped.

    `logy` is for panels whose efficiency spans decades rather than the usual fraction of
    one — checkpoint I/O falls to 0.004, which a linear axis pins to the floor from 125
    ranks on, hiding the difference between "poor" and "catastrophic". A log axis cannot
    start at zero, so the ylim is left to matplotlib there.
    """
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="ideal")
    if not logy:
        ax.set_ylim(0, max(1.05, (ymax or 0) * 1.1))


def efficiency(ys, baseline_index=0):
    """T(baseline)/T(N). baseline_index selects which point is called 1.0."""
    b = ys[baseline_index]
    return [b / y for y in ys]


def efficiency_bounds(ys, lo, hi, baseline_index=0):
    """Replicate range carried through T(baseline)/T(N).

    The baseline is held at its mean rather than being varied too. A baseline that ran
    slow does not scatter the points — it rescales the WHOLE curve coherently, so folding
    its spread into each point independently would overstate the per-point uncertainty and
    misdescribe the error as random. The baseline's own spread belongs in the caption.
    """
    b = ys[baseline_index]
    return ([b / h if h else 0.0 for h in hi],
            [b / l if l else 0.0 for l in lo])


def baseline_index(xs, ranks_per_node=None, packed_at=None):
    """Index of the first job in which a NODE fills — the alternative to a serial baseline.

    Not "the first fully-packed job": on Gadi (48-core nodes) an i^3 sweep reaches 64 ranks
    as 48 + 16 across two nodes, averaging 32 per node. What the threshold detects is the
    first job where the BUSIEST node is full, which is where the occupancy knee appears.
    Average occupancy keeps rising afterwards (32 -> 41.7 -> 42.9 -> 47.6 at 1000 ranks), so
    curves past this point are not perfectly occupancy-matched either.

    Below full occupancy each rank gets a larger share of its node's memory bandwidth than
    any later job does, so efficiency measured against those points is partly measuring node
    occupancy. That is the single biggest reporting trap in this data: the same Stokes runs
    read as 0.32 or 0.87 depending on which point is called 1.0.

    `packed_at` is DERIVED, not assumed, when placement was recorded: the sweep's own maximum
    ranks-per-node is full occupancy for that machine. This matters as soon as a second site
    is added — Gadi packs 48 to a node so an i^3 sweep first fills one at 64 ranks, Setonix
    packs 128 so it first fills one at 343. Hardcoding 48 would silently mis-baseline every
    Setonix curve, in exactly the way that produced the 0.42-vs-0.90 error.

    Falls back to the 64-rank job when placement is absent, which is right for the Gadi
    campaigns that predate `rank_placement()` — the only ones in that position.
    """
    known = [r for r in (ranks_per_node or []) if r]
    if known:
        threshold = packed_at or max(known)
        for i, rpn in enumerate(ranks_per_node):
            if rpn and rpn >= threshold:
                return i
    return xs.index(64) if 64 in xs else 0


def caption(fig, text):
    """State what was pinned. A reader must never have to guess the protocol.

    Wrapped to the figure's own width. Without this a long caption is laid out as one
    unbroken line and `bbox_inches="tight"` in `finish` widens the SAVED IMAGE to contain
    it — a 17-inch figure came out 29 inches wide, squashing the panels to illegible
    slivers. The panels set the width; the caption flows into it.
    """
    # ~16 characters per inch at fontsize 9, less a margin.
    wrapped = textwrap.fill(text, width=int(fig.get_figwidth() * 15))
    # Anchored at the bottom so extra lines grow upward into the space `finish` reserves.
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=9)
    fig._uw_caption_lines = wrapped.count("\n") + 1


def finish(fig, suptitle, path):
    fig.suptitle(suptitle, fontsize=13)
    # Reserve bottom space per caption line, or the wrapped text overlaps the x-axis labels.
    lines = getattr(fig, "_uw_caption_lines", 1)
    fig.tight_layout(rect=[0, 0.02 + 0.030 * lines, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
