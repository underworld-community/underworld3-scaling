# `uw.function.evaluate` at a mesh variable's own coordinates costs as much as a full point search

## Summary

`evaluate(T.sym[0], T.coords)` — evaluating at the variable's **own** nodal coordinates, where the
values already exist in `T.array` — costs 19–26 µs/point. No search is logically required, yet the
cost is close to that of evaluating at arbitrary points, which does require locating an owning cell.

`ddt.py` already avoids this internally. The question is whether that short-circuit can be lifted
into the public `evaluate` path.

## Measurements

NCI Gadi, Underworld3 v3.1.0 (container), `UnstructuredSimplexBox`, scalar P1 field, 3 replicates.
Weak scaling: `ranks = i³`, `res = 24 i`, so points per rank is constant and a flat row is ideal.

Microseconds per point per rank:

| ranks | 1 | 8 | 27 | 64 | 125 | 343 |
|---|---|---|---|---|---|---|
| **own coordinates** | **19.3** | **22.3** | **23.8** | **24.4** | **24.7** | **26.0** |
| cell interiors (needs a search, for contrast) | 22.4 | 38.6 | 40.6 | 42.8 | 42.6 | 44.2 |

The own-coordinates case scales acceptably — efficiency 0.80 at 343 ranks — so this is a
constant-factor question, not a scaling one. The point is the absolute cost: in serial, returning
values that are already in hand costs 19.3 µs/point against 22.4 µs/point for a genuine search.

### How this was measured

Each timed region is preceded by an `MPI.Barrier()` in its own PETSc log stage, so rank imbalance
is measured separately from work. Without that barrier the first timed region absorbs skew
accumulated earlier — its first internal collective (`Mesh.get_max_radius`, reached via
`points_in_domain`) is where the wait surfaces. Our own first attempt at this made exactly that
mistake and reported a figure roughly 20× too high; the numbers above are the corrected ones.

## The short-circuit that already exists

`Timestepping._record_psi_star_from_field_data` in `src/underworld3/systems/ddt.py` handles this
case directly, and its docstring states the reasoning:

> When `psi_fn` is a single mesh-variable component living on this mesh with the same nodal
> layout as `psi_star[0]`, "evaluate at own nodes" is exactly that variable's nodal data, so we
> copy it directly — no point location.

It looks up the variable with `meshVariable_lookup_by_symbol`, checks the shapes agree, and copies
`var.array`.

## Question

Would you consider lifting this into the public `evaluate` dispatch
(`function/functions_unit_system.py`, `_evaluate_impl`), or is it deliberately scoped to `ddt`?

The obvious difficulty: the `ddt` version never verifies that the supplied coordinates *are* the
variable's own — it knows that by construction. A general version cannot assume it, so it would
need to compare the passed array against `var.coords` and fall through to the normal path on any
doubt. For the optimisation to be worth anything, that check has to be cheap relative to the
location cost it avoids, and a false positive would silently return the wrong array.

If that guard is the blocker, we are happy to prototype and measure it — we already have the
weak-scaling harness set up to report before/after µs/point.

## Reproducer

`repro_evaluate_barrier.py` in
[underworld-community/underworld3-scaling](https://github.com/underworld-community/underworld3-scaling)
runs the case with the barriers described above and prints µs/point at any rank count.
