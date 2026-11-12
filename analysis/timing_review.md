# UW3 timing: how it works, how to improve it, and what our data says

Analysed 2026-08-25. **All four solvers**, each at its largest completed rank count.
Figure: `investigations/figures/timing_metrics.png`.

| campaign | run | ranks | total |
|---|---|---|---|
| Poisson | `weak-scaling-2026-BASE24-container2` | 2197 | 5676 s |
| Stokes | `weak-scaling-2026-stokes-BASE5-3way` | 1000 | 1345 s |
| AdvDiff | `weak-scaling-2026-advdiff-BASE24` | 1000 | 6677 s |
| Checkpoint | `weak-scaling-2026-h5mesh` | 343 | 548 s |

---

## 1. How UW3 timing works

Three layers, PETSc underneath. `PETSc.Log.Stage` for phases; `@timing.routine_timer_decorator`
(a thin `event.begin()`/`event.end()` wrapper) for functions; PETSc's own events for everything
else. `uw.timing.print_table("timing.csv")` is `PETSc.Log.view()` through an `ASCII_CSV` viewer.

**The property we were under-using: the CSV has a `Rank` column.** Every event is recorded on
every rank — 4.7 M rows for the 343-rank run. Imbalance is directly measurable, not inferred.

Two gotchas: `stage.pop()` is **local**, so a stage measures its own *duration*, never the
wall-clock *span*; and use `.csv` not `.txt` above ~1000 ranks (UW3 #134 — the ASCII path can hang).

---

## 2. Improving our own timing harness

**(a) One metric does the classifying:**

```
n_working = #ranks above 50% of max time     1 → serial,  N → collective
```

**(b) `idle%` (= 1 − mean/max) is NOT sufficient, and this matters.** When *all* ranks block
equally inside one collective, mean equals max and idle reads ~0. `mesh_setup/SFBcastEnd` at
2197 ranks is 596 s with **idle 4%** — that is 2196 ranks waiting, and idle% calls it balanced.
Asymmetric waiting shows in `idle%`; symmetric waiting only shows as its *paired* serial event.
Use both, and read them together.

**(c) The classifier has a known failure mode.** `n_working ≤ N/100` catches "one rank works",
but also catches "a few ranks wait longest" — e.g. Stokes `steady_solves/VecScatterEnd`
(136 s, 7/1000 ranks) is solve imbalance, not a serial phase. Compute events and
communication-wait events (`*End`, `VecScatter*`, `SFBcast*`) need reading differently.

**(d) Our reduction pipeline discards the evidence.** `figdata.py` correctly takes the max
across ranks for stages, but for events keeps only `count_max`, `count_min`, `time_rank0` — no
`time_min`, no `time_mean`. The exported 3.2 MB dataset cannot answer the sync question at all;
this analysis had to re-read the raw 241 MB CSVs. **Add `time_min`/`time_mean` per event to
`export_data.py`.**

**(e) Keep the `<name>_sync` barrier pattern.** Only it measures *arrival spread*, which no
per-event statistic recovers.

**(f) A visualisation app is not worth building.** The metric reduces 4.7 M rows to ~40; the
two static panels already answer the question.

---

## 3. What actually took the longest

### Serial share (lower bound — largest serial event per stage; PETSc events nest and the CSV has no call tree)

| campaign | serial | of total | dominant serial event |
|---|---|---|---|
| **Poisson** (2197) | **2329 s** | **41%** | `DMPlexCrFromFile` — Gmsh read, **1 rank** |
| **Stokes** (1000) | 275 s | 20% | `DMPlexCrFromFile` — **1 rank** |
| **AdvDiff** (1000) | 864 s | 13% | `DMPlexCrFromFile` — **1 rank** |
| **Checkpoint** (343) | 77 s | 14% | `DMPlexSymmetrize` 39 s, `KDTree.query` 38 s |

### The headline

**Poisson spends 41% of a 2197-rank run inside a Gmsh read that only rank 0 executes.** Not
"rank 0 is slower" — the other 2196 ranks never enter the event at all. Same event dominates
Stokes and AdvDiff. Only Checkpoint escapes it, because it is the one campaign already loading
a PETSc `.msh.h5`.

**So the h5 mesh-cache fix we measured on Checkpoint (670 → 281 s, 2.39×) has not been applied
to the other three, and the data says it is worth more there than it was there.**

### What is *not* synchronisation

- `io_write` (50 s) and `io_write_native` (97 s) in Checkpoint are honest collective I/O —
  `ISView`/`DMView` run at max/min ≈ 1.01 across all 343 ranks.
- AdvDiff's `steady_solves` (4017 s, 60% of that run) is genuinely distributed: `global_evaluate`
  3297 s and `KDTree.query` 1982 s at `n_working` = 1000/1000. Semi-Lagrangian point location is
  expensive, not serial.
- `jit_warmup` is real per-rank compilation — 14.45–19.19 s spread, which is exactly
  `jit_warmup_sync` (4.74 s) and accounts for essentially all campaign skew.
  `mesh_setup_sync` and `field_setup_sync` are **0.00**.

### `DMPlexDistribute` is not the serial part

`DMPlexDistribute` measures 1547 s at 2197 ranks and 186 s at 343 with `n_working` = *all*
ranks; `Mesh Partition` and `Mesh Migration` likewise. The serial cost is mesh **creation** —
`DMPlexCrFromFile`, `DMPlexCreateGmsh`, `DMPlexInterp`, `DMPlexSymmetrize` — which runs
*before* distribution, because UW3 builds the mesh on one rank and then distributes it.

Whether `DMPlexSymmetrize` nests inside `DMPlexDistribute` is **not determinable** from this
CSV, which carries no call tree. PETSc 3.25 `plexdistribute.c` indicates the partitioner works
from a single rank's complete mesh view before broadcasting, but that conflicts with all ranks
measuring 976 s inside `Mesh Partition`. **The source reading is unverified** — a summarised
fetch, not a full read (the conda install ships binaries only).

---

## 4. Actions

1. **Apply the h5 mesh cache to poisson / stokes / advdiff** (and ns, vep). Highest-value item
   here by a wide margin — 41% of the Poisson run.
2. Add `time_min` / `time_mean` to `export_data.py` so this is one command next time.
3. `io_read`'s rank-0 KDTree (38 s of 69 s) is avoidable via `read_checkpoint`, blocked on the
   parked parallel-correctness bug.
4. Read `plexdistribute.c` in full before any upstream claim about partitioner parallelism (§3).
5. **Pin BLAS to one thread per rank.** Neither Gadi launcher sets `OPENBLAS_NUM_THREADS`,
   `OMP_NUM_THREADS`, `MKL_NUM_THREADS` or `NUMEXPR_NUM_THREADS`, so each rank's BLAS is free to
   spawn threads against cores MPI already owns. Upstream `performance.md` now prescribes all
   four. Unquantified here, but it is a candidate explanation for per-rank variation we have
   been attributing elsewhere.

---

## 5. Review of UW3's timing module

`src/underworld3/timing.py`, 776 lines, PETSc 3.25. **Every claim tested**, reproducer
`upstream/repro_timing_parallel.py`. The file is **identical to `upstream/development`**, so
this reviews current code. No stage helper exists anywhere in `src/`; no parallel timing test
exists.

**Prior-art check — by ENUMERATION, not search.** GitHub search is unreliable here: querying
`print_table` does **not** return #134, whose title contains it. All **71 open issues** were
listed and scanned instead. Result:

| our finding | prior art | status |
|---|---|---|
| **Timing-module bugs** (§5.1) | **none — 0 open timing issues** | **novel** |
| **`.msh` vs `.msh.h5` mesh load** (§3) | none found | **novel** |
| `write_mesh_files_only` extension | none | **novel** |
| `evaluate` anti-scaling / nodal cost | **#551** (lmoresi, 13 Aug) | covered |
| `evaluate` wrong in parallel | **#606** (lmoresi, 19 Aug) | covered |
| `jit_warmup` skew | **#547** (bknight1, 13 Aug) | covered |
| `io_read` rank-0 KDTree | **#333** | covered |
| Gadi rank placement | **#635** (gthyagi, 23 Aug) | covered |
| boundary-flux memory | #408, fixed by #593 | closed |
| `eval_offnode` **serial** error | #567 / #641 — unresolved | **check** |

Three consequences.

**Do not file `issue_evaluate_timing.md`.** #551 measures `global_evaluate` at 2.83× slower than
serial at np=8 and names rank-local nodal evaluation as the benign case (efficiency
1.00/0.84/0.90 — our 0.80 at 343 ranks, independently reproduced). The only part it lacks (0 hits
for `ddt`, `_record_psi_star`, `memcpy`, `var.array`) is the proposal to lift the existing `ddt`
nodal memcpy into the public path. That is a **comment on #551**, not a new issue.

**Our `jit_warmup` skew has a named cause.** #547: `cache=False` bypasses the rank-0 compile gate,
so **every MPI rank compiles its own copy**. That is exactly why `jit_warmup` is the only stage
producing skew (14.45–19.19 s spread, `jit_warmup_sync` 4.74 s at 343 ranks). Worth adding as
corroboration at scale.

**Our `eval_offnode` serial anomaly may be genuinely distinct.** #606 reports serial as *exact*
(8.9e-16) and the failure as parallel-ownership; ours was **15.6% of points wrong in serial** with
error ≈ half a cell. Different signature. Characterise before filing or dismissing.

### Priority

| # | item | kind | severity |
|---|---|---|---|
| 1 | `get_summary`/`print_summary` are rank-local and say otherwise | bug | **high** |
| 2 | `reset()`/`stop()` are no-ops; docstrings promise behaviour | bug | **high** |
| 3 | Recursive decorated functions report `count=1` | bug | medium |
| 4 | Nothing encodes that `stage.pop()` is local | design | **high** |
| 5 | `get_summary()` cannot see stages | gap | medium |
| 6 | CSV is ~95% empty rows | practical | medium |
| 7 | Parallel semantics undocumented everywhere | docs | **high** |
| 8 | No canonical stage vocabulary; internal phases opaque | design | medium |

### 5.0 Scope: only the convenience API is broken

`print_table()` → ASCII **reduces correctly**. Our 343-rank `timing.txt`:

```
MPI Msg Count:   9.388e+06   247.450   6.704e+04      <- max, ratio, avg
1:  mesh_setup:  2.8772e+02   50.8%
```

| API | parallel behaviour |
|---|---|
| `print_table()` → `.txt` | **reduced** (max/ratio/avg); can hang ≳1000 ranks (#134) |
| `print_table()` → `.csv` | **per-rank rows**, ratio columns dropped — correct but raw |
| `get_summary()` / `print_summary()` | **rank-local, silently** |

PETSc's `Ratio` (max/min) is already most of what is needed. It cannot separate *one rank works*
from *one rank waits* — that needs `n_working` — and the CSV path loses it entirely.

### 5.1 BUGS

**(1) `get_summary()` / `print_summary()` report one rank, labelled as the run.** No MPI reduction
exists in the module. Test — 4 ranks, rank 0 sleeps 0.1 s, others 2.0 s:

```
get_summary() per rank: [(0, 0.105), (1, 2.005), (2, 2.005), (3, 2.004)]

print_summary():  "UNDERWORLD3 TIMING SUMMARY"   Total time: 0.105 seconds
                  Mesh.slow_on_everyone_but_rank0   1   0.105013   100.0%
TRUTH: max=2.00  mean=1.53  min=0.11
```

**5% of the real cost reported as 100%.** Fix: reduce; return `max`/`mean`/`min`/`n_working`.

**(2) `reset()` and `stop()` are no-ops.**

```
RESET_TEST  before_reset still present: True  t=0.303
STOP_TEST   after_stop recorded:        True  t=0.105
```

`reset()` ships a worked example — *"setup code (not timed) … reset() … performance-critical code
(timed)"* — that does nothing. Raise `NotImplementedError`, or implement via stages.

**(3) Recursion collapses the call count.** Isolated against a flat control — the decorator
counts repeated *non-nested* calls correctly, so this is specific to re-entry:

```
flat      count=4   time=0.212      4 calls, never nested   -- CORRECT
recurse   count=1   time=0.216      4 calls, 3 nested       -- WRONG count
inner/outer count=1 each            nested, DIFFERENT events -- CORRECT
```

Time is right — the outermost span covers the nest — but **count is wrong**, so any per-call
average derived from a recursive routine is wrong by the recursion depth. Either detect
re-entry and skip the inner `begin()`, or document that counts are outermost-entry counts.

*Verified sound:* exceptions (`finally` fires, `count=1 t=0.055`) and re-decoration (no
double-registration; same qualname, same event).

### 5.2 GAPS

**(4) Nothing encodes that `stage.pop()` is local.** Tested prototype:

```python
with uw.timing.stage("unequal_phase", barrier=True):   # emits <name> AND <name>_sync
```
```
unequal_phase        2.00  1.53  0.11    <- work; ranks diverge
unequal_phase_sync   1.90  0.48  0.00    <- the wait it created
```

Rank 0 worked 0.11 s then waited 1.90 s; the pair sums to the slowest rank's 2.00 s.

**(5) `get_summary()` cannot see stages.** It iterates `_petsc_events` only, so
`PETSc.Log.Stage` phases are invisible — `unequal_phase` never appeared in its output.

**(6) The CSV is ~95% empty rows.** Every registered event is written for every rank regardless
of whether it fired:

| run | size | rows | zero-count | if skipped |
|---|---|---|---|---|
| Checkpoint 343 ranks | 230 MB | 4.70 M | 98% | ~4 MB |
| Poisson 2197 ranks | 423 MB | 8.59 M | 94% | ~27 MB |

`skip_empty=True` on `print_table()` — 50–90× smaller, nothing lost.

### 5.3 DOCUMENTATION — verified absent, not just thin

| where | state |
|---|---|
| `docs/developer/profiling.md` | **referenced by the module docstring; does not exist** |
| `docs/advanced/performance.md` | was 27 lines with **zero** timing mentions; `28b0b791` (3 Sep) added 65 lines of Gadi `MatMult` placement guidance (#635). Still nothing on the timing API or its parallel semantics — and it closes by asking for "a small repeatable PETSc timing benchmark", which is what this campaign is |
| `docs/examples/Tutorial_Timing_System.py` | 282 lines, **zero** mentions of mpi/parallel/rank |
| `docs/advanced/parallel-computing.md` | timing appears only re: the #134 CSV hang |
| `tests/test_0055_timing_print_table.py` | 77 lines, deprecated-kwarg tests only; **no parallel correctness test** |

Proposed, in order of value per effort:

1. **Which API is parallel-safe** — the 5.0 table. Nothing states this anywhere today.
2. **"`stage.pop()` is a local call; a stage measures duration, never wall-clock span."** One
   sentence. It would have saved us a published-then-retracted result.
3. **A "reading parallel timings" page**: max sets the pace; ratio flags imbalance;
   `n_working == 1` means serial; *symmetric* waiting is invisible to both (our `SFBcastEnd` at
   2197 ranks is 596 s with idle 4% — 2196 ranks blocked, scored as balanced).
4. **The CSV schema.** Column meanings are undocumented and the rank count appears only as the
   last field of the header row.
5. **Fix the `reset()`/`stop()` docstrings** regardless of what else is done.
6. **Add a parallel timing test** — `mpirun -n 4`, unequal work, assert the reported max matches
   the slowest rank. Would have caught bug (1) at write time.

### 5.4 PHASE SEGMENTATION

**No canonical stage vocabulary.** Every script invents its own, so timings are not comparable
across models or releases; ours match only by convention. Ship a small vocabulary —
`mesh_setup`, `solver_setup`, `first_solve`, `steady_solves`, `io_write`, `io_read` — as
constants alongside the (4) context manager.

**Library-internal phases are opaque.** `mesh_setup` is 52% of a 343-rank run and 41% of a
2197-rank one, as a single block: file read, DMPlex distribute and variable setup are
indistinguishable without dropping to per-event data and guessing at nesting. **PETSc events nest
and the CSV carries no call tree** — that is why summing serial events naively gave Poisson 126%
of its own runtime, and why we still cannot say whether `DMPlexSymmetrize` sits inside
`DMPlexDistribute`. Sub-stages pushed inside `Mesh.__init__` (load / distribute / label /
coordinate-system) would make attribution free instead of inferential.

**Decorator coverage is 3.5%** — 105 occurrences against 2961 `def` lines, concentrated in
`solvers.py` (27), `swarm.py` (18), `discretisation_mesh.py` (11). Defensible, since PETSc's own
events cover the numerics; but it means UW3-level attribution is thin exactly where our data says
the time goes (mesh setup, I/O).

### Not a problem

The decorator is sound — a thin `begin()`/`end()` wrapper with a module-level cache, exception-safe,
idempotent under re-decoration. Routing through PETSc events rather than a bespoke parallel Python
timing layer is the right architecture, and the `.csv`-at-scale advice (#134) is correct.
