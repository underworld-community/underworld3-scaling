# Underworld3 weak-scaling characterisation

**UW3 v3.1.0** · **Benchmark v1** · **Gadi (NCI)** · August 2026

Weak-scaling benchmarks for [Underworld3](https://github.com/underworldcode/underworld3),
run once per release. Each round characterises one release: a fixed UW3 version, PETSc
version and container image and is frozen into [`reports/`](reports/) when the environment
changes. Revised conclusions are in [CHANGELOG.md](CHANGELOG.md).

Protocol: `NTASKS = i³`, `res = BASE × i`, so elements per rank are constant and the ideal
is a **flat** curve.

## Findings

In the order the results are presented below.

| # | Finding | Evidence |
|---|---|---|
| | **Poisson** | |
| 1 | Mesh distribution dominates for a cheap solver — **but 41% of the largest run was a serial `.msh` conversion in our own harness**, since removed | `mesh_setup / steady_solves` = 1.4 at 1 rank, **620 at 2197**; `DMPlexCrFromFile` 2329 s of 5676 s on one rank |
| | **Stokes** | |
| 2 | **The parallel penalty is paid filling the first node, then efficiency flattens** — consistent across all four Stokes configurations. The steep loss is 1 → 27 ranks, entirely within one 48-core node; adding nodes beyond that costs little | from 27 ranks to the largest run all four configurations retain 0.82–0.92 (e.g. 1e-6: 0.47 → 0.42 over a 37× rank increase) |
| 3 | Two levers move the level but not the shape — `stokes.tolerance` and memory bandwidth per rank | tolerance 1e-8 → 1e-5: 1.8× faster, achieved residual 2.2e-8, reductions per solve 5731 → 2981. `--map-by node`, ~3.4× bandwidth per rank: efficiency 0.36 → 0.44 at 125 ranks, nothing at serial — about a fifth of the loss, the rest being communication |
| | **Advection–diffusion** | |
| 4 | Semi-Lagrangian advection does not scale, and is still climbing at 1000 ranks | SLCN 460.6 → 3496.0 s (eff 0.13) with solver work pinned; SNES in the same stage holds 0.89 |
| 5 | A third of a serial SLCN timestep loop is a fallback projection — legitimate, since a derivative cannot go through `evaluate`, but invisible to users | `SNES_MultiComponent.solve` 326.9 s vs `SNES_Scalar.solve` 156.2 s at 1 rank |
| | **Checkpoint I/O and `evaluate`** | |
| 6 | `read_timestep` does not scale — but it is a coordinate-remap reader (rank-0 file read, swarm migration, per-rank KDTree), not a field read. `write_timestep` degrades gracefully | read 1.1 → 263.4 s (eff 0.004) against write 4.3 → 69.7 s (eff 0.06); the native `read_checkpoint()` path was not measured |
| 7 | **Retracted.** `evaluate` at a variable's own coordinates is the *cheap* case; the 2026-08 collapse (eff 0.05) was an unbarriered stage absorbing upstream JIT skew | barriered: 19.3 → 26.0 µs/pt (eff **0.80**) vs 22.4 → 44.2 at cell interiors (0.55). `eval_offnode` unchanged between campaigns (0.97–1.00×) — the control |

**About the release artefact.** Three properties of the v3.1.0 container shaped these runs but
are not UW3 behaviour. `cache=False` in `SNES_Stokes_SaddlePt._build()` disables the JIT cache,
so every rank recompiles every run (5.8 s serial → 13.6 s at 1000 ranks; commit `2d3a0895`).
`solve()` overwrites user-set options — already fixed upstream (#477), but present here, and it
is why the Stokes inner tolerance had to be set by moving the outer one. And 32-bit PETSc
indices cap the container at 2197 ranks (`2147483650 is too big for PetscInt` at 4913), which
is the hard ceiling on this study; `--with-64-bit-indices` would lift it.

## Poisson

![Poisson weak scaling](figures/poisson.png)

Fixed-work: unreachable tolerance with `ksp_max_it`, giving exactly 10 KSP iterations at
every rank count. BASE=24 box mesh, 2–3 replicates.

| stage | 1 rank | 2197 ranks | eff (serial) |
|---|---|---|---|
| `mesh_setup` | 10.4 s | **5621 s** | 0.002 |
| `first_solve` | 11.0 s | 25.5 s | 0.43 |
| `steady_solves` | 7.2 s | 9.1 s | **0.79** |
| `error_analysis` | 1.5 s | 4.1 s | 0.37 |

The solver scales at 0.79 against serial, 0.90 against the 64-rank job — and it does not
matter: at 2197 ranks UW3 spends 5621 s distributing the mesh against 9 s solving, growing as
`t = 0.784 × N^1.152`.

**2329 s of that was our harness.** `Mesh()` given a `.msh` first runs a serial rank-0
Gmsh→HDF5 conversion (`DMPlexCrFromFile`); we cached the `.msh` instead of the `.msh.h5` it
produces. Caching the `.h5` removes it: advdiff 657.8 → 279.1 s, checkpoint 670 → 281 s at 343
ranks, solver stages unchanged. `DMPlexDistribute` (1547 s) is the real distribution cost.
Poisson is not yet re-run; the power law includes the serial term.

**Caveat.** The script advertises `k = 1 + u²`, but under fixed work SNES aborts before
updating, so `u` stays zero and the operator is a constant-coefficient Laplacian. A control
run seeding `u` with the analytic solution changed `steady_solves` by ≤0.1% and
`SNESJacobianEval` by ≤0.3%, so the timings stand — the operator simply is not the nonlinear
one described.

## Stokes

![Stokes weak scaling](figures/stokes.png)

Fixed-tolerance: every point converges in 1 outer Krylov iteration, so cost is what varies.
BASE=5 spherical shell, 3 replicates; BASE=10 at tolerance 1e-5.

**Only one knob was turned: `stokes.tolerance`.** Everything else follows from it in fixed
proportion — `snes_rtol` directly, `ksp_atol` at ×1e-6, and the fieldsplit block rtols at
×0.1 (pressure) and ×0.033 (velocity). The outer KSP has no fixed rtol at all; the setter
enables Eisenstat–Walker, which chooses it adaptively per Newton step.

| `stokes.tolerance` | snes_rtol | ksp_atol | inner: pressure / velocity | residual | eff @1000 |
|---|---|---|---|---|---|
| **1e-8** (UW3 default) | 1e-8 | 1e-14 | 1e-9 / 3.3e-10 | 2.6e-11 | 0.323 |
| **1e-5** (recommended) | 1e-5 | 1e-11 | 1e-6 / 3.3e-7 | 2.2e-8 | **0.418** |
| **1e-2** | 1e-2 | 1e-8 | 1e-3 / 3.3e-4 | 2.5e-5 | — |

This is therefore a one-parameter sweep, not a study of the inner solve. Isolating the inner
tolerance would need `fieldsplit_*_ksp_rtol` set independently, which `#477` prevents in the
v3.1.0 container — `solve()` re-derives them from `tolerance` immediately before PETSc reads
them.

Tolerance is a constant-factor cost, not a scaling defect: cost is offset by ~1.8× while the
efficiency curves are near-identical, and loosening cuts global reductions from 5731 to 2981
per solve at 125 ranks. **1e-5 recommended** — the residual is still 2.2e-8.

A separate effect, independent of tolerance, sets the shape of those curves: the parallel
penalty is paid filling the first node. The steep loss (1 → 27 ranks) sits entirely within one
48-core node, and beyond it the curve is nearly flat — 0.47 → 0.42 over a 37× rank increase.
All four configurations do this, at both tolerances and both BASE=10 placements, each
retaining 0.82–0.92 across that range. The consistency points to a fixed parallelisation
overhead paid while the first node fills, rather than to anything configuration-specific.

More memory bandwidth per rank somewhat helps. Spreading a BASE=10 job with `--map-by node`
gives each rank ~3.4× the bandwidth (14 ranks/node instead of 48) and lifts efficiency from
0.36 to 0.44 at 125 ranks, while changing nothing at serial. It recovers roughly a fifth of
the loss and no more: BASE=5, whose ~1 GB per rank was never starved, sits at the same 0.44.
The rest is communication and synchronisation, which bandwidth cannot remove.

## Advection–diffusion

![Advection-diffusion weak scaling](figures/advdiff.png)

Fixed-work: 20 SNES solves, 20 Jacobians, ~200 preconditioner applications at every rank
count. BASE=24, 10 timesteps.

| ranks | SLCN | SNES | SLCN share |
|---|---|---|---|
| 1 | 460.6 s | 462.2 s | 50% |
| 1000 | **3496.0 s** | 520.5 s | **87%** |
| efficiency | **0.13** | 0.89 | |

SLCN is not timed directly. PETSc times `SNESSolve`, and the stage covers the whole timestep
loop, so the semi-Lagrangian work, characteristic tracing and interpolation, is simply
what is left over: **SLCN = `steady_solves` − `SNESSolve`**. Both are rank-0 times, so the
subtraction does not mix a max-across-ranks total with a single-rank event. Note this puts
the projection below on the SNES side of the split, not the SLCN side.

Form evaluation and Krylov scale nearly perfectly. The semi-Lagrangian step does not, and
shows no plateau. It is **not** collectives: reductions are flat (+8% over a 125× rank
increase) and message volume grows 1.7×, but point-to-point message *count* grows **7.2×**
(74 835 → 540 771), tracking the cost. The cause is **not established**.

The right-hand panel shows it is protocol-independent: a converged control lands on the same advection
curve with only the SNES half shifted.

**Two thirds of the "SNES" curve is not the solve.**

| ranks | `SNES_Scalar.solve` | `SNES_MultiComponent.solve` (projection) |
|---|---|---|
| 1 | 156.2 s | **326.9 s** |
| 1000 | 180.8 s | 366.8 s |

`SNES_AdvectionDiffusion.solve` calls `super().solve()` once per timestep; the second PETSc
`SNESSolve` comes from the history term. `SemiLagrangian._record_current_field_into_history`
evaluates the flux `κ∇u`, which contains a derivative, so `evaluate` raises and an
`except Exception` branch runs a full projection instead. `SNESLineSearch` = 10 against
`SNESSolve` = 20 confirms it. This is characterisation, not a defect — a derivative cannot go
through `evaluate`.

## Checkpoint I/O

![Checkpoint I/O](figures/checkpoint.png)

Data per rank is constant (30.2 → 27.1 MB/rank), so the ideal is **flat time**.

| ranks | written | `write_timestep` | `read_timestep` | read/write |
|---|---|---|---|---|
| 1 | 30 MB | 4.3 s | 1.1 s | 0.25 |
| 64 | 1.8 GB | 11.4 s | 12.0 s | 1.05 |
| 1000 | 27 GB | 69.7 s | **263.4 s** | **3.78** |

**The two stages are not symmetric, and the read is not an I/O measurement.**
`mesh.write_timestep()` writes the mesh and all three fields in one call.
`MeshVariable.read_timestep()` is the *coordinate-remap* reader: rank 0 reads the file, saved
`(coord, value)` pairs migrate to the ranks that own them, each rank runs a local KDTree
against what it received, and interpolated values migrate back. It exists for reloading onto
a *different* mesh.

So the read stage times point location and swarm migration. Its collapse (efficiency 0.004
at 1000 ranks) is real and unexplained; it was once attributed to the same mechanism as the
`evaluate` result below, an inference that fell with that result's retraction. For
same-mesh reload — which is what this benchmark does — UW3 provides `read_checkpoint()`, a
native PETSc DMPlex path. It was not measured.

`io_write` replicates disagree by up to **79%** (reads ≤6%), because writes contend with
other filesystem users. `meshUpdates=True` throughout, so every checkpoint writes the mesh as
well as the fields — most of the bytes, ~90% at res=24 and 66% at res=240. Round-trip is
bit-exact on all three fields, which same-mesh nearest-neighbour matching guarantees.

## `uw.function.evaluate`

![evaluate weak scaling](figures/evaluate.png)

Same expression, mesh, rank count and point count (~12k/rank); the stages differ only in
whether target points are locally owned, so the ideal is flat.

**Nodal evaluation is the cheap case:** 19.3 → 26.0 µs/pt to 343 ranks (efficiency **0.80**)
against 22.4 → 44.2 at cell interiors (0.55). No search is needed for a point that is a node.

**The 2026-08 result (efficiency 0.05) was an artefact.** The stages had no barrier;
`eval_nodal` ran first and its first collective (`Mesh.get_max_radius`) absorbed the ranks'
JIT skew — `jit_warmup_sync` 4.76 s at 343 ranks, `mesh_setup_sync` and `field_setup_sync`
0.00 (upstream #547). `eval_offnode` ran second on synchronised ranks and is unchanged between
campaigns (0.97–1.00×): the control. In the figure the superseded curve lies on the sync
curve. Retraction in [CHANGELOG.md](CHANGELOG.md). The correctness question — on-vertex
sampling can mis-locate at a process seam (`ddt.py:2224`, #606) — is separate and stands.

## Measurement pitfalls

Each of these produced a wrong conclusion here before being caught.

- **Baseline choice moves Stokes from 0.32 to 0.87.** Ranks-per-node across an `i³` sweep is
  1, 8, 27, 48, 48, 48, 48, so low-rank points measure node occupancy. Every efficiency panel
  shows both baselines; full occupancy is *derived* from the sweep's own maximum
  ranks-per-node, since Gadi packs 48 and Setonix 128.
- **A larger per-rank working set flatters its own serial baseline** — the same trap, and why
  "work per rank scales worse" looked true.
- **Per-node memory footprint** sets the bandwidth regime, not work per rank or placement
  separately. PBS enforces memory per node.
- **`PCApply − 1`** is a valid Krylov-iteration proxy for scalar solvers; it reads 0 for
  nested ones, because PETSc folds inner `KSPSolve` calls into the outer counter.
- **PETSc's FLOP column is per rank** — do not divide by `nprocs`. Stage time must be the max
  across ranks. Event counts differ ~100× between ranks under AMG consolidation.
- **Reduction counts miss non-PETSc MPI** — direct `MPI_*` calls in Python do not appear.
- **An unreachable tolerance is safe for scalar solvers and poisons nested ones.** For Stokes,
  UW3 derives inner tolerances from the outer one, so at 1e-50 neither inner KSP can converge;
  with a matrix-free Schur complement every pressure iteration triggers a full velocity solve.
  This hung the campaign for days.
- **A nonlinear benchmark can silently measure a linear operator.** `zero_init_guess` is
  tri-state, defaulting to auto-detect → `True` on a cold solver, which discards any initial
  state written beforehand.
- **JIT lands in whichever stage touches an expression first** — without an untimed warm-up
  the first timed stage absorbs it, which once inverted an I/O comparison.
- **`stage.pop()` is a local call.** A stage records each rank's own duration; early ranks
  carry their lead into the next stage, whose first collective absorbs it. This retracted
  finding 7. Every stage is now followed by a timed barrier in its own `<name>_sync` stage
  (work / arrival spread; they sum to the true span). Cost: 0.01 s in a 3791 s run.
- **Uniform stage time across ranks means a collective boundary, not balance.** `mesh_setup`
  is 5637.28 max / 5637.27 min at 2197 ranks and contains a 2329 s single-rank phase. Use
  per-event max/mean/min and `n_working`; `idle% = 1 − mean/max` is ~0 when all ranks wait.

## Running a campaign

```bash
vi params.sh                        # set the campaign
bash scaling_test_job_launcher.sh   # one job per JOBS × RUN_INDICES
```

The launcher refuses to write into a `job/iter` that already has output
(`UW_ALLOW_OVERWRITE=1` overrides) — an unchanged `UW_NAME` once silently overwrote three
finished jobs.

**Adding a parameter takes three places.** Miss one and the job silently uses the default
rather than failing — this cost a five-job campaign:

1. `params.sh` — `export UW_THING=...`
2. `gadi_{container,baremetal}_go.sh` — `-uw_thing ${UW_THING:-default}`
3. `scaling_test_job_launcher.sh` — add `UW_THING` to `EXPORTVARS`

**Protocol choice.** Fixed-work (unreachable tolerance + `ksp_max_it`) holds iteration count
constant, so efficiency is not confounded by conditioning growth; use it for scalar solvers.
Fixed-tolerance (reachable rtol) lets every point converge; use it for nested solvers, where a
derived inner tolerance makes the unreachable variant pathological.

**Before spending service units:** run serial first (catches a hang for ~1 SU, not 50); check
`PCApply` is flat across the sweep (scalar solvers only — for nested ones it is a
degenerate 1 regardless); check `ranks_per_node` matches intent; confirm the mesh's
**`.msh.h5`** is in `MESH_CACHE` — a `.msh` alone works but pays a serial conversion that was
41% of the largest Poisson run; and if running a bind-mounted source tree, print `uw.__file__` — a failed
`PYTHONPATH` shadow silently runs the image's copy.

```bash
python3 analysis/export_data.py --round <label> --uw3-version ... --site gadi
python3 analysis/fig_poisson.py      # and the other fig_*.py
python3 analysis/archive_round.py --round <label> --dry-run
```

## How results are stored

Raw PETSc logs are 5.8 GB — one 2197-rank `timing.csv` is 423 MB, since PETSc writes a row per
(stage, event, **rank**). `/scratch` is purged, so raw output is ephemeral and gitignored.

Committed instead is the reduction the figures consume — **3.2 MB**:

```
data/<round>/{manifest.json, runs.csv, stages.csv, events.csv, info.json}
```

`figdata.load()` prefers `data/`, then archived rounds, then raw. All five figures regenerate
byte-identically from the CSVs, so archived figures stay reproducible without Gadi.

**Lossy in one way:** per-event max/min across ranks and rank 0's time are kept, but not
per-rank rows — a histogram across 2197 ranks could not be rebuilt. Keep raw until a round
closes.

**`benchmark_version`** guards cross-release comparison, which is only valid while the
benchmark measures the same work. Bump it when a model script changes what it measures. This
round is **v1**; nothing earlier is comparable.

**Infrastructure is now v2** (September 2026): timed barrier after every stage, mesh read
from `.msh.h5`. v1 and v2 *stages* are not comparable; totals are. The v1 stage curves stand:
per-rank stage durations are uniform to ≤0.27 s in every campaign's largest run, so every
stage ended on a collective and the artefact could only occur at the bare `evaluate` warm-up.

## Archive

| round | release | benchmark | site |
|---|---|---|---|
| *(open)* `2026-08_uw3-v3.1.0` | v3.1.0 | 1 | Gadi |

## Open questions

- **SLCN mechanism** — message count grows 7.2× while volume grows 1.7× and collectives are
  flat. Cause not established.
- **`error_analysis` grows 2.7×** with flat, negligible communication. Recorded as a curiosity.
- **Should `evaluate` short-circuit at a variable's own nodes?** Cost only: 26 µs/pt to
  return values already in `var.array`. `ddt.py` has the memcpy for the semi-Lagrangian path.
  Comment on upstream #551.
- **SLCN within-stage imbalance** — `KDTree.query` 116 / 83 / 99 s max/min/mean at 343 ranks
  (33% spread), hidden by the stage's trailing collective. Now visible via exported
  `time_max/min/mean`; unquantified across the sweep.
- **`read_checkpoint()` unmeasured** — the native same-mesh reload path. This campaign timed
  the coordinate-remap reader instead, so UW3's actual restart cost is unknown.
- **Setonix** — would separate "UW3 does this" from "Gadi does this". Blocked on
  `setonix_baremetal_go.sh` (absent) and a Setonix container.
