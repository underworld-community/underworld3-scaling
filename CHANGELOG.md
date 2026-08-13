# Changelog

Conclusions that were **revised**, not merely added to. The README carries only current
claims; this file records what changed and why, because the reasoning is often more useful
than the number.

A new campaign that overturns an earlier result belongs here. One that merely extends a
result does not.

---

## Round `2026-08_uw3-v3.1.0` (benchmark v1)

### Stokes weak-scaling efficiency: 0.42 → 0.90

**Was:** "Stokes scales at 0.42 to 1000 ranks."
**Now:** 0.42 against serial, **0.90 against the first fully-packed job**.

Ranks-per-node across an `i³` sweep is 1, 8, 27, 48, 48, 48, 48. Only from 64 ranks is a node
full, so the 1/8/27 points enjoy memory bandwidth per rank that no later job gets. Efficiency
normalised to them measures node occupancy as much as parallel scaling.

Both numbers are true and answer different questions. Every efficiency panel now shows both.

### "More work per rank scales worse" — withdrawn

**Was:** BASE=10 scales worse than BASE=5 (0.37 vs 0.47 at 64 ranks), so work per rank
matters.
**Now:** an artefact of the serial baseline. Against the first packed job the ordering
reverses — 0.967 vs 0.932 at 125 ranks.

BASE=10's single-rank run holds 8× the working set and so gains more from owning a whole
node's bandwidth, inflating its serial point and depressing every efficiency measured against
it. A larger per-rank working set flatters its own baseline.

Two separate effects had been conflated. **Rank placement is real** — the matched pair is a
ratio of times, immune to baseline choice, and worth 21–35%. **Work per rank is not.**

Still not airtight: BASE=5 was run packed, so the pair is not placement-matched, and only 125
ranks lies beyond the packed baseline. A BASE=5 spread run (~2 KSU) would close it.

### "Point location is the SLCN bottleneck" — retracted

**Was:** the semi-Lagrangian collapse is caused by distributed point location.
**Now:** unproven. Report what does not scale and what the message counts show; leave the
cause open.

It was always an inference from message counts. The `evaluate` experiment intended as a
control neither confirms nor refutes it, for three reasons found only by reading the source:

1. A coordinate-only expression never touches the mesh —
   `is_pure_sympy_expression()` routes it to a lambdify fast path
   (`functions_unit_system.py:157`). The first test measured arithmetic.
2. `mode="default"` is DMInterp+RBF, approximate **by design**, so error is expected.
3. Evaluating at a variable's own coordinates is the pathological case, not the cheap one —
   and SLCN's interpolation is the *interior* case, which scales acceptably.

**The advdiff SLCN result itself is untouched.** It measured real wall time in a real
timestep loop with solver work pinned.

### Poisson "may have under-measured the solve" — tested, it did not

**Was:** the campaign assembled a constant-coefficient Laplacian, because `u` never leaves
zero under the fixed-work protocol, so the nonlinear `k = 1 + u²` timings might be wrong.
**Now:** the timings are unaffected. The scaling numbers stand as published.

A controlled re-run with `uw_init=analytic` (3 replicates, jobs 1 and 5) seeded `u` with the
MMS solution so `k` genuinely varies in space. `rel_norm` fell from exactly 1.0 to ~1e-17,
confirming the nonlinearity engaged. `steady_solves` changed by ≤0.1% and `SNESJacobianEval`
by ≤0.3% — the JIT kernel evaluates `1 + u²` pointwise at the same flop count whatever `u`
holds, and nothing branches on `u = 0`.

Downgraded from a possible measurement error to an interpretation caveat: the benchmark is a
Laplacian in disguise.

The first attempt at this test **failed silently** and reported no difference, because
`solve()`'s `zero_init_guess` defaults to `None` → `True` on a cold solver and zeroed the
initial state. Now in the README's pitfalls.

### "Two `SNESSolve` per `solve()` is a Picard warm-up" — refuted

**Was:** `SNESSolve` counts 20 for 10 advdiff timesteps; likely a Picard warm-up before the
Newton solve. Filed as unverified.
**Now:** it is a **fallback L2 projection**, and it costs more than the solve it supports.

`SNES_AdvectionDiffusion.solve` calls `super().solve()` exactly once per timestep. The second
PETSc `SNESSolve` comes from the history terms: `SemiLagrangian._record_current_field_into_history`
evaluates `psi_fn` at the upstream node positions, and for `DFDt` that expression is the flux
`κ∇u`. It contains a derivative, `uw.function.evaluate` raises, and an `except Exception`
branch runs a full `SNES_MultiComponent_Projection` instead (`ddt.py:2470`).

UW3's own solve-method events price it:

| ranks | `SNES_Scalar.solve` | `SNES_MultiComponent.solve` |
|---|---|---|
| 1 | 156.2 s | **326.9 s** |
| 1000 | 180.8 s | 366.8 s |

**2.1× the real solve, every timestep** — 35.7% of the serial timestep loop. Confirmed in
the converged control too (599.9 s vs 221.9 s), so it is not an artefact of the fixed-work
protocol. `SNESLineSearch` = 10 against `SNESSolve` = 20 is the confirming detail: only the
genuinely nonlinear solve line-searches.

It also explains the doubled `SNESJacobianEval` and `PCSetUp` counts previously filed
separately — 20 each, being 10 for the solve and 10 for the projection.

**Recorded as characterisation, not a defect.** A derivative cannot go through
`uw.function.evaluate`, so an L2 projection is a legitimate route; the source comment shows
the same path is expected for the Navier-Stokes viscous flux. How much of the 327 s is
recoverable cannot be determined from aggregated counters.

A second correction, made while checking this: the recoverable part is NOT the AMG rebuild.
`PCSetUp` is 10.1 s of ~480 s at 1 rank — about 2%. The cost is form evaluation
(`SNESJacobianEval` 282.7 s, `SNESFunctionEval` 166.8 s).

Verified from event counts and source, not from `snes_monitor` output, which was never
downloaded.

### "Outer rtol 1e-8 throughout" — corrected

**Was:** the Stokes campaigns all ran at outer rtol 1e-8, differing only in the inner
fieldsplit tolerance.
**Now:** only the default configuration did. The three settings differ in the WHOLE tolerance
chain.

| label | outer rtol | inner rtol | achieved residual |
|---|---|---|---|
| 1e-9 (default) | 1e-8 | 1e-9 | 2.6e-11 |
| 1e-6 | 1e-5 | 1e-6 | 2.2e-8 |
| 1e-3 | 1e-2 | 1e-3 | 2.5e-5 |

UW3 derives the inner tolerances from the outer one, and `#477` makes `solve()` overwrite any
instance-level override in the v3.1.0 container, so the only way to reach a chosen inner
value was `stokes.tolerance = inner_rtol / 0.1`. That moved the outer tolerance too.

The conclusions are unaffected — every point still converges in 1 outer Krylov iteration and
the efficiency curves are still near-identical, so tolerance remains a constant-factor cost
rather than a scaling defect. What changed is the description: "1.8× faster at the same outer
tolerance" was wrong, and it is a claim UW3 developers would have spotted immediately.

### `cache=False` as the cause of the Stokes hang — wrong

**Was:** the JIT cache being disabled explained why Stokes runs hung for hours.
**Now:** it costs ~4–7 s. The hang was `_INNER_RTOL_MARGIN` combined with `tol=1e-50`.

UW3 derives the inner fieldsplit tolerances from the outer one, so at 1e-50 neither inner KSP
could ever converge; each ran its full 200 iterations, and with a matrix-free Schur
complement every pressure iteration triggered a full velocity solve — roughly
10 × 200 × 200 multigrid cycles per solve, **independent of mesh size**, which is why res=3
was as slow as res=5.

Found by `gdb -p <pid> -batch -ex bt` on a hung run. `cache=False` remains a real bug worth
reporting; it was not this one. Lesson: get a stack trace before reasoning forward from a
plausible culprit.
