# v3.2.0 against v3.1.0 — attribution

What moved between the releases, and which PETSc event moved it. Same three models, same
protocol, same Gadi placement; six rank counts (1, 8, 27, 64, 125, 343), two replicates each.
Front-page figures: [`fig_release.py`](../analysis/fig_release.py). Tables below:
`python3 analysis/compare_releases.py`, from the exported rounds under `data/`.

## What the events say

- **Poisson solver +4 … +12%, rising with rank count — upstream #584.** v3.2.0 no longer
  sets `pc_gamg_type agg`, `pc_gamg_repartition`, `pc_mg_type additive`,
  `pc_gamg_agg_nsmooths 2` or `mg_levels_ksp_max_it 3`; the solve runs on PETSc's GAMG
  defaults. Krylov iterations are pinned by the protocol, so the cost is per application:
  `PCApply` 2.2–2.4× per call, `PCSetUp` +25% at 343 ranks, `MatResidual` appears and
  `MatMult` counts fall as the cycle changes shape. Assembly (`DMPlexJacobianFE`,
  `DMPlexResidualFE`) is +4% at 1 rank and +1% at 343.
- **Poisson `error_analysis` −3 … −42%.** v3.1.0's `evaluate` PETSc event — a rank-0
  collective wait, 1.4 s at 64 ranks — no longer exists; the stage is the two integrals.
  Consistent with the retracted `evaluate` finding on the investigations page.
- **Advection–diffusion −47 … −53% in `steady_solves` from 8 ranks.** `SNESSolve` is
  unchanged. `global_evaluate` fell 2341 → 769 s at 343 ranks, `KDTree.query` calls
  4960 → 2144, and the collective wrapper around the off-rank evaluation 1390 → 95 s.
  `MatMult` counts agree within 1%: the solver did the same work, the semi-Lagrangian
  evaluation did a third of it.
- **Stokes within ±6%.** `KSPSolve_FS_Schu` equal to three digits, same 1 Newton / 1 outer
  Krylov. v3.2.0 resolves `zero_init_guess=None` from `has_solution`, so `steady_solves`
  reports "non-zero initial guess" and runs one extra `SNESSolve` (+1 `SNESFunctionEval`,
  ~2 s at 343 ranks) although the script zeroed the fields. #625's
  `velocity.fell_back_from_fmg → gamg` is recorded in every run.
- **`mesh_setup` −41 … −58% from 27 ranks: the harness, not the release.** v3.2.0 read the
  cached `.msh.h5`; v3.1.0 converted the `.msh` on rank 0 first (`DMPlexCrFromFile`). Of the
  283 s left at 343 ranks, `DMPlexDistribute` is 178 s (Mesh Partition 104, Migration 50),
  identical in both releases. The +14% at 1 rank is un-instrumented Python in
  `Mesh.__init__` (6.9 → 8.4 s).
- **Cold JIT.** The first runs on the new container compiled every expression: poisson job 1
  replicates 1–2 (4.7 and 5.3 s of `getext`), advdiff job 1 replicate 1 (3.6 s), stokes job 1
  replicate 1 (2.5 s). Flagged by `figdata._flag_cold` and excluded; the v3.1.0 Stokes runs
  recompile every time (2–5 s in every replicate), which is why the flag is relative to the
  fastest replicate rather than an absolute threshold.

## Tables

### Poisson — stage wall time, v3.1.0 → v3.2.0

| stage | 1 | 8 | 27 | 64 | 125 | 343 |
|---|---|---|---|---|---|---|
| `mesh_setup` | 10.4 → 11.8 (+14%) | 18.9 → 16.0 (-15%) | 46.2 → 27.0 (-41%) | 109.5 → 55.7 (-49%) | 223.0 → 96.8 (-57%) | 659.4 → 283.0 (-57%) |
| `solver_setup` | 0.9 → 1.0 (+8%) | 0.9 → 1.0 (+8%) | 1.0 → 1.0 (+8%) | 1.0 → 1.1 (+11%) | 1.0 → 1.1 (+14%) | 1.0 → 1.1 (+10%) |
| `first_solve` | 11.0 → 11.5 (+4%) | 14.4 → 15.0 (+4%) | 16.5 → 17.2 (+4%) | 16.9 → 17.9 (+6%) | 17.7 → 18.9 (+7%) | 18.5 → 20.1 (+9%) |
| `steady_solves` | 7.2 → 7.6 (+5%) | 7.5 → 8.0 (+7%) | 8.1 → 8.7 (+7%) | 8.2 → 8.9 (+9%) | 8.4 → 9.2 (+10%) | 8.4 → 9.4 (+12%) |
| `error_analysis` | 1.5 → 1.5 (-3%) | 1.8 → 1.6 (-10%) | 2.2 → 1.7 (-22%) | 3.1 → 1.8 (-42%) | 3.1 → 1.8 (-41%) | 3.4 → 2.0 (-40%) |

Seconds, max over ranks, mean over warm replicates (v3.1.0: 3/3/3/3/3/3; v3.2.0: 2/2/2/2/2/2 per point; 2 cold-JIT replicates excluded).

### Poisson — events in `mesh_setup` at 343 ranks (stage 659.4 → 283.0 s, -57%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `_from_gmsh` | 1 | 0 | 458.66 | 0.00 | -100% |
| `Mesh.__init__` | 1 | 1 | 645.85 | 278.00 | -57% |
| `DMPlexCrFromFile` | 1 | 0 | 262.42 | 0.00 | -100% |
| `DMPlexCreateGmsh` | 1 | 0 | 262.41 | 0.00 | -100% |
| `DMPlexInterp` | 50 | 49 | 210.17 | 0.01 | -100% |
| `DMView` | 2 | 1 | 130.37 | 1.02 | -99% |
| `DMPlexSymmetrize` | 82 | 79 | 123.17 | 37.32 | -70% |
| `_from_plexh5` | 1 | 1 | 65.97 | 89.25 | +35% |
| `DMPlexTopologyLoad` | 1 | 1 | 48.85 | 71.93 | +47% |
| `ISLoad` | 16 | 16 | 5.03 | 20.39 | +305% |
| `ISView` | 17 | 0 | 10.43 | 0.00 | -100% |
| `DMPlexStratify` | 82 | 79 | 5.48 | 2.04 | -63% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Poisson — events in `first_solve` at 343 ranks (stage 18.5 → 20.1 s, +9%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `SNES_Scalar.solve` | 1 | 1 | 18.43 | 19.92 | +8% |
| `SNESSolve` | 1 | 1 | 13.12 | 14.25 | +9% |
| `PCSetUp` | 1 | 1 | 6.54 | 7.17 | +10% |
| `PCSetUp_GAMG+` | 1 | 1 | 6.54 | 7.17 | +10% |
| `KSPSolve` | 1 | 1 | 0.42 | 0.81 | +94% |
| ` PCGAMGCreateL` | 3 | 3 | 1.83 | 2.21 | +21% |
| `PCApply` | 11 | 11 | 0.32 | 0.69 | +119% |
| `MatMult` | 14158 | 14290 | 0.32 | 0.69 | +113% |
| `  GAMG Reduce` | 2 | 2 | 0.67 | 1.01 | +51% |
| `MatAssemblyEnd` | 92 | 90 | 0.58 | 0.86 | +48% |
| `DMCreateMat` | 1 | 1 | 3.53 | 3.80 | +8% |
| `SNESSetUp` | 1 | 1 | 3.53 | 3.80 | +8% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Poisson — events in `steady_solves` at 343 ranks (stage 8.4 → 9.4 s, +12%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `SNES_Scalar.solve` | 1 | 1 | 8.36 | 9.24 | +10% |
| `SNESSolve` | 1 | 1 | 7.25 | 8.04 | +11% |
| `KSPSolve` | 1 | 1 | 0.42 | 0.84 | +99% |
| `MatMult` | 14158 | 14290 | 0.32 | 0.72 | +123% |
| `PCApply` | 11 | 11 | 0.32 | 0.72 | +123% |
| `PCSetUp_GAMG+` | 1 | 1 | 0.90 | 1.12 | +25% |
| `PCSetUp` | 1 | 1 | 0.90 | 1.12 | +25% |
| `PCGAMG Gal l01` | 1 | 1 | 0.16 | 0.33 | +111% |
| `MatPtAPSymbolic` | 2 | 2 | 0.09 | 0.22 | +156% |
| `VecScatterEnd` | 176 | 308 | 0.09 | 0.22 | +130% |
| `SNESJacobianEval` | 1 | 1 | 3.77 | 3.86 | +2% |
| `MatPtAPNumeric` | 3 | 3 | 0.69 | 0.78 | +13% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Stokes — stage wall time, v3.1.0 → v3.2.0

| stage | 1 | 8 | 27 | 64 | 125 | 343 |
|---|---|---|---|---|---|---|
| `mesh_setup` | 3.6 → 4.1 (+15%) | 6.1 → 5.4 (-12%) | 13.8 → 8.9 (-36%) | 32.2 → 14.8 (-54%) | 65.5 → 29.3 (-55%) | 204.8 → 84.2 (-59%) |
| `solver_setup` | 0.0 → 0.0 (+6%) | 0.0 → 0.0 (+3%) | 0.0 → 0.0 (-0%) | 0.1 → 0.0 (-11%) | 0.0 → 0.0 (-6%) | 0.0 → 0.0 (-3%) |
| `first_solve` | 113.6 → 111.7 (-2%) | 212.4 → 198.8 (-6%) | 282.1 → 280.0 (-1%) | 300.5 → 298.9 (-1%) | 307.7 → 306.2 (-0%) | 335.8 → 321.3 (-4%) |
| `steady_solves` | 107.7 → 109.1 (+1%) | 203.5 → 193.6 (-5%) | 272.8 → 274.7 (+1%) | 289.2 → 292.0 (+1%) | 295.9 → 297.8 (+1%) | 323.5 → 311.3 (-4%) |

Seconds, max over ranks, mean over warm replicates (v3.1.0: 3/3/3/3/3/3; v3.2.0: 2/2/2/2/2/2 per point; 1 cold-JIT replicates excluded).

### Stokes — events in `mesh_setup` at 343 ranks (stage 204.8 → 84.2 s, -59%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `_from_gmsh` | 1 | 0 | 144.14 | 0.00 | -100% |
| `Mesh.__init__` | 1 | 1 | 202.47 | 80.94 | -60% |
| `DMPlexCrFromFile` | 1 | 0 | 82.28 | 0.00 | -100% |
| `DMPlexCreateGmsh` | 1 | 0 | 82.27 | 0.00 | -100% |
| `DMPlexInterp` | 56 | 55 | 66.15 | 0.00 | -100% |
| `DMView` | 2 | 1 | 42.05 | 0.00 | -100% |
| `DMPlexSymmetrize` | 90 | 87 | 32.69 | 10.04 | -69% |
| `ISView` | 13 | 0 | 3.65 | 0.00 | -100% |
| `_from_plexh5` | 1 | 1 | 19.59 | 22.47 | +15% |
| `ISLoad` | 12 | 12 | 1.76 | 4.50 | +155% |
| `DMPlexTopologyLoad` | 1 | 1 | 14.61 | 17.04 | +17% |
| `DMPlexStratify` | 90 | 87 | 1.96 | 0.62 | -69% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Stokes — events in `first_solve` at 343 ranks (stage 335.8 → 321.3 s, -4%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `SolverBaseClass._build` | 1 | 1 | 3.31 | 0.56 | -83% |
| `SNES_Stokes_SaddlePt._setup_pointwise_functions` | 1 | 1 | 3.03 | 0.31 | -90% |
| `getext` | 1 | 1 | 2.88 | 0.18 | -94% |
| `SNESSolve` | 1 | 1 | 314.82 | 316.92 | +1% |
| `PCSetUpOnBlocks` | 1 | 1 | 67.98 | 70.00 | +3% |
| `PCSetUp_GAMG+` | 1 | 1 | 8.18 | 9.87 | +21% |
| ` PCGAMGCreateL` | 3 | 3 | 5.03 | 5.86 | +17% |
| `  GAMG Reduce` | 2 | 2 | 1.06 | 1.89 | +78% |
| `MatAssemblyEnd` | 136 | 135 | 0.80 | 1.35 | +70% |
| `MatMultTranspose` | 3408 | 3408 | 13.63 | 14.15 | +4% |
| `KSPSolve_FS_Schu` | 1 | 1 | 189.82 | 189.35 | -0% |
| `   GAMG Repart` | 2 | 2 | 0.27 | 0.71 | +165% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Stokes — events in `steady_solves` at 343 ranks (stage 323.5 → 311.3 s, -4%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `SNESFunctionEval` | 2 | 3 | 2.83 | 4.52 | +60% |
| `DMPlexResidualFE` | 2 | 3 | 2.49 | 4.01 | +61% |
| `SNES_Stokes.solve` | 1 | 1 | 310.32 | 311.76 | +0% |
| `SNES_Stokes_SaddlePt.solve` | 1 | 1 | 310.32 | 311.76 | +0% |
| `SNESSolve` | 1 | 2 | 310.11 | 311.45 | +0% |
| `MatMult` | 7388 | 8979 | 259.92 | 258.90 | -0% |
| `VecNorm` | 1202 | 1203 | 17.11 | 16.19 | -5% |
| `KSPSolve_FS_Schu` | 1 | 1 | 190.27 | 189.47 | -0% |
| `KSPSolve` | 1 | 1 | 201.78 | 201.05 | -0% |
| `PCApply` | 1 | 1 | 201.73 | 201.00 | -0% |
| `PCSetUp_GAMG+` | 1 | 1 | 3.38 | 3.70 | +10% |
| `DMPlexJacobianFE` | 1 | 1 | 41.42 | 41.70 | +1% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Advection-diffusion — stage wall time, v3.1.0 → v3.2.0

| stage | 1 | 8 | 27 | 64 | 125 | 343 |
|---|---|---|---|---|---|---|
| `mesh_setup` | 10.2 → 11.8 (+15%) | 18.8 → 16.1 (-14%) | 46.4 → 27.0 (-42%) | 109.6 → 56.9 (-48%) | 220.2 → 93.3 (-58%) | 657.8 → 273.2 (-58%) |
| `solver_setup` | 0.9 → 0.6 (-28%) | 1.0 → 0.7 (-31%) | 1.4 → 0.8 (-46%) | 2.2 → 0.8 (-64%) | 2.2 → 0.8 (-63%) | 2.3 → 1.1 (-51%) |
| `first_solve` | 210.4 → 199.0 (-5%) | 294.4 → 209.2 (-29%) | 333.2 → 224.4 (-33%) | 366.8 → 236.7 (-35%) | 386.9 → 251.2 (-35%) | 444.6 → 279.6 (-37%) |
| `steady_solves` | 922.8 → 820.6 (-11%) | 1662.3 → 873.2 (-47%) | 1963.9 → 974.1 (-50%) | 2271.4 → 1063.3 (-53%) | 2472.5 → 1188.0 (-52%) | 3059.7 → 1462.5 (-52%) |

Seconds, max over ranks, mean over warm replicates (v3.1.0: 2/2/2/2/2/2; v3.2.0: 2/2/2/2/2/2 per point; 1 cold-JIT replicates excluded).

### Advection-diffusion — events in `mesh_setup` at 343 ranks (stage 657.8 → 273.2 s, -58%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `_from_gmsh` | 1 | 0 | 462.77 | 0.00 | -100% |
| `Mesh.__init__` | 1 | 1 | 650.84 | 279.52 | -57% |
| `DMPlexCrFromFile` | 1 | 0 | 265.48 | 0.00 | -100% |
| `DMPlexCreateGmsh` | 1 | 0 | 265.47 | 0.00 | -100% |
| `DMPlexInterp` | 60 | 59 | 211.94 | 0.01 | -100% |
| `DMView` | 2 | 1 | 130.66 | 0.00 | -100% |
| `DMPlexSymmetrize` | 97 | 94 | 123.06 | 38.16 | -69% |
| `_from_plexh5` | 1 | 1 | 66.07 | 87.36 | +32% |
| `DMPlexTopologyLoad` | 1 | 1 | 49.17 | 70.39 | +43% |
| `ISLoad` | 16 | 16 | 5.06 | 19.63 | +288% |
| `ISView` | 17 | 0 | 11.48 | 0.00 | -100% |
| `DMPlexStratify` | 97 | 94 | 5.73 | 2.07 | -64% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Advection-diffusion — events in `first_solve` at 343 ranks (stage 444.6 → 279.6 s, -37%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `evaluate` | 5 | 0 | 179.39 | 0.00 | -100% |
| `SNES_AdvectionDiffusion.solve` | 1 | 1 | 445.51 | 279.70 | -37% |
| `global_evaluate` | 4 | 3 | 233.58 | 70.06 | -70% |
| `collective_operation.<locals>.wrapper` | 4 | 3 | 140.65 | 10.26 | -93% |
| `KDTree.query` | 520 | 230 | 98.70 | 9.98 | -90% |
| `KDTree.find_closest_n_points` | 520 | 382 | 98.56 | 43.22 | -56% |
| `Function.evaluate_nd` | 13 | 11 | 253.90 | 222.90 | -12% |
| `Function.rbf_evaluate` | 13 | 11 | 53.75 | 35.50 | -34% |
| `DMSwarmMigrate` | 44 | 33 | 5.15 | 1.61 | -69% |
| `VecNorm` | 270 | 244 | 3.24 | 0.64 | -80% |
| `SNESFunctionEval` | 7 | 7 | 41.79 | 44.09 | +5% |
| `DMPlexJacobianFE` | 5 | 5 | 93.72 | 91.84 | -2% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.

### Advection-diffusion — events in `steady_solves` at 343 ranks (stage 3059.7 → 1462.5 s, -52%)

| event | calls v3.1.0 | calls v3.2.0 | rank-0 s v3.1.0 | rank-0 s v3.2.0 | change |
|---|---|---|---|---|---|
| `SNES_AdvectionDiffusion.solve` | 10 | 10 | 3056.85 | 1470.32 | -52% |
| `global_evaluate` | 40 | 30 | 2340.57 | 769.38 | -67% |
| `collective_operation.<locals>.wrapper` | 40 | 30 | 1389.99 | 95.34 | -93% |
| `KDTree.query` | 4960 | 2144 | 1020.86 | 79.75 | -92% |
| `KDTree.find_closest_n_points` | 4960 | 3714 | 1019.64 | 479.15 | -53% |
| `evaluate` | 30 | 0 | 524.19 | 0.00 | -100% |
| `Function.evaluate_nd` | 110 | 90 | 1313.41 | 1055.25 | -20% |
| `Function.rbf_evaluate` | 110 | 90 | 608.73 | 426.40 | -30% |
| `DMSwarmMigrate` | 440 | 330 | 16.94 | 7.74 | -54% |
| `VecNorm` | 1006 | 943 | 12.47 | 3.75 | -70% |
| `SNES_MultiComponent.solve` | 10 | 10 | 364.51 | 356.47 | -2% |
| `SNESFunctionEval` | 30 | 30 | 181.56 | 188.35 | +4% |

Calls are the max over ranks; times are rank 0, first warm replicate. `PCApply` etc. are per-call costs × counts, so a count change and a per-call change read differently.
