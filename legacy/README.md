# Legacy timing scripts

Pre-dating the campaign structure described in the repository README. Not run by
`scaling_test_job_launcher.sh` — it copies only top-level `*.py` — and not part of any round.
Kept because some still target live UW3 APIs.

## `timed_individual/`

Micro-benchmarks of single operations: mesh creation and init, swarm create/populate/advect,
Stokes and advection-diffusion create/solve, and several checkpoint paths. One operation per
script, unlike the campaign models which time a whole run in PETSc log stages.

**Two are directly relevant to the unmeasured native-reload path** (see Open questions in the
main README). `3_timed_mesh_petsc_save_chkpt.py` and `4_timed_mesh_write_timestep.py` already
exercise `petsc_save_checkpoint` and `write_timestep`, both of which still exist in UW3
v3.1.0. The 2026-08 checkpoint campaign measured `write_timestep` + `read_timestep` — the
coordinate-remap reader — so the native `petsc_save_checkpoint` / `read_checkpoint` pair has
never been timed at scale. These scripts are a starting point rather than something to rewrite.

## `timed_models/`

Whole-model timing scripts for Navier-Stokes examples (DFG 2D SLCN, lid-driven flow,
Poiseuille), SolC, a PETSc MMS case, and vector/tensor advection. Superseded by the
`*-scaling.py` models at the repository root, which use PETSc log stages and the
`params.sh` / launcher protocol.

The `go.sh` launchers still carry commented-out `timed_model_*.py` invocations from this era.
