# Upstream material

Minimal reproducers, kept separate from the campaign scripts so they can be pasted into an
issue without dragging in `params.sh`, the launcher, or the PETSc log-stage scaffolding.

## `repro_read_checkpoint_parallel.py` — UNVERIFIED, investigate before filing

`MeshVariable.read_checkpoint()` appears to return values attached to the wrong DOFs in
parallel. Serial is exact.

Observed 2026-08-19 at 4 and 8 ranks locally, and at 8 and 125 ranks on Gadi (v3.1.0
container), with identical signatures: the round-trip error equals each field's full range
(v 2.00, p 1.00, T 1.00 for fields spanning [-1,1] and [0,1]).

The reproducer shows the mechanism more precisely than the error norm does: on every rank the
reloaded array spans the WHOLE GLOBAL range, while that rank's original spans only its own
subdomain. So values are populated but permuted — not missing, not zero. `read_checkpoint`
maps point-by-point from `sectiondm.getSection()` to `subdm.getSection()`; if those disagree
in parallel the permutation is silently wrong.

Run:  `mpirun -n 4 python repro_read_checkpoint_parallel.py`  (needs PROBE_OUT set)

DO NOT FILE YET. Confirm the usage is correct first — in particular whether
`write_timestep(petsc_reload=True)` needs `create_xdmf=True` alongside it, and whether
`data_name` is being passed as intended. A usage error would look identical from outside.

## `repro_evaluate_barrier.py` — supports a RETRACTION, not a bug report

Shows that `uw.function.evaluate` at a mesh variable's own coordinates does NOT scale badly.
Our 2026-08 campaign reported efficiency 0.05; with an `MPI.Barrier()` before each timed
region it is 0.84. The published number was the first timed stage absorbing rank imbalance
accumulated earlier, which its first internal collective (`Mesh.get_max_radius`, inside
`points_in_domain`) happened to synchronise.

Nothing to file against UW3 here. The value is as a measurement-pitfall example, and as the
evidence for correcting our own README.
