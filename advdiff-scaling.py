"""
Semi-Lagrangian advection-diffusion scaling benchmark.

MMS: u_ana = sin(πx)sin(πy)sin(πz) on a 3D box with a prescribed
divergence-free velocity field (solid-body rotation).

Tests the new monotone_mode and theta parameters introduced in PRs #186-189.

PETSc log stages:
  mesh_setup    — mesh + variable creation
  solver_setup  — AdvDiff system + BCs + analytical velocity
  first_solve   — step 0 (includes JIT compilation)
  steady_solves — UW_NSTEPS timesteps (the hot path / scaling metric)

UW_NSTEPS (timestep count) and UW_MAX_ITS (KSP iterations per solve) are independent:
they pin different quantities and a weak-scaling campaign needs both held fixed.

Usage (via launcher):
  mpiexec -n $NTASKS python advdiff-scaling.py \
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_tol $UW_SOL_TOLERANCE \
      -uw_maxits $UW_MAX_ITS -uw_nsteps $UW_NSTEPS -uw_job $JOB_IDX -uw_idx $RUN_IDX \
      [-uw_theta 0.5] [-uw_monotone clamp] [-uw_memprobe true]
"""

import os
import json
import csv

import numpy as np
import sympy
from petsc4py import PETSc
from mpi4py import MPI
from enum import Enum

import underworld3 as uw

# --------------------------------------------------------------------------- #
# PETSc logging                                                                #
# --------------------------------------------------------------------------- #
uw.timing.start()

# --------------------------------------------------------------------------- #
# Parameters (CLI: -uw_name value; notebook: params.uw_name = value)          #
# --------------------------------------------------------------------------- #
params = uw.Params(
    uw_scaling  = "none",
    uw_res      = 8,
    uw_tol      = 1e-6,
    uw_job      = 0,
    uw_idx      = 0,
    uw_maxits   = 10,
    uw_theta    = 1.0,
    uw_monotone = "none",
    uw_memprobe = False,
    uw_nsteps   = 10,
)
params.summary("advdiff-scaling parameters")


#: Options whose effective values decide this campaign's cost. Read back rather than
#: trusted: UW3's solve() overrides some of what you configure (snes_max_it 3 -> 50 was
#: observed on the container for Stokes), so "we set X" is not evidence X was used.
_OPTIONS_TO_REPORT = (
    "ksp_rtol", "ksp_atol", "ksp_max_it",
    "snes_rtol", "snes_atol", "snes_max_it", "snes_ksp_ew",
    "pc_type",
)


def report_effective_options(solver, label):
    """Read options back from the PETSc database and return them for run_info.json."""
    opts = solver.petsc_options
    effective = {}
    uw.pprint(f"--- effective PETSc options ({label}) ---")
    for key in _OPTIONS_TO_REPORT:
        effective[key] = opts.getString(key) if opts.hasName(key) else None
        shown = effective[key] if effective[key] not in (None, "") else (
            "<unset — PETSc default>" if effective[key] is None else "<set, no value>")
        uw.pprint(f"    {key:24s} = {shown}")
    uw.pprint("-" * 46)
    return effective


def rank_placement():
    """How ranks are distributed over nodes.

    Memory bandwidth is shared per node, so ranks_per_node is a first-order control on
    solve time — an i^3 sweep with default packing runs at 1, 8, 27, 48, 48 ranks/node,
    which confounds efficiency measured against the low-rank baselines.
    """
    from collections import Counter

    per_node = Counter(MPI.COMM_WORLD.allgather(MPI.Get_processor_name()))
    return {
        "n_nodes":            len(per_node),
        "ranks_per_node_max": max(per_node.values()),
        "ranks_per_node_min": min(per_node.values()),
    }


def _reason_name(reason_cls, code):
    """PETSc converged-reason integer → its symbolic name, e.g. 2 → CONVERGED_RTOL."""
    for name in dir(reason_cls):
        if not name.startswith("_") and getattr(reason_cls, name) == code:
            return name
    return str(code)


def capture_solver_stats(solver, since=0):
    """
    Convergence counters for the solve(s) that just finished.

    `since` subtracts a previously captured cumulative KSP count, so a stage that
    loops over timesteps reports only its own iterations rather than the running
    total for the solver object.
    """
    snes = solver.snes
    ksp  = snes.getKSP()
    stats = {
        "snes_its":      snes.getIterationNumber(),
        "snes_reason":   _reason_name(PETSc.SNES.ConvergedReason, snes.getConvergedReason()),
        "ksp_its_outer": ksp.getIterationNumber(),
        "ksp_its_total": snes.getLinearSolveIterations() - since,
        "ksp_reason":    _reason_name(PETSc.KSP.ConvergedReason, ksp.getConvergedReason()),
    }

    # How far the solve actually got. A converged REASON alone does not make two runs
    # comparable — record what was achieved, not just that something was satisfied.
    try:
        stats["ksp_rnorm_final"]  = ksp.getResidualNorm()
        stats["snes_fnorm_final"] = snes.getFunctionNorm()
    except Exception as exc:
        stats["ksp_rnorm_final"]  = None
        stats["snes_fnorm_final"] = None
        uw.pprint(f"  (residual norms unavailable: {exc})")

    return stats


scaling       = params.uw_scaling
res           = params.uw_res
tol           = params.uw_tol
job           = params.uw_job
it            = params.uw_idx
max_it        = params.uw_maxits
nsteps        = params.uw_nsteps
theta         = params.uw_theta
monotone_mode = None if params.uw_monotone == "none" else params.uw_monotone
memprobe_on   = params.uw_memprobe

uw.pprint(
    f"scaling={scaling}  res={res}  tol={tol}  job={job}  idx={it}  "
    f"maxits={max_it}  nsteps={nsteps}  theta={theta}  monotone={monotone_mode}  "
    f"memprobe={memprobe_on}"
)

# --------------------------------------------------------------------------- #
# Optional memory diagnostics                                                  #
# --------------------------------------------------------------------------- #
memprobe_rows = []

if memprobe_on:
    from underworld3.utilities import memprobe
    memprobe.enable()

def _rss():
    if not memprobe_on:
        return 0.0
    return memprobe.snapshot().get("rss_mib", 0.0)

def _record_rss(label, before, after):
    if memprobe_on and uw.mpi.rank == 0:
        memprobe_rows.append({"stage": label, "rss_delta_mib": after - before})

# --------------------------------------------------------------------------- #
# Output directory                                                             #
# --------------------------------------------------------------------------- #
output_base  = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name     = os.environ.get("NAME", "out")
udeg, vdeg_v = 2, 2
qdeg         = 3
mesh_cache   = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")
monotone_str = params.uw_monotone

output_dir = (
    f"{output_base}/{dir_name}/advdiff_out/"
    f"{scaling}_theta{theta}_mono{monotone_str}_udeg{udeg}"
    f"_tol{tol}_res{res}_job{job}_iter{it}"
)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    uw.pprint(f"output: {output_dir}")

# --------------------------------------------------------------------------- #
# STAGE: mesh_setup                                                            #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("mesh_setup"); stage.push()

if uw.mpi.rank == 0:
    os.makedirs(mesh_cache, exist_ok=True)
uw.mpi.barrier()

cache_file = f"{mesh_cache}/UnstructuredSimplexBox_res{res}.msh"

if os.path.exists(cache_file):
    import shutil
    local_msh = f"{output_dir}/mesh.msh"
    if uw.mpi.rank == 0:
        shutil.copy2(cache_file, local_msh)
    uw.mpi.barrier()

    class _boundaries_3D(Enum):
        Bottom = 11; Top = 12; Right = 13; Left = 14; Front = 15; Back = 16

    class _boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1]); Top   = sympy.Matrix([0, 0, 1])
        Right  = sympy.Matrix([1, 0, 0]); Left  = sympy.Matrix([1, 0, 0])
        Front  = sympy.Matrix([0, 1, 0]); Back  = sympy.Matrix([0, 1, 0])

    from underworld3.coordinates import CoordinateSystemType
    mesh = uw.discretisation.Mesh(
        local_msh,
        degree=1,
        qdegree=qdeg,
        boundaries=_boundaries_3D,
        boundary_normals=_boundary_normals_3D,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
    )
else:
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0., 0.),
        maxCoords=(1., 1., 1.),
        cellSize=1.0 / res,
        qdegree=qdeg,
        filename=cache_file,
    )

T     = uw.discretisation.MeshVariable("T",   mesh, 1, degree=udeg)
T_ana = uw.discretisation.MeshVariable("T_a", mesh, 1, degree=udeg)
# Velocity carried as a mesh variable for the SLCN scheme
V     = uw.discretisation.MeshVariable("V",   mesh, 3, degree=vdeg_v, vtype=uw.VarType.VECTOR)

if uw.mpi.rank == 0:
    uw.pprint("--- mesh partition ---")
mesh.dm.view()

stage.pop()
_record_rss("mesh_setup", rss_before, _rss())

# --------------------------------------------------------------------------- #
# STAGE: solver_setup                                                          #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("solver_setup"); stage.push()

x, y, z = mesh.X

# Analytical solution (MMS)
u_ana_expr = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y) * sympy.sin(sympy.pi * z)

# Divergence-free solid-body rotation velocity (keeps the problem bounded)
v_expr = sympy.Matrix([
     sympy.cos(sympy.pi * y) * sympy.sin(sympy.pi * z),
    -sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * z),
     0.0,
])
V.data[:, 0] = uw.function.evaluate(v_expr[0], V.coords).reshape(-1)
V.data[:, 1] = uw.function.evaluate(v_expr[1], V.coords).reshape(-1)
V.data[:, 2] = uw.function.evaluate(v_expr[2], V.coords).reshape(-1)

# Advection-diffusion solver (semi-Lagrangian)
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=V,
    order=1,
    monotone_mode=monotone_mode,
)
adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1.0
adv_diff.theta = theta          # property, not constructor kwarg

for face in ("Back", "Front", "Bottom", "Top", "Left", "Right"):
    adv_diff.add_dirichlet_bc(0., face)

adv_diff.tolerance = tol
adv_diff.petsc_options["ksp_rtol"]  = tol
adv_diff.petsc_options["ksp_atol"]  = tol
adv_diff.petsc_options["snes_atol"] = tol

# Diagnostics. SNESSolve counts 2 per adv_diff.solve() call (20 for 10 timesteps), which
# suggests UW3 runs a Picard warm-up before the Newton solve — if so, that is a second
# avoidable cost in the dominant component, alongside the per-solve Jacobian re-assembly.
# Both print from rank 0 only, and snes_monitor reports a norm SNES already computes for
# its convergence test, so neither adds a collective.
adv_diff.petsc_options["snes_monitor"]          = None
adv_diff.petsc_options["snes_converged_reason"] = None
if max_it > 0:
    adv_diff.petsc_options["ksp_max_it"]  = max_it
    adv_diff.petsc_options["snes_max_it"] = max_it

# Initialise T to the analytical solution
T.data[:, 0] = uw.function.evaluate(u_ana_expr, T.coords).reshape(-1)

# Stable CFL timestep based on mesh resolution and unit velocity
dt = 0.5 * (1.0 / res)

stage.pop()
_record_rss("solver_setup", rss_before, _rss())

# --------------------------------------------------------------------------- #
# STAGE: first_solve  (includes JIT compilation)                              #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("first_solve"); stage.push()

# zero_init_guess=False so T keeps the MMS initial condition. With the fixed-work
# protocol (tol=1e-50) the solve returns DIVERGED_LINEAR_SOLVE without updating the
# solution vector — so zeroing it first left T identically zero for the entire run, and
# the semi-Lagrangian step then advected zeros. Timing was unaffected (linear problem, so
# the matrix does not depend on T, and characteristic tracing is geometric), but the run
# was not operating on a physically meaningful field.
adv_diff.solve(timestep=dt, zero_init_guess=False)

stage.pop()
solver_stats = {"first_solve": capture_solver_stats(adv_diff)}
effective_options = report_effective_options(adv_diff, "after first_solve")
_record_rss("first_solve", rss_before, _rss())

# --------------------------------------------------------------------------- #
# STAGE: steady_solves  (hot path)                                            #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("steady_solves"); stage.push()

# Timestep count is independent of ksp_max_it: they pin different quantities, and a
# weak-scaling campaign needs to hold both fixed at once (this was previously one knob,
# so changing the iteration budget silently changed how much physics was simulated).
n_steady = nsteps
for step in range(n_steady):
    adv_diff.solve(timestep=dt, zero_init_guess=False)

stage.pop()
# getLinearSolveIterations() accumulates over the SNES object's whole lifetime, so
# subtract the first_solve total to get the iterations spent in this stage alone.
solver_stats["steady_solves"] = capture_solver_stats(
    adv_diff, since=solver_stats["first_solve"]["ksp_its_total"]
)
solver_stats["steady_solves"]["n_timesteps"] = n_steady
_record_rss("steady_solves", rss_before, _rss())

uw.pprint(f"Completed {n_steady} steady timesteps")

# --------------------------------------------------------------------------- #
# Error at final step                                                          #
# --------------------------------------------------------------------------- #
# NOT an accuracy metric: u_ana_expr is the INITIAL condition, and T has been advected
# and diffused for nsteps+1 steps, so this measures departure from the starting field
# rather than solution error. Kept as a cheap "did the solution stay bounded" smoke test
# — a NaN or a huge value means the run is broken. Do not report it as error.
T_ana.data[:, 0] = uw.function.evaluate(u_ana_expr, T_ana.coords).reshape(-1)
drift_field = T.data[:, 0] - T_ana.data[:, 0]
local_max = float(np.max(np.abs(drift_field))) if len(drift_field) > 0 else 0.0
global_max = uw.mpi.comm.allreduce(local_max, op=MPI.MAX)
uw.pprint(f"Max drift from initial condition (sanity check, not error): {global_max:.4e}")

# --------------------------------------------------------------------------- #
# Write memprobe CSV                                                           #
# --------------------------------------------------------------------------- #
if memprobe_on and uw.mpi.rank == 0 and memprobe_rows:
    with open(f"{output_dir}/memprobe.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(memprobe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(memprobe_rows)

# --------------------------------------------------------------------------- #
# Run metadata                                                                 #
# --------------------------------------------------------------------------- #
# Collective — every rank must call this, so it sits outside the rank-0 block below.
placement = rank_placement()

if uw.mpi.rank == 0:
    with open(f"{output_dir}/run_info.json", "w") as fp:
        json.dump({
            "model": "advdiff-scaling",
            "theta": theta,
            "monotone_mode": monotone_mode,
            "res": res,
            "nprocs": uw.mpi.size,
            "scaling": scaling,
            "n_steady_steps": n_steady,
            "dt": dt,
            "tol": tol,
            "max_drift_from_initial": global_max,   # sanity check, NOT solution error
            "effective_options": effective_options,
            "placement": placement,
            "solver_stats": solver_stats,
        }, fp, indent=4)

# --------------------------------------------------------------------------- #
# Timing output                                                                #
# --------------------------------------------------------------------------- #
uw.mpi.barrier()
uw.timing.print_table(filename=f"{output_dir}/timing.csv")
uw.timing.print_table(filename=f"{output_dir}/timing.txt")
uw.pprint(f"Timing written to {output_dir}/timing.csv")
