"""
Spherical shell Stokes benchmark (isoviscous, incompressible) — scaling test.

Simple no-slip spherical shell driven by a radial body force.
Tests Stokes multigrid preconditioner scaling without analytical overhead.

Removed from the original accuracy benchmark:
  - assess library (analytical solution evaluation)
  - analytical_setup stage (Python loops over nodes — not parallel-scalable)
  - error_analysis stage (same issue)
  - natural BCs based on analytical velocity

PETSc log stages:
  mesh_setup   — SphericalShell mesh creation
  solver_setup — Stokes object, constitutive model, BCs
  first_solve  — single KSP/SNES solve (includes JIT compilation)
  steady_solves — re-solve from zero IC (JIT-warm; isolates pure solver cost)

Usage (via launcher):
  mpiexec -n $NTASKS python stokes-scaling.py \\
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_tol $UW_SOL_TOLERANCE \\
      -uw_maxits $UW_MAX_ITS -uw_job $JOB_IDX -uw_idx $RUN_IDX
"""

import os
import json
from enum import Enum

import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes

# --------------------------------------------------------------------------- #
# PETSc logging — must be started before any UW3 operations                   #
# --------------------------------------------------------------------------- #
uw.timing.start()

# --------------------------------------------------------------------------- #
# Parameters (CLI: -uw_name value; notebook: params.uw_name = value)          #
# --------------------------------------------------------------------------- #
params = uw.Params(
    uw_scaling = "none",
    uw_res     = 8,
    uw_tol     = 1e-8,
    uw_job     = 0,
    uw_idx     = 0,
    uw_maxits  = 10,
    uw_vdegree = 2,
    uw_pdegree = 1,
    uw_pcont   = True,
    uw_inner_rtol = 0.0,   # >0 overrides the fieldsplit inner tolerances; 0 = UW3 default
)
params.summary("stokes-scaling parameters")


#: Options whose effective values decide this campaign's cost, checked after the first
#: solve because UW3 rewrites some of them during solve() (_reassert_outer_tolerances,
#: and the tolerance setter's Eisenstat-Walker flags).
_OPTIONS_TO_REPORT = (
    "ksp_rtol",
    "ksp_atol",
    "ksp_max_it",
    "snes_rtol",
    "snes_max_it",
    "snes_ksp_ew",
    "fieldsplit_pressure_ksp_rtol",
    "fieldsplit_pressure_ksp_max_it",
    "fieldsplit_velocity_ksp_rtol",
    "fieldsplit_velocity_ksp_max_it",
)


def report_effective_options(solver, label):
    """Read options back from the PETSc database, rather than trusting what we set.

    Setting an option and PETSc using it are different things here: the tolerance setter
    derives the inner fieldsplit rtols, solve() re-asserts the outer ones, and
    Eisenstat-Walker overrides ksp_rtol per Newton step. Print what actually landed.
    """
    opts = solver.petsc_options
    effective = {}
    uw.pprint(f"--- effective PETSc options ({label}) ---")
    for key in _OPTIONS_TO_REPORT:
        if opts.hasName(key):
            value = opts.getString(key)
            effective[key] = value if value != "" else "<set, no value>"
        else:
            effective[key] = None
        uw.pprint(f"    {key:34s} = {effective[key] if effective[key] is not None else '<unset — PETSc default>'}")
    uw.pprint("-" * 52)
    return effective


def rank_placement():
    """How ranks are distributed over nodes.

    Not cosmetic: memory bandwidth is shared per node, so ranks_per_node is a first-order
    control on solve time. With default dense packing an i^3 weak-scaling sweep runs at
    1, 8, 27, 48, 48 ranks/node — so the low-rank jobs get several times the bandwidth per
    rank of the high-rank ones, and efficiency measured against them is confounded.
    PBS enforces its memory limit per node against this same number.
    """
    from mpi4py import MPI
    from collections import Counter

    names = MPI.COMM_WORLD.allgather(MPI.Get_processor_name())
    per_node = Counter(names)
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


def capture_solver_stats(solver):
    """
    Convergence counters for the solve that just finished.

    These are what make a fixed-tolerance campaign readable: wall time alone
    conflates parallel overhead with algorithmic degradation, but wall time
    divided by ksp_its_total separates them.

    ksp_its_total is cumulative across Newton steps; ksp_its_outer is the last
    linear solve only. For this linear problem SNES takes one step, so they agree.
    """
    snes = solver.snes
    ksp  = snes.getKSP()
    stats = {
        "snes_its":      snes.getIterationNumber(),
        "snes_reason":   _reason_name(PETSc.SNES.ConvergedReason, snes.getConvergedReason()),
        "ksp_its_outer": ksp.getIterationNumber(),
        "ksp_its_total": snes.getLinearSolveIterations(),
        "ksp_reason":    _reason_name(PETSc.KSP.ConvergedReason, ksp.getConvergedReason()),
    }

    # How far the solve actually got. Necessary because the converged REASON does not mean
    # the same thing between configurations: Eisenstat-Walker picks the outer KSP tolerance
    # adaptively (~0.3 on the first Newton step), so two runs can both report
    # CONVERGED_RTOL while differing by orders of magnitude in achieved residual — the
    # BASE=5 container runs gave 2.5e-11 with tight inner solves and 2.5e-05 with loose
    # ones. Without these numbers a cheaper-but-less-converged run looks like a free
    # speedup.
    try:
        stats["ksp_rnorm_final"]  = ksp.getResidualNorm()
        stats["snes_fnorm_final"] = snes.getFunctionNorm()
    except Exception as exc:
        stats["ksp_rnorm_final"]  = None
        stats["snes_fnorm_final"] = None
        uw.pprint(f"  (residual norms unavailable: {exc})")

    return stats


scaling = params.uw_scaling
res     = params.uw_res
tol     = params.uw_tol
job     = params.uw_job
it      = params.uw_idx
max_it  = params.uw_maxits

# --------------------------------------------------------------------------- #
# Output directory                                                             #
# --------------------------------------------------------------------------- #
output_base = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name    = os.environ.get("NAME", "out")
vdegree     = params.uw_vdegree
pdegree     = params.uw_pdegree
qdeg        = max(pdegree, vdegree)
mesh_cache  = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")
pcont       = params.uw_pcont
stokes_tol  = tol

output_dir = (
    f"{output_base}/{dir_name}/stokes_out/"
    f"{scaling}_vdeg{vdegree}_pdeg{pdegree}_tol{tol}_res{res}_job{job}_iter{it}"
)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    uw.pprint(f"output: {output_dir}")

# --------------------------------------------------------------------------- #
# Mesh geometry                                                                #
# --------------------------------------------------------------------------- #
r_o      = 2.22
r_i      = 1.22
cellsize = 1.0 / res

# --------------------------------------------------------------------------- #
# STAGE: mesh_setup                                                            #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("mesh_setup"); stage.push()

if uw.mpi.rank == 0:
    os.makedirs(mesh_cache, exist_ok=True)
uw.mpi.barrier()

cache_file = f"{mesh_cache}/SphericalShell_ri{r_i}_ro{r_o}_res{res}.msh"

if os.path.exists(cache_file):
    import shutil
    local_msh = f"{output_dir}/mesh.msh"
    if uw.mpi.rank == 0:
        shutil.copy2(cache_file, local_msh)
    uw.mpi.barrier()

    class _boundaries_spherical(Enum):
        Lower = 11
        Upper = 12

    from underworld3.coordinates import CoordinateSystemType
    mesh = uw.discretisation.Mesh(
        local_msh,
        qdegree=qdeg,
        boundaries=_boundaries_spherical,
        boundary_normals=None,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
    )
else:
    mesh = uw.meshing.SphericalShell(
        radiusInner=r_i, radiusOuter=r_o,
        cellSize=cellsize,
        qdegree=qdeg,
        filename=cache_file,
    )

v_uw = uw.discretisation.MeshVariable("V_u", mesh, mesh.dim, degree=vdegree)
p_uw = uw.discretisation.MeshVariable("P_u", mesh, 1, degree=pdegree, continuous=pcont)

if uw.mpi.rank == 0:
    uw.pprint("--- mesh partition ---")
mesh.dm.view()

stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: solver_setup                                                          #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("solver_setup"); stage.push()

r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
unit_rvec   = mesh.CoordinateSystem.unit_e_0

# Simple body force: cosine-colatitude density, radially directed gravity.
# Avoids assoc_legendre/Piecewise — fast sympy compilation.
rho = (r_uw / r_o) ** 2 * sympy.cos(th_uw)

stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

stokes.bodyforce = rho * (-1.0 * unit_rvec)

stokes.add_essential_bc(sympy.Matrix([0., 0., 0.]), mesh.boundaries.Upper.name)
stokes.add_essential_bc(sympy.Matrix([0., 0., 0.]), mesh.boundaries.Lower.name)

# Convergence tolerance. Unlike poisson-scaling.py, this campaign is FIXED-TOLERANCE,
# not fixed-iteration: every job solves to the same tolerance and we report both wall
# time and iteration count, so parallel efficiency (time per iteration) and algorithmic
# scalability (iteration count vs. nprocs) can be separated.
#
# A fixed-iteration setup is not viable here. UW3's Stokes tolerance setter derives
#     fieldsplit_pressure_ksp_rtol = tolerance * 0.1
#     fieldsplit_velocity_ksp_rtol = tolerance * 0.033
# from this value (_INNER_RTOL_MARGIN in petsc_generic_snes_solvers.pyx), and both
# inner KSPs default to ksp_max_it = 200. An unreachable tolerance (e.g. the 1e-50
# poisson-scaling.py uses to pin its iteration count) means neither inner solve ever
# converges, so each runs all 200 iterations — and because the Schur complement is
# applied matrix-free, every pressure iteration triggers a full velocity solve. That
# is ~10 x 200 x 200 multigrid W-cycles per solve, independent of mesh size.
#
# Guard rather than clamp: a tolerance below double precision is always a config error
# (usually UW_SOL_TOLERANCE left at the 1e-50 poisson uses), and the failure mode is a
# job that hangs until it burns its whole walltime. Fail in seconds instead.
if stokes_tol < 1e-12:
    raise ValueError(
        f"uw_tol={stokes_tol:g} is below double precision and cannot be reached by the "
        f"inner fieldsplit solves, which would each run their full 200 iterations and "
        f"hang the job. Stokes is a fixed-tolerance campaign — set UW_SOL_TOLERANCE=1e-8. "
        f"(poisson-scaling.py uses 1e-50 deliberately; that trick does not transfer here.)"
    )
stokes.tolerance = stokes_tol

# The tolerance setter derives the inner fieldsplit tolerances as tol*0.1 (pressure) and
# tol*0.033 (velocity) — at tol=1e-8 that is 1e-9 and 3.3e-10, which is not inexact by any
# reading, despite the UW3 source describing the inner solves as "deliberately inexact".
# Tight inner solves mean many inner Krylov iterations, and each one costs a global
# MPI_Allreduce: the BASE=5 container campaign showed ~5500 reductions per solve against
# Poisson's 189, which is what caps parallel efficiency at 0.36 by 125 cores.
#
# A flexible outer method (fgmres) tolerates sloppy preconditioning by design, so loosening
# these should cut reductions without changing the outer iteration count. Must be set AFTER
# stokes.tolerance, which writes them.
# Setting the fieldsplit rtols directly does NOT work on the v3.1.0 container: solve()
# round-trips self.tolerance just before setFromOptions() and re-derives them (UW3 issue
# #477, fixed upstream but not in this image). Verified 2026-08-07 — the readback showed
# 0.001 -> 1e-09. Overriding _INNER_RTOL_MARGIN on the instance does not work either; the
# container reads its margins from the class.
#
# So drive the derivation instead of fighting it: the container computes
# pressure_rtol = tolerance * 0.1, so tolerance = inner_rtol / 0.1 produces the wanted
# inner tolerance THROUGH its own mechanism, which survives the round-trip.
#
# This is safe here because stokes.tolerance does not actually govern outer convergence in
# this configuration: ksp_rtol is unset (PETSc default 1e-5 applies) and snes_rtol is
# satisfied trivially — a linear Stokes solve drops the residual ~2e-10 in one iteration,
# far inside any of these thresholds. Confirm via ksp_its_total == 1 and ksp_reason.
inner_rtol = params.uw_inner_rtol
if inner_rtol > 0:
    _CONTAINER_PRESSURE_MARGIN = 0.1
    stokes.tolerance = inner_rtol / _CONTAINER_PRESSURE_MARGIN
    uw.pprint(
        f"inner fieldsplit rtol target {inner_rtol:g} "
        f"via stokes.tolerance = {stokes.tolerance:g}"
    )

stokes.petsc_options["snes_max_it"] = 3      # linear problem: converges in 1 Newton step
stokes.petsc_options["ksp_monitor"]          = None
stokes.petsc_options["ksp_converged_reason"] = None
stokes.petsc_options["snes_type"]            = "newtonls"
stokes.petsc_options["ksp_type"]             = "fgmres"

stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type",       "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"]              = "redundant"
stokes.petsc_options["fieldsplit_velocity_ksp_type"]                       = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"]             = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"]           = 5
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type",          "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type",       "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# A cap, not a target: the solve converges on tolerance well inside this. It exists so
# a job that stops converging fails fast instead of burning its walltime.
#
# Set UW_MAX_ITS generously for this campaign (~100). If the cap binds, the solve did
# NOT converge and the data point is invalid — ksp_reason in run_info.json records
# DIVERGED_ITS when that happens, so check it before trusting a run.
if max_it > 0:
    stokes.petsc_options["ksp_max_it"] = max_it

stage.pop()

# Snapshot the options BEFORE solving. Comparing this against the after-solve readback
# separates "our value never landed" from "solve() overwrote it" — UW3's solve() pushes
# its own snes_max_it and (in versions predating issue #477) re-derives the inner
# fieldsplit tolerances from self.tolerance just before setFromOptions().
options_before = report_effective_options(stokes, "before first_solve")

# --------------------------------------------------------------------------- #
# STAGE: first_solve  (includes JIT compilation of residuals/Jacobians)       #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("first_solve"); stage.push()

stokes.solve(verbose=True, debug=False)

stage.pop()
solver_stats = {"first_solve": capture_solver_stats(stokes)}
effective_options = report_effective_options(stokes, "after first_solve")

overwritten = {k: (options_before.get(k), effective_options.get(k))
               for k in _OPTIONS_TO_REPORT
               if options_before.get(k) != effective_options.get(k)}
if overwritten:
    uw.pprint("!!! options changed by solve() — these are NOT what was configured:")
    for k, (before, after) in overwritten.items():
        uw.pprint(f"    {k:34s} {before} -> {after}")

# --------------------------------------------------------------------------- #
# STAGE: steady_solves  (JIT-warm re-solve — isolates pure solver cost)       #
# Reset to zero so iteration count matches first_solve.                        #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("steady_solves"); stage.push()

v_uw.data[:] = 0.0
p_uw.data[:] = 0.0
stokes.solve(verbose=True, debug=False)

stage.pop()
solver_stats["steady_solves"] = capture_solver_stats(stokes)

# --------------------------------------------------------------------------- #
# Run metadata (rank 0)                                                        #
# --------------------------------------------------------------------------- #
# Collective — every rank must call this, so it sits outside the rank-0 block below.
placement = rank_placement()

if uw.mpi.rank == 0:
    with open(f"{output_dir}/run_info.json", "w") as fp:
        json.dump({
            "model": "stokes-scaling",
            "res": res,
            "nprocs": uw.mpi.size,
            "scaling": scaling,
            "tolerance_requested": stokes_tol,
            "tolerance_effective": stokes.tolerance,   # differs when inner_rtol is set
            "inner_rtol": inner_rtol if inner_rtol > 0 else None,
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
