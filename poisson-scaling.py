"""
Nonlinear Poisson (diffusion) benchmark — scaling test.

MMS: u_ana = sin(πx)sin(πy)sin(πz), nonlinear diffusivity k = 1 + u²
Solved on a 3D unstructured simplex box.

Timing uses PETSc log stages:
  mesh_setup    — UnstructuredSimplexBox mesh creation
  solver_setup  — Poisson system + MMS source expression + BCs
  first_solve   — single KSP/SNES solve (includes JIT compilation)
  error_analysis — relative L2 norm vs analytical solution

Usage (via launcher):
  mpiexec -n $NTASKS python poisson-scaling.py \
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_tol $UW_SOL_TOLERANCE \
      -uw_maxits $UW_MAX_ITS -uw_job $JOB_IDX -uw_idx $RUN_IDX
"""

import os
import json
import math
from enum import Enum

import sympy
from sympy.vector import CoordSys3D, gradient, divergence
from petsc4py import PETSc

import underworld3 as uw

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
    # "zero" reproduces the original campaign; "analytic" starts u at the MMS solution so
    # the diffusivity k = 1 + u**2 actually varies in space. See the initial-state block
    # before first_solve for why this matters.
    uw_init    = "zero",
)
params.summary("poisson-scaling parameters")

scaling = params.uw_scaling
res     = params.uw_res
tol     = params.uw_tol
job     = params.uw_job
it      = params.uw_idx
max_it  = params.uw_maxits
init_state = params.uw_init


# Read back what PETSc will actually use: UW3 rewrites options during solve(), and v3.2.0's
# preconditioner="auto" can silently fall back from FMG to GAMG (#625).
_OPTIONS_TO_REPORT = ("ksp_type", "ksp_rtol", "ksp_atol", "ksp_max_it",
                      "snes_rtol", "snes_atol", "snes_max_it", "pc_type")


def report_effective_options(solver):
    opts = solver.petsc_options
    return {k: (opts.getString(k) if opts.hasName(k) else None) for k in _OPTIONS_TO_REPORT}


def rank_placement():
    """Collective — every rank must call it. ranks_per_node sets memory bandwidth per rank."""
    from mpi4py import MPI
    from collections import Counter
    per_node = Counter(MPI.COMM_WORLD.allgather(MPI.Get_processor_name()))
    return {"n_nodes": len(per_node),
            "ranks_per_node_max": max(per_node.values()),
            "ranks_per_node_min": min(per_node.values())}


def environment_info(solver):
    """Release identity for the round manifest: image, UW3 version, 32/64-bit PETSc ints."""
    int_type = getattr(PETSc, "IntType", None)
    return {"uw_version": getattr(uw, "__version__", None),
            "sif": os.environ.get("UW_SIF"),
            "petsc_int_type": getattr(int_type, "__name__", str(int_type)),
            "pc_fallbacks": getattr(solver, "pc_fallbacks", None)}   # v3.2.0+, else None

# --------------------------------------------------------------------------- #
# Output directory                                                             #
# --------------------------------------------------------------------------- #
output_base = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name    = os.environ.get("NAME", "out")
mesh_cache  = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")
qdeg        = 3
udeg        = 2

outdir = (
    f"{output_base}/{dir_name}/poisson_out/"
    f"{scaling}_qdeg{qdeg}_udeg{udeg}_tol{tol}_res{res}_job{job}_iter{it}"
)
if uw.mpi.rank == 0:
    os.makedirs(outdir, exist_ok=True)
    uw.pprint(f"output: {outdir}")

# --------------------------------------------------------------------------- #
# STAGE: mesh_setup                                                            #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("mesh_setup"); stage.push()

if uw.mpi.rank == 0:
    os.makedirs(mesh_cache, exist_ok=True)
uw.mpi.barrier()

# Prefer the cached .msh.h5: handed a .msh, Mesh() first runs a serial rank-0 gmsh->h5
# conversion (DMPlexCrFromFile) — 2329 s of the 5621 s mesh_setup at 2197 ranks in round 1.
# The .h5 is read in place; the copy exists only for the .msh path, whose conversion writes
# beside its input so concurrent jobs would race (#565). Same pattern as advdiff-scaling.py.
cache_msh = f"{mesh_cache}/UnstructuredSimplexBox_res{res}.msh"
cache_h5  = f"{cache_msh}.h5"

if os.path.exists(cache_h5) or os.path.exists(cache_msh):
    class _boundaries_3D(Enum):
        Bottom = 11; Top = 12; Right = 13; Left = 14; Front = 15; Back = 16

    class _boundary_normals_3D(Enum):
        Bottom = sympy.Matrix([0, 0, 1]); Top   = sympy.Matrix([0, 0, 1])
        Right  = sympy.Matrix([1, 0, 0]); Left  = sympy.Matrix([1, 0, 0])
        Front  = sympy.Matrix([0, 1, 0]); Back  = sympy.Matrix([0, 1, 0])

    from underworld3.coordinates import CoordinateSystemType

    if os.path.exists(cache_h5):
        mesh_file = cache_h5
    else:
        import shutil
        mesh_file = f"{outdir}/mesh.msh"
        if uw.mpi.rank == 0:
            shutil.copy2(cache_msh, mesh_file)
        uw.mpi.barrier()

    meshbox = uw.discretisation.Mesh(
        mesh_file,
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
    meshbox = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0., 0.),
        maxCoords=(1., 1., 1.),
        cellSize=1.0 / res,
        qdegree=qdeg,
        filename=cache_msh,
    )

if uw.mpi.rank == 0:
    uw.pprint("--- mesh partition ---")
meshbox.dm.view()

u_sol = uw.discretisation.MeshVariable("U", meshbox, 1, degree=udeg)
u_ana = uw.discretisation.MeshVariable("u", meshbox, 1, degree=udeg)

stage.pop()

# Every stage is followed by a timed barrier in its own <name>_sync stage. stage.pop() is
# rank-local, so without it early ranks carry their lead into the next stage, whose first
# collective absorbs it — the artefact behind the retracted eval_nodal result. <name> is the
# slowest rank's work, <name>_sync the arrival spread; they sum to the true span. Costs
# 0.01 s in a 3791 s advdiff run. Rationale in full in advdiff-scaling.py.
stage = PETSc.Log.Stage("mesh_setup_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: solver_setup                                                          #
# Compute MMS source expression symbolically, then build the Poisson system.  #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("solver_setup"); stage.push()

x, y, z = meshbox.N.x, meshbox.N.y, meshbox.N.z

R = CoordSys3D("R")
u_ana_expr  = sympy.sin(sympy.pi * R.x) * sympy.sin(sympy.pi * R.y) * sympy.sin(sympy.pi * R.z)
grad_u      = gradient(u_ana_expr)
src_expr_R  = -divergence(grad_u * (1 + u_ana_expr ** 2))
src_expr    = src_expr_R.simplify().subs(R.x, x).subs(R.y, y).subs(R.z, z)
u_ana_expr  = u_ana_expr.subs(R.x, x).subs(R.y, y).subs(R.z, z)

poisson = uw.systems.Poisson(meshbox, u_Field=u_sol, verbose=True)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1 + u_sol.sym[0] ** 2
poisson.f = src_expr

for face in ("Back", "Front", "Bottom", "Top", "Left", "Right"):
    poisson.add_dirichlet_bc(0., face)

poisson.tolerance = tol                        # snes_rtol
poisson.petsc_options["ksp_rtol"]  = tol       # prevent early KSP convergence
poisson.petsc_options["ksp_atol"]  = tol
poisson.petsc_options["snes_atol"] = tol
if max_it > 0:
    poisson.petsc_options["ksp_max_it"]  = max_it
    poisson.petsc_options["snes_max_it"] = max_it

stage.pop()

stage = PETSc.Log.Stage("solver_setup_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# Initial state for both solve stages.
#
# With uw_init="zero" (the original behaviour) the nonlinearity never acts. The fixed-work
# protocol makes SNES abort at iteration 0 without applying its update, so u stays at zero —
# and at u=0 the Newton Jacobian
#     J.d = -div( k(u) grad d ) - div( k'(u) d grad u ),   k = 1 + u**2,  k' = 2u
# collapses to -div(grad d), the constant-coefficient Laplacian, because k(0)=1 and both
# k'(0) and grad u vanish. The campaign therefore assembles a plain Laplacian, and
# rel_norm comes out as exactly 1.0 (the error is ||0 - u_ana|| / ||u_ana||).
#
# With uw_init="analytic" the field starts at the MMS solution, so k = 1 + u_ana**2 varies
# in space and the assembled operator is genuinely variable-coefficient — while the
# fixed-work protocol is untouched, since the iteration count is still capped.
#
# Both solves below MUST pass zero_init_guess=False. The argument is tri-state and defaults
# to None, which resolves to `not has_solution` — True on a cold solver — and the solve then
# does gvec.array[:] = 0.0, discarding whatever was written here. The first attempt at this
# test set u correctly and still measured u=0, because the solve zeroed it a moment later.
# Passing False is a no-op for uw_init="zero" (copying a zero vector in place of zeroing it),
# so the original campaign's behaviour is preserved exactly.
if init_state == "analytic":
    u_sol.data[:, 0] = uw.function.evaluate(u_ana_expr, u_sol.coords).reshape(-1)
_u_initial = u_sol.data[:, 0].copy()

# --------------------------------------------------------------------------- #
# STAGE: first_solve  (includes JIT compilation of Firedrake kernels)         #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("first_solve"); stage.push()

poisson.solve(zero_init_guess=False)

stage.pop()

stage = PETSc.Log.Stage("first_solve_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

effective_options = report_effective_options(poisson)
uw.pprint(f"effective options after first_solve: {effective_options}")

# --------------------------------------------------------------------------- #
# STAGE: steady_solves  (JIT-warm re-solve — isolates pure solver cost)       #
# Restore the same initial state so the iteration count matches first_solve.  #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("steady_solves"); stage.push()

u_sol.data[:, 0] = _u_initial
poisson.solve(zero_init_guess=False)

stage.pop()

stage = PETSc.Log.Stage("steady_solves_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: error_analysis                                                        #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("error_analysis"); stage.push()

u_ana.data[:, 0] = uw.function.evaluate(u_ana_expr, u_ana.coords).reshape(-1)

u_diff    = (u_sol.sym - u_ana.sym) ** 2
u_ana_sq  = u_ana.sym ** 2
u_diff_l2 = math.sqrt(uw.maths.Integral(meshbox, u_diff).evaluate())
u_ana_l2  = math.sqrt(uw.maths.Integral(meshbox, u_ana_sq).evaluate())
rel_norm  = u_diff_l2 / u_ana_l2

uw.pprint(f"Relative L2 error: {rel_norm:.6e}  ({100*rel_norm:.4f} %)")

placement = rank_placement()   # collective: outside the rank-0 block

if uw.mpi.rank == 0:
    with open(f"{outdir}/errors.json", "w") as fp:
        json.dump({"model": "poisson-scaling", "rel_norm": rel_norm, "res": res,
                   "nprocs": uw.mpi.size, "scaling": scaling,
                   "init_state": init_state,
                   "effective_options": effective_options,
                   "placement": placement,
                   **environment_info(poisson)}, fp, indent=4, default=str)

stage.pop()

stage = PETSc.Log.Stage("error_analysis_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# --------------------------------------------------------------------------- #
# Timing output                                                                #
# --------------------------------------------------------------------------- #
uw.mpi.barrier()
uw.timing.print_table(filename=f"{outdir}/timing.csv")
uw.timing.print_table(filename=f"{outdir}/timing.txt")
uw.pprint(f"Timing written to {outdir}/timing.csv")
