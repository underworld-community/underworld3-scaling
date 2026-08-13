"""
Navier-Stokes scaling benchmark.

Tests the SNES_NavierStokes solver (inertial terms + SL momentum DDt) on a 3D
lid-driven cavity — a different code path from stokes-scaling.py since
inertial terms add a non-linear advective correction at every timestep.

PETSc log stages:
  mesh_setup    — mesh + variable creation
  solver_setup  — NavierStokes system + constitutive model + BCs
  first_solve   — step 0 (includes JIT compilation of residuals/Jacobians)
  steady_solves — steps 1 … UW_MAX_ITS-1 (hot path; primary scaling metric)

Usage (via launcher):
  mpiexec -n $NTASKS python ns-scaling.py \
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_tol $UW_SOL_TOLERANCE \
      -uw_maxits $UW_MAX_ITS -uw_job $JOB_IDX -uw_idx $RUN_IDX \
      [-uw_re $UW_NS_RE]
"""

import os
import json

import sympy
from petsc4py import PETSc
from enum import Enum

import underworld3 as uw
from underworld3.systems import NavierStokes

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
    uw_tol     = 1e-6,
    uw_job     = 0,
    uw_idx     = 0,
    uw_maxits  = 10,
    uw_re      = 100.0,
)
params.summary("ns-scaling parameters")

scaling = params.uw_scaling
res     = params.uw_res
tol     = params.uw_tol
job     = params.uw_job
it      = params.uw_idx
max_it  = params.uw_maxits
Re      = params.uw_re

# dt = CFL ≈ 0.5 with unit lid velocity and characteristic cell size
cellsize = 1.0 / res
dt       = 0.5 * cellsize / 1.0

uw.pprint(
    f"scaling={scaling}  res={res}  tol={tol}  job={job}  idx={it}  "
    f"maxits={max_it}  Re={Re}  dt={dt:.4f}"
)

# --------------------------------------------------------------------------- #
# Output directory                                                             #
# --------------------------------------------------------------------------- #
output_base = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name    = os.environ.get("NAME", "out")
vdegree, pdegree = 2, 1
qdeg        = max(pdegree, vdegree)
mesh_cache  = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")

output_dir = (
    f"{output_base}/{dir_name}/ns_out/"
    f"{scaling}_re{int(Re)}_vdeg{vdegree}_pdeg{pdegree}_res{res}_job{job}_iter{it}"
)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    uw.pprint(f"output: {output_dir}")

# --------------------------------------------------------------------------- #
# STAGE: mesh_setup                                                            #
# --------------------------------------------------------------------------- #
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
        cellSize=cellsize,
        qdegree=qdeg,
        filename=cache_file,
    )

v = uw.discretisation.MeshVariable("U", mesh, 3, degree=vdegree, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=pdegree, continuous=False)

if uw.mpi.rank == 0:
    uw.pprint("--- mesh partition ---")
mesh.dm.view()

stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: solver_setup                                                          #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("solver_setup"); stage.push()

# rho=1.0 is required — default is 0.0 which disables inertia
ns = NavierStokes(mesh, velocityField=v, pressureField=p, rho=1.0, order=2)
ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
ns.constitutive_model.Parameters.viscosity = 1.0 / Re   # mu = rho * U * L / Re = 1/Re
ns.saddle_preconditioner = 1.0
ns.tolerance = tol
ns.petsc_options["ksp_rtol"]  = tol
ns.petsc_options["ksp_atol"]  = tol
ns.petsc_options["snes_atol"] = tol
ns.petsc_options["ksp_type"]  = "fgmres"
ns.petsc_options["snes_type"] = "newtonls"
if max_it > 0:
    ns.petsc_options["ksp_max_it"]  = max_it
    ns.petsc_options["snes_max_it"] = max_it

# Lid-driven cavity: top face moves at u_x=1; all other faces no-slip
ns.add_essential_bc(sympy.Matrix([1.0, 0.0, 0.0]), "Top")
ns.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), "Bottom")
ns.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), "Left")
ns.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), "Right")
ns.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), "Front")
ns.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), "Back")
ns.bodyforce = sympy.Matrix([0.0, 0.0, 0.0])

stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: first_solve  (includes JIT compilation)                              #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("first_solve"); stage.push()

ns.solve(timestep=dt, zero_init_guess=True)

stage.pop()

uw.pprint("First solve complete.")

# --------------------------------------------------------------------------- #
# STAGE: steady_solves  (hot path — the scaling metric)                       #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("steady_solves"); stage.push()

n_steady = max(max_it - 1, 1)
for step in range(n_steady):
    ns.solve(timestep=dt, zero_init_guess=False)

stage.pop()

uw.pprint(f"Completed {n_steady} steady timesteps at Re={Re}.")

# --------------------------------------------------------------------------- #
# Run metadata (rank 0)                                                        #
# --------------------------------------------------------------------------- #
if uw.mpi.rank == 0:
    with open(f"{output_dir}/run_info.json", "w") as fp:
        json.dump({
            "model":    "ns-scaling",
            "res":      res,
            "nprocs":   uw.mpi.size,
            "scaling":  scaling,
            "Re":       Re,
            "dt":       dt,
            "n_steady_steps": n_steady,
            "tol":      tol,
        }, fp, indent=4)

# --------------------------------------------------------------------------- #
# Timing output                                                                #
# --------------------------------------------------------------------------- #
uw.mpi.barrier()
uw.timing.print_table(filename=f"{output_dir}/timing.csv")
uw.timing.print_table(filename=f"{output_dir}/timing.txt")
uw.pprint(f"Timing written to {output_dir}/timing.csv")
