"""
Viscoelastic-plastic (VEP) Stokes scaling benchmark.

Tests the ETD-1 and BDF-1 integrators for ViscoElasticPlasticFlowModel on a
3D box with a fault-layer yield stress — the "killer test" from the ETD design
document (docs/developer/design/EXPONENTIAL_VE_INTEGRATOR.md).

PETSc log stages:
  mesh_setup      — mesh + variable creation + passive tracer swarm init
  solver_setup    — Stokes + constitutive model + BCs + swarm populate
  first_solve     — step 0 (includes JIT compilation of residuals/Jacobians)
  steady_solves   — steps 1 … UW_MAX_ITS-1 (hot path; the primary scaling metric)
  swarm_advection — particle advection + MPI migration per step (nested inside steady_solves loop)
  io              — checkpoint write

Usage (via launcher):
  mpiexec -n $NTASKS python vep-scaling.py \
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_tol $UW_SOL_TOLERANCE \
      -uw_maxits $UW_MAX_ITS -uw_job $JOB_IDX -uw_idx $RUN_IDX \
      -uw_integrator $UW_VEP_INTEGRATOR [-uw_memprobe true]
"""

import os
import json
import csv

import numpy as np
import sympy
from petsc4py import PETSc
from enum import Enum

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
    uw_scaling    = "none",
    uw_res        = 8,
    uw_tol        = 1e-6,
    uw_job        = 0,
    uw_idx        = 0,
    uw_maxits     = 10,
    uw_integrator = "etd",
    uw_memprobe   = False,
)
params.summary("vep-scaling parameters")

scaling     = params.uw_scaling
res         = params.uw_res
tol         = params.uw_tol
job         = params.uw_job
it          = params.uw_idx
max_it      = params.uw_maxits
integrator  = params.uw_integrator
memprobe_on = params.uw_memprobe

# --------------------------------------------------------------------------- #
# Optional memory diagnostics                                                  #
# --------------------------------------------------------------------------- #
memprobe_rows = []

if memprobe_on:
    from underworld3.utilities import memprobe
    memprobe.enable()
    memprobe.dump_petsc_leaks_at_finalize(None)  # PETSc leak summary at exit

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
output_base = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name    = os.environ.get("NAME", "out")
vdegree, pdegree = 2, 1
qdeg        = max(pdegree, vdegree)
mesh_cache  = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")

output_dir = (
    f"{output_base}/{dir_name}/vep_out/"
    f"{scaling}_integrator{integrator}_vdeg{vdegree}_pdeg{pdegree}"
    f"_tol{tol}_res{res}_job{job}_iter{it}"
)
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    uw.pprint(f"output: {output_dir}")

# Elastic timestep — constant across all steps for fixed-work comparison
dt_elastic = 0.02

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

v = uw.discretisation.MeshVariable("U", mesh, 3, degree=vdegree, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=pdegree, continuous=True)

passive_swarm = uw.swarm.Swarm(mesh=mesh)

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

stokes = Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    stokes.Unknowns, order=1, integrator=integrator
)
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.constitutive_model.Parameters.shear_modulus     = 1.0
stokes.constitutive_model.Parameters.shear_viscosity_min    = 1.0e-3
stokes.constitutive_model.Parameters.strainrate_inv_II_min  = 1.0e-10

# Fault-layer yield stress: weak zone in the middle third in z
x, y, z = mesh.X
tau_y = sympy.Piecewise(
    (0.3,   (z >= 0.45) & (z <= 0.55)),
    (1.0e6, True),
)
stokes.constitutive_model.Parameters.yield_stress = tau_y

stokes.saddle_preconditioner = 1.0
stokes.tolerance = tol
stokes.petsc_options["ksp_rtol"]  = tol
stokes.petsc_options["ksp_atol"]  = tol
stokes.petsc_options["snes_atol"] = tol
stokes.petsc_options["ksp_type"]  = "fgmres"
stokes.petsc_options["snes_type"] = "newtonls"
if max_it > 0:
    stokes.petsc_options["ksp_max_it"]  = max_it
    stokes.petsc_options["snes_max_it"] = max_it

# Boundary conditions: shear-driven flow
stokes.add_essential_bc(sympy.Matrix([0.5, 0.0, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0, 0.0), "Right")
stokes.add_essential_bc((sympy.oo, sympy.oo, 0.0), "Front")
stokes.add_essential_bc((sympy.oo, sympy.oo, 0.0), "Back")
stokes.bodyforce = sympy.Matrix([0.0, 0.0, 0.0])

passive_swarm.populate(fill_param=4)

stage.pop()
_record_rss("solver_setup", rss_before, _rss())

# --------------------------------------------------------------------------- #
# STAGE: first_solve  (includes JIT compilation)                              #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("first_solve"); stage.push()

stokes.solve(timestep=dt_elastic, zero_init_guess=True)

stage.pop()
_record_rss("first_solve", rss_before, _rss())

# --------------------------------------------------------------------------- #
# STAGE: steady_solves  (hot path — the scaling metric)                       #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("steady_solves"); stage.push()

n_steady = max(max_it - 1, 1)
for step in range(n_steady):
    stokes.solve(timestep=dt_elastic, zero_init_guess=False)

    stage_sw = PETSc.Log.Stage("swarm_advection"); stage_sw.push()
    passive_swarm.advection(v.sym, dt_elastic, order=2)
    stage_sw.pop()

    if memprobe_on and uw.mpi.rank == 0:
        memprobe_rows.append({
            "stage": f"steady_step_{step}",
            "rss_delta_mib": _rss(),
        })

stage.pop()
_record_rss("steady_solves", rss_before, _rss())

uw.pprint(f"Completed {n_steady} steady timesteps with integrator={integrator}")

# --------------------------------------------------------------------------- #
# STAGE: io  (checkpoint write)                                               #
# --------------------------------------------------------------------------- #
rss_before = _rss()
stage = PETSc.Log.Stage("io"); stage.push()

mesh.write_timestep(
    f"vep_res{res}",
    meshUpdates=True,
    meshVars=[v, p],
    outputPath=output_dir,
    index=0,
)

stage.pop()
_record_rss("io", rss_before, _rss())

# --------------------------------------------------------------------------- #
# Write memprobe CSV (rank 0 only)                                            #
# --------------------------------------------------------------------------- #
if memprobe_on and uw.mpi.rank == 0 and memprobe_rows:
    with open(f"{output_dir}/memprobe.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(memprobe_rows[0].keys()))
        writer.writeheader()
        writer.writerows(memprobe_rows)

# --------------------------------------------------------------------------- #
# Run metadata (rank 0)                                                       #
# --------------------------------------------------------------------------- #
if uw.mpi.rank == 0:
    with open(f"{output_dir}/run_info.json", "w") as fp:
        json.dump({
            "model": "vep-scaling",
            "integrator": integrator,
            "res": res,
            "nprocs": uw.mpi.size,
            "scaling": scaling,
            "n_steady_steps": n_steady,
            "dt_elastic": dt_elastic,
            "tol": tol,
        }, fp, indent=4)

# --------------------------------------------------------------------------- #
# Timing output                                                                #
# --------------------------------------------------------------------------- #
uw.mpi.barrier()
uw.timing.print_table(filename=f"{output_dir}/timing.csv")
uw.timing.print_table(filename=f"{output_dir}/timing.txt")
uw.pprint(f"Timing written to {output_dir}/timing.csv")
