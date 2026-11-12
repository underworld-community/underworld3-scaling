"""
Dense swarm advection scaling benchmark.

A particle swarm filling every cell (UW_FILL points per dimension per cell) is advected
through a steady 3D cellular flow for UW_NSTEPS steps. There is no solve. The cost is the
particle half of a PIC model, isolated: velocity evaluation at particle positions
(global_evaluate — cell search, interpolation, off-rank points resolved collectively), the
RK2 position update, and the migration of particles that crossed a rank boundary.

Velocity: a divergence-free cell with zero normal flow on every wall, so no particle leaves
the box and the count is conserved:

    v = ( sin(πx) cos(πy) cos(πz),  -½ cos(πx) sin(πy) cos(πz),  -½ cos(πx) cos(πy) sin(πz) )

It is carried on a degree-2 mesh variable rather than used as a coordinate expression, so
every evaluation goes through the mesh (find the owning cell, interpolate its dofs) — the
path a velocity from a Stokes solve takes. A pure coordinate expression could be evaluated
by substitution with no mesh involvement, and would measure nothing.

Timestep: |v| <= 1, so dt = 0.5/res moves the fastest particles half a cell per step. Under
weak scaling (res = BASE*i, ranks = i^3) the per-rank subdomain keeps the same cell count,
so the FRACTION of particles that change rank each step is held fixed across the sweep.

PETSc log stages:
  mesh_setup    — mesh + velocity variable
  swarm_setup   — swarm + one material variable, populate, velocity assignment
  first_advect  — step 0 (includes JIT compilation of the velocity expression)
  advect_steps  — UW_NSTEPS further steps (the hot path / scaling metric)
Every stage is followed by a timed barrier in its own <name>_sync stage (benchmark v2).

Usage (via launcher):
  mpiexec -n $NTASKS python swarm-scaling.py \
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_nsteps $UW_NSTEPS -uw_fill $UW_FILL \
      -uw_job $JOB_IDX -uw_idx $RUN_IDX

UW_SOL_TOLERANCE and UW_MAX_ITS are accepted for launcher compatibility but unused.
"""

import os
import json

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
    uw_scaling = "none",
    uw_res     = 8,
    uw_tol     = 1e-6,
    uw_job     = 0,
    uw_idx     = 0,
    uw_maxits  = 10,
    uw_nsteps  = 10,
    uw_fill    = 3,
)
params.summary("swarm-scaling parameters")


def environment_info():
    """Release identity for the round manifest: image, UW3 version, 32/64-bit PETSc ints."""
    int_type = getattr(PETSc, "IntType", None)
    return {"uw_version": getattr(uw, "__version__", None),
            "sif": os.environ.get("UW_SIF"),
            "petsc_int_type": getattr(int_type, "__name__", str(int_type)),
            "pc_fallbacks": None}   # no solver in this model


def rank_placement():
    from collections import Counter

    per_node = Counter(MPI.COMM_WORLD.allgather(MPI.Get_processor_name()))
    return {
        "n_nodes":            len(per_node),
        "ranks_per_node_max": max(per_node.values()),
        "ranks_per_node_min": min(per_node.values()),
    }


def particle_census(swarm):
    """Global count and the per-rank spread — the imbalance a flow creates as it drives
    particles into some subdomains and out of others. Collective."""
    n = int(max(swarm.local_size, 0))
    comm = uw.mpi.comm
    return {
        "global":    comm.allreduce(n, op=MPI.SUM),
        "local_max": comm.allreduce(n, op=MPI.MAX),
        "local_min": comm.allreduce(n, op=MPI.MIN),
    }


scaling = params.uw_scaling
res     = params.uw_res
job     = params.uw_job
it      = params.uw_idx
nsteps  = params.uw_nsteps
fill    = params.uw_fill

uw.pprint(f"scaling={scaling}  res={res}  job={job}  idx={it}  nsteps={nsteps}  fill={fill}")

# --------------------------------------------------------------------------- #
# Output directory                                                             #
# --------------------------------------------------------------------------- #
output_base = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name    = os.environ.get("NAME", "out")
vdeg        = 2
qdeg        = 2
mesh_cache  = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")

output_dir = (
    f"{output_base}/{dir_name}/swarm_out/"
    f"{scaling}_fill{fill}_vdeg{vdeg}_res{res}_job{job}_iter{it}"
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

# Same cached box as poisson/advdiff (the cache name does not encode qdegree), read as
# .msh.h5 in place — see advdiff-scaling.py for why the .msh path is avoided.
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
        mesh_file = f"{output_dir}/mesh.msh"
        if uw.mpi.rank == 0:
            shutil.copy2(cache_msh, mesh_file)
        uw.mpi.barrier()

    mesh = uw.discretisation.Mesh(
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
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0., 0.),
        maxCoords=(1., 1., 1.),
        cellSize=1.0 / res,
        qdegree=qdeg,
        filename=cache_msh,
    )

V = uw.discretisation.MeshVariable("V", mesh, 3, degree=vdeg, vtype=uw.VarType.VECTOR)

if uw.mpi.rank == 0:
    uw.pprint("--- mesh partition ---")
mesh.dm.view()

stage.pop()

stage = PETSc.Log.Stage("mesh_setup_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: swarm_setup                                                           #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("swarm_setup"); stage.push()

x, y, z = mesh.X
pi = sympy.pi
v_expr = sympy.Matrix([
           sympy.sin(pi * x) * sympy.cos(pi * y) * sympy.cos(pi * z),
    -0.5 * sympy.cos(pi * x) * sympy.sin(pi * y) * sympy.cos(pi * z),
    -0.5 * sympy.cos(pi * x) * sympy.cos(pi * y) * sympy.sin(pi * z),
])
for d in range(3):
    V.data[:, d] = uw.function.evaluate(v_expr[d], V.coords).reshape(-1)

swarm = uw.swarm.Swarm(mesh=mesh)
# One scalar carried by the particles, so migration moves data as well as coordinates.
# Its mesh proxy is never touched (.sym is not read), so no projection cost hides here.
material = uw.swarm.SwarmVariable("M", swarm, 1, proxy_degree=1)
swarm.populate(fill_param=fill)
material.data[...] = 1.0

census_initial = particle_census(swarm)
cells_local = mesh.dm.getHeightStratum(0)
cells_global = uw.mpi.comm.allreduce(int(cells_local[1] - cells_local[0]), op=MPI.SUM)
uw.pprint(f"particles: {census_initial['global']} over {cells_global} cells "
          f"({census_initial['global'] / max(cells_global, 1):.1f} per cell); "
          f"per rank max {census_initial['local_max']} min {census_initial['local_min']}")

# |v| <= 1 by construction: half a cell per step for the fastest particle.
dt = 0.5 * (1.0 / res)

stage.pop()

stage = PETSc.Log.Stage("swarm_setup_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: first_advect  (includes JIT compilation)                             #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("first_advect"); stage.push()

swarm.advection(V.sym, dt, order=2)

stage.pop()

stage = PETSc.Log.Stage("first_advect_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: advect_steps  (hot path)                                             #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("advect_steps"); stage.push()

for step in range(nsteps):
    swarm.advection(V.sym, dt, order=2)

stage.pop()

stage = PETSc.Log.Stage("advect_steps_sync"); stage.push()
uw.mpi.barrier()
stage.pop()

uw.pprint(f"Completed {nsteps} advection steps")

# --------------------------------------------------------------------------- #
# Sanity: particle conservation and the imbalance the flow created            #
# --------------------------------------------------------------------------- #
census_final = particle_census(swarm)
lost = census_initial["global"] - census_final["global"]
uw.pprint(f"particles lost: {lost} of {census_initial['global']}; per rank max "
          f"{census_final['local_max']} min {census_final['local_min']}")

# --------------------------------------------------------------------------- #
# Run metadata                                                                 #
# --------------------------------------------------------------------------- #
placement = rank_placement()

if uw.mpi.rank == 0:
    with open(f"{output_dir}/run_info.json", "w") as fp:
        json.dump({
            "model": "swarm-scaling",
            "res": res,
            "nprocs": uw.mpi.size,
            "scaling": scaling,
            "fill_param": fill,
            "n_steps": nsteps,
            "dt": dt,
            "cells_global": cells_global,
            "particles_initial": census_initial,
            "particles_final": census_final,
            "particles_lost": lost,
            "placement": placement,
            **environment_info(),
        }, fp, indent=4, default=str)

# --------------------------------------------------------------------------- #
# Timing output                                                                #
# --------------------------------------------------------------------------- #
uw.mpi.barrier()
uw.timing.print_table(filename=f"{output_dir}/timing.csv")
uw.timing.print_table(filename=f"{output_dir}/timing.txt")
