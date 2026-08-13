"""
Checkpoint I/O scaling benchmark.

Measures the parallel write and read performance of the new DMPlex checkpoint
infrastructure (PRs #146, #155, #160) introduced in underworld3.

Workflow:
  1. Build a 3D box mesh and three field variables (velocity vector, pressure, temperature)
  2. Populate them from analytic expressions via uw.function.evaluate (field_setup stage)
  3. Write a checkpoint (io_write stage)
  4. Reload the checkpoint on the same decomposition (io_read stage)
  5. Verify round-trip accuracy (io_verify stage)

There is deliberately NO solver. The bytes written are the same whether a value came
from a Stokes solve or an expression, so a solve would add cost and risk without
changing what is measured — including the tolerance trap that hung the Stokes campaign.

PETSc log stages:
  mesh_setup   — mesh + variable creation
  field_setup  — uw.function.evaluate onto the mesh variables
  eval_nodal   — evaluate at the variable's OWN coordinates (node-local, no search)
  eval_offnode — evaluate at half-cell-offset points (requires point location)
  io_write     — checkpoint write (mesh + variables)
  io_read      — reload all variables from checkpoint
  io_verify    — compare reloaded fields against originals

eval_nodal vs eval_offnode is a controlled test of point location: identical expression,
mesh, rank count and point count, differing only in whether the target points are owned
locally. That is the machinery semi-Lagrangian advection depends on.

Usage (via launcher):
  mpiexec -n $NTASKS python checkpoint-scaling.py \
      -uw_scaling $TYPE -uw_res $UW_RESOLUTION -uw_tol $UW_SOL_TOLERANCE \
      -uw_maxits $UW_MAX_ITS -uw_job $JOB_IDX -uw_idx $RUN_IDX

UW_SOL_TOLERANCE and UW_MAX_ITS are accepted for launcher compatibility but unused —
there is no solve.
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
)
params.summary("checkpoint-scaling parameters")


def rank_placement():
    """How ranks are distributed over nodes.

    For I/O this matters differently than for compute: writes are aggregated per node,
    so ranks_per_node affects how many processes contend for the same filesystem
    connection. Recorded so the placement is in the data rather than inferred.
    """
    from collections import Counter

    per_node = Counter(MPI.COMM_WORLD.allgather(MPI.Get_processor_name()))
    return {
        "n_nodes":            len(per_node),
        "ranks_per_node_max": max(per_node.values()),
        "ranks_per_node_min": min(per_node.values()),
    }


scaling = params.uw_scaling
res     = params.uw_res
tol     = params.uw_tol
job     = params.uw_job
it      = params.uw_idx
max_it  = params.uw_maxits

uw.pprint(f"scaling={scaling}  res={res}  tol={tol}  job={job}  idx={it}  maxits={max_it}")

# --------------------------------------------------------------------------- #
# Output directory                                                             #
# --------------------------------------------------------------------------- #
output_base = os.environ.get("OUTPUT_BASE", "/scratch/el06/jg0883")
dir_name    = os.environ.get("NAME", "out")
vdegree, pdegree = 2, 1
qdeg        = max(pdegree, vdegree)
mesh_cache  = os.environ.get("MESH_CACHE", f"{output_base}/mesh_cache")

output_dir = (
    f"{output_base}/{dir_name}/checkpoint_out/"
    f"{scaling}_vdeg{vdegree}_pdeg{pdegree}_res{res}_job{job}_iter{it}"
)
chkpt_base = f"chkpt_res{res}_job{job}_iter{it}"

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
        cellSize=1.0 / res,
        qdegree=qdeg,
        filename=cache_file,
    )

v_soln = uw.discretisation.MeshVariable("U", mesh, 3, degree=vdegree, vtype=uw.VarType.VECTOR)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=pdegree, continuous=True)
# A scalar field to make the I/O test more representative
T_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=pdegree, continuous=True)

if uw.mpi.rank == 0:
    uw.pprint("--- mesh partition ---")
mesh.dm.view()

stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: field_setup                                                           #
#                                                                              #
# No solver. This campaign measures I/O, and the bytes written are identical    #
# whether a value came from a Stokes solve or an expression — so a solve would  #
# only add cost and risk. It would also inherit the tolerance trap that hung    #
# the Stokes campaign: UW3 derives the inner fieldsplit rtols from              #
# stokes.tolerance, and an unreachable value makes both inner solves run their  #
# full 200 iterations against a matrix-free Schur complement.                   #
#                                                                              #
# uw.function.evaluate is timed as its own stage because it is used pervasively #
# in real UW3 scripts (initial conditions, diagnostics) and has never been      #
# measured at scale. Evaluating at a variable's OWN coords should be node-local #
# and scale well; evaluating at arbitrary points would need the same point      #
# location that makes semi-Lagrangian advection scale badly.                    #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("field_setup"); stage.push()

x, y, z = mesh.X

# Smooth, non-constant fields. Non-constant matters in case the HDF5 layer ever
# compresses — a constant field would compress away and understate write cost.
v_expr = sympy.Matrix([
    sympy.sin(sympy.pi * y) * sympy.cos(sympy.pi * z),
    sympy.sin(sympy.pi * z) * sympy.cos(sympy.pi * x),
    sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y),
])
for _i in range(3):
    v_soln.data[:, _i] = uw.function.evaluate(v_expr[_i], v_soln.coords).reshape(-1)

p_soln.data[:, 0] = uw.function.evaluate(
    sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y), p_soln.coords).reshape(-1)

T_soln.data[:, 0] = uw.function.evaluate(1.0 - z, T_soln.coords).reshape(-1)

stage.pop()

# --------------------------------------------------------------------------- #
# STAGE: eval_nodal / eval_offnode                                             #
#                                                                              #
# Same expression, same mesh, same rank count, same number of points — the ONLY #
# difference is where they sit. At a variable's own coordinates the value is    #
# already owned locally, so no search is needed. At arbitrary points the owning  #
# cell must be located first, which is the machinery semi-Lagrangian advection   #
# uses and which scales poorly (SLCN cost grew 7.7x from 1 to 1000 ranks while   #
# solver work stayed fixed). If eval_offnode degrades and eval_nodal does not,   #
# point location is confirmed as the mechanism.                                 #
#                                                                              #
# The offset is half a cell — the same order as SLCN characteristic travel, so   #
# most probes stay rank-local and a minority cross a boundary.                   #
# --------------------------------------------------------------------------- #
# The probe MUST reference a mesh variable, not just coordinates. A pure coordinate
# expression like sin(pi*x)*sin(pi*y)*sin(pi*z) can be evaluated by direct substitution
# with no mesh involvement at all — so timing it would say nothing about point location.
# T_soln.sym forces the real path: find the owning cell, then interpolate its dofs.
_probe_expr = T_soln.sym[0]

# Warm-up, untimed. The first evaluate of an expression pays JIT compilation, so without
# this the FIRST timed stage absorbs it and the comparison inverts — the serial run showed
# eval_nodal 6.2 ms vs eval_offnode 2.7 ms, i.e. the stage that ran second looked faster.
uw.function.evaluate(_probe_expr, T_soln.coords)

stage = PETSc.Log.Stage("eval_nodal"); stage.push()
_nodal = uw.function.evaluate(_probe_expr, T_soln.coords)
stage.pop()

# Clipped to stay inside the unit box; points outside the domain have no owning cell.
_off = np.clip(T_soln.coords + 0.5 / res, 1.0e-6, 1.0 - 1.0e-6)

stage = PETSc.Log.Stage("eval_offnode"); stage.push()
_offnode = uw.function.evaluate(_probe_expr, _off)
stage.pop()

# CORRECTNESS CHECK, untimed.
#
# eval_offnode came out the same cost as eval_nodal at every rank count up to 1000, which
# would mean locating a cell for an arbitrary point is free. The alternative reading is
# that off-rank points are not being resolved at all and evaluate is quietly returning
# something wrong. Timings cannot distinguish those, so compare against the closed form.
#
# At 1000 ranks a half-cell offset should push roughly 6% of points across a subdomain
# boundary, so if the distributed search were broken this error would be O(1), not O(1e-16).
# T_soln was set to (1 - z), which is linear, and it is a P1 field — so P1 interpolation
# reproduces it EXACTLY at any point inside the mesh. Any departure from (1 - z) therefore
# means the value came from the wrong cell, or from no cell at all.
_analytic = 1.0 - _off[:, 2]
_off_vals = np.asarray(_offnode).reshape(-1)
_local_err = float(np.max(np.abs(_off_vals - _analytic))) if _off_vals.size else 0.0
_offnode_err = uw.mpi.comm.allreduce(_local_err, op=MPI.MAX)
_n_nonfinite = int(uw.mpi.comm.allreduce(
    int(np.count_nonzero(~np.isfinite(_off_vals))), op=MPI.SUM))

uw.pprint(f"eval_offnode max |error| vs analytic: {_offnode_err:.3e}"
          f"   non-finite values: {_n_nonfinite}")

_n_probe = int(uw.mpi.comm.allreduce(len(T_soln.coords), op=MPI.SUM))
uw.pprint(f"evaluate probes: {_n_probe} points, nodal and half-cell-offset")

# --------------------------------------------------------------------------- #
# STAGE: io_write                                                              #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("io_write"); stage.push()

mesh.write_timestep(
    chkpt_base,
    meshUpdates=True,
    meshVars=[v_soln, p_soln, T_soln],
    outputPath=output_dir,
    index=0,
)

stage.pop()
uw.pprint("Checkpoint written.")

# --------------------------------------------------------------------------- #
# STAGE: io_read                                                               #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("io_read"); stage.push()

# Reload into new variables on the SAME mesh so partitioning matches v_soln/p_soln/T_soln
# and the direct .data comparison in io_verify is valid.
v_rd = uw.discretisation.MeshVariable("U_rd", mesh, 3, degree=vdegree, vtype=uw.VarType.VECTOR)
p_rd = uw.discretisation.MeshVariable("P_rd", mesh, 1, degree=pdegree, continuous=True)
T_rd = uw.discretisation.MeshVariable("T_rd", mesh, 1, degree=pdegree, continuous=True)

v_rd.read_timestep(chkpt_base, "U", 0, outputPath=output_dir)
p_rd.read_timestep(chkpt_base, "P", 0, outputPath=output_dir)
T_rd.read_timestep(chkpt_base, "T", 0, outputPath=output_dir)

stage.pop()
uw.pprint("Checkpoint reloaded.")

# --------------------------------------------------------------------------- #
# STAGE: io_verify  (round-trip accuracy)                                     #
# --------------------------------------------------------------------------- #
stage = PETSc.Log.Stage("io_verify"); stage.push()

def _max_abs_err(a, b):
    diff = np.abs(a - b)
    local_max = float(np.max(diff)) if diff.size > 0 else 0.0
    return uw.mpi.comm.allreduce(local_max, op=MPI.MAX)

v_err = _max_abs_err(v_rd.data, v_soln.data)
p_err = _max_abs_err(p_rd.data, p_soln.data)
T_err = _max_abs_err(T_rd.data, T_soln.data)

uw.pprint(f"Round-trip max abs error — v: {v_err:.2e}  p: {p_err:.2e}  T: {T_err:.2e}")

stage.pop()

# --------------------------------------------------------------------------- #
# Write summary                                                                #
# --------------------------------------------------------------------------- #
# Collective — every rank must call this, so it sits outside the rank-0 block below.
placement = rank_placement()

# Bytes actually written, so io_write time can be turned into a bandwidth figure.
# Measured on rank 0 and summed, because the checkpoint is one shared file set.
_chkpt_bytes = 0
if uw.mpi.rank == 0:
    for _root, _dirs, _files in os.walk(output_dir):
        for _f in _files:
            if _f.endswith((".h5", ".xdmf")):
                _chkpt_bytes += os.path.getsize(os.path.join(_root, _f))

if uw.mpi.rank == 0:
    with open(f"{output_dir}/run_info.json", "w") as fp:
        json.dump({
            "model": "checkpoint-scaling",
            "res": res,
            "nprocs": uw.mpi.size,
            "scaling": scaling,
            "placement": placement,
            "checkpoint_bytes": _chkpt_bytes,
            "n_eval_points": _n_probe,
            "eval_offnode_max_err": _offnode_err,
            "eval_offnode_nonfinite": _n_nonfinite,
            "roundtrip_v_err": v_err,
            "roundtrip_p_err": p_err,
            "roundtrip_T_err": T_err,
        }, fp, indent=4)

# --------------------------------------------------------------------------- #
# Timing output                                                                #
# --------------------------------------------------------------------------- #
uw.mpi.barrier()
uw.timing.print_table(filename=f"{output_dir}/timing.csv")
uw.timing.print_table(filename=f"{output_dir}/timing.txt")
uw.pprint(f"Timing written to {output_dir}/timing.csv")
