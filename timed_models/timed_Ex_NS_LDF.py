# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy
import argparse
import pickle

idx = 0
prev = 0

# %%
resolution      = int(os.getenv("UW_RESOLUTION",16))
nprocs          = int(os.getenv("NTASKS", 1))

# %%
tol             = float(os.getenv("UW_SOL_TOLERANCE",1.e-6))

timing.reset()
timing.start()

# %%
Re_num = 100
wall_res_factor = 0.125
maxsteps = 1
save_every = 5

refinement = 0
qdeg = 3
Vdeg = 2
Pdeg = Vdeg - 1
ns_order = 1
Pcont = False

dt_ns = 0.0004 # originally 0.1

show_vis = False
gen_mesh = True

# %%
outfile = f"NS_LDF_run{idx}"
outdir = f"./NS_LDF_res{resolution}_Re{Re_num}"

# %%
if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    infile = f"NS_LDF_run{prev_idx}"

if uw.mpi.rank == 0:
    os.makedirs(".meshes", exist_ok = True)
    os.makedirs(outdir, exist_ok = True)

# %%
# dimensional quantities
width = 1.
height = 1.
fluid_rho = 1.


# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                 minCoords=(0.0, 0.0),
                                                 maxCoords=(width, height),
                                                 cellSize=1.0 / resolution,
                                                 regular=False,
                                                 qdegree = qdeg
                                         )

# %%
if uw.mpi.size == 1 and show_vis:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )

    pl.show(cpos="xy")

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=Pdeg, continuous = Pcont)

# %%
# passive_swarm = uw.swarm.Swarm(mesh=pipemesh)

if infile is None:
    pass
else:
    v_soln.read_timestep(data_filename = infile, data_name = "U", index = maxsteps, outputPath = outdir)
    p_soln.read_timestep(data_filename = infile, data_name = "P", index = maxsteps, outputPath = outdir)

# %%
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

navier_stokes = uw.systems.NavierStokesSLCN(
    meshbox,
    velocityField = v_soln,
    pressureField = p_soln,
    rho = fluid_rho,
    verbose = True,
    solver_name = "navier_stokes",
    order=ns_order,
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Constant visc
navier_stokes.constitutive_model.Parameters.viscosity = 1./Re_num

navier_stokes.penalty = 0
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc((1.0, 0.0), "Top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Left")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Right")

navier_stokes.tolerance = tol

# %%
# navier_stokes.petsc_options["snes_monitor"] = None
# navier_stokes.petsc_options["snes_converged_reason"] = None
# navier_stokes.petsc_options["snes_monitor_short"] = None
# navier_stokes.petsc_options["ksp_monitor"] = None

# navier_stokes.petsc_options["snes_type"] = "newtonls"
# navier_stokes.petsc_options["ksp_type"] = "fgmres"

# navier_stokes.petsc_options["snes_max_it"] = 50
# navier_stokes.petsc_options["ksp_max_it"] = 50

navier_stokes.petsc_options["snes_monitor"] = None
navier_stokes.petsc_options["ksp_monitor"] = None

navier_stokes.petsc_options["snes_type"] = "newtonls"
navier_stokes.petsc_options["ksp_type"] = "fgmres"

navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %%
ts = 0
elapsed_time = 0.0
timeVal =  np.zeros(maxsteps)*np.nan      # time values

# %%
for step in range(0, maxsteps):

    delta_t = dt_ns

    navier_stokes.solve(timestep=delta_t, zero_init_guess = True)

    elapsed_time += delta_t
    timeVal[step] = elapsed_time

    if uw.mpi.rank == 0:
        print("Timestep {}, t {}, dt {}".format(ts, elapsed_time, delta_t))

    ts += 1

# end timing and save results
timing.stop()

module_timing_data_orig = uw.timing.get_data(group_by="routine")

if uw.mpi.rank == 0:
    print(module_timing_data_orig)

# write out data
filename = f"res{resolution}_nproc{nprocs}"

import json
if module_timing_data_orig:
    module_timing_data = {}
    for key,val in module_timing_data_orig.items():
        module_timing_data[key[0]] = val
    with open(f"{outdir}/{filename}.json", 'w') as fp:
        json.dump(module_timing_data, fp,sort_keys=True, indent=4)

uw.timing.print_table(group_by="routine", output_file=f"{outdir}/{filename}.txt", display_fraction = 1.00)


