# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw3-venv-run
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Navier Stokes test: flow around a circular inclusion (2D)
# timing script
#

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

circle_res = 0.007
refinement = 0
model = 1
save_every = 1

maxsteps = 1

order = 2
qdeg = 3
Vdeg = 2
Pdeg = Vdeg - 1
Pcont = False
gen_mesh = True

# %%
if model == 1:
    U0 = 0.3
    delta_t = 0.05
    expt_name = f"NS_benchmark_DFG2d_SLCN_1_{resolution}"
elif model == 2:
    U0 = 0.3
    delta_t = 0.1
    expt_name = f"NS_benchmark_DFG2d_SLCN_1_ss_{resolution}"
elif model == 3:
    U0 = 1.5
    delta_t = 0.03
    expt_name = f"NS_benchmark_DFG2d_SLCN_2_{resolution}"
elif model == 4:
    U0 = 3.75
    expt_name = f"NS_test_Re_250_SLCN_{resolution}"
elif model == 5:
    U0 = 15
    expt_name = f"NS_test_Re_1000i_SLCN_{resolution}"

outfile = f"{expt_name}_run{idx}"
outdir = f"./{expt_name}_res{resolution}_nprocs{nprocs}"

# %%
if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    infile = f"{expt_name}_run{prev_idx}"

if uw.mpi.rank == 0:
    os.makedirs(".meshes", exist_ok=True)
    os.makedirs(f"{outdir}", exist_ok=True)

# %%
# dimensional quantities
width = 2.2
height = 0.41
radius = 0.05
dyn_visc = 0.001
fluid_rho = 1
kin_visc = 0.001

centre = (0.2, 0.205)

# %%
# unit registry for ease in converting between units
u = uw.scaling.units

ndim = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise

# KL = height * u.meter
# Kt = ((KL)**2) / (kin_visc * u.meter**2/u.second)
KL = radius * u.meter
Kt = KL / (U0 * u.meter/u.second)
KM = (dyn_visc * u.pascal * u.second) * KL * Kt

scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"] = KM
scaling_coefficients

# %%
minX, maxX = 0, ndim(width * u.meter)
minY, maxY = 0, ndim(height * u.meter)

if uw.mpi.rank == 0:
    print("min X, max X:", minX, maxX)
    print("min Y, max Y:", minY, maxY)
    print("ndim kinematic viscosity: ", ndim(kin_visc * u.meter**2/u.second))
    print("ndim fluid density: ", ndim(fluid_rho * u.kilogram / u.meter**3))
    print("ndim dynamic viscosity: ", ndim((kin_visc * u.meter**2/u.second)*(fluid_rho * u.kilogram / u.meter**3)))
    print("ndim U0: ", ndim(U0 * u.meter / u.second))

# %%
# cell size calculation

csize = ndim (height * u.meter) / resolution
csize_circle = circle_res * csize
res = csize_circle

# %%
import pygmsh
from enum import Enum

## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh

class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left  = 4
    inclusion = 5
    All_Boundaries = 1001


def pipemesh_mesh_refinement_callback(dm):

    r_p = ndim(radius * u.meter)

    # print(f"Refinement callback - spherical", flush=True)

    c2 = dm.getCoordinatesLocal()
    coords = c2.array.reshape(-1, 2) - (ndim(centre[0] * u.meter), ndim(centre[1] * u.meter))

    R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2).reshape(-1, 1)

    pipeIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        dm, "inclusion"
    )

    coords[pipeIndices] *= r_p / R[pipeIndices]
    coords = coords + (ndim(centre[0] * u.meter), ndim(centre[1] * u.meter))

    c2.array[...] = coords.reshape(-1)
    dm.setCoordinatesLocal(c2)

    return

## Restore inflow samples to inflow points
def pipemesh_return_coords_to_bounds(coords):
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = ndim(0.0001 * u.meter)

    return coords

if uw.mpi.rank == 0 and infile is None and gen_mesh:
    # Generate local mesh on boss process

    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (ndim(centre[0] * u.meter), ndim(centre[1] * u.meter), 0.0),
            ndim(radius * u.meter),
            make_surface=False,
            mesh_size=csize_circle,
        )
        domain = geom.add_rectangle(
            xmin = 0.0,
            ymin = 0.0,
            xmax = ndim(width * u.meter),
            ymax = ndim(height * u.meter),
            z=0,
            holes=[inclusion],
            mesh_size=csize,
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label=boundaries.bottom.name)
        geom.add_physical(domain.surface.curve_loop.curves[1], label=boundaries.right.name)
        geom.add_physical(domain.surface.curve_loop.curves[2], label=boundaries.top.name)
        geom.add_physical(domain.surface.curve_loop.curves[3], label=boundaries.left.name)
        geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry(f".meshes/ns_pipe_flow_{resolution}_nprocs{nprocs}.msh")

pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}_nprocs{nprocs}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=pipemesh_mesh_refinement_callback,
    return_coords_to_bounds= pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=qdeg,
)

pipemesh.dm.view()

# Some useful coordinate stuff
x = pipemesh.N.x
y = pipemesh.N.y

# relative to the centre of the inclusion
r = sympy.sqrt((x - ndim(centre[0] * u.meter)) ** 2 + (y - ndim(centre[1] * u.meter)) ** 2)
th = sympy.atan2(y - ndim(centre[1] * u.meter), x - ndim(centre[0] * u.meter))

# need a unit_r_vec equivalent
inclusion_rvec = pipemesh.rvec - ndim(centre[0] * u.meter) * pipemesh.N.i - ndim(centre[1] * u.meter) * pipemesh.N.j
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# Boundary condition as specified in the diagram
#Vb = (4.0 * ndim(U0 * u.meter / u.second) * y * (0.41 - y)) / 0.41**2
# non-dimensionalised version
Vb = (4.0 * ndim(U0 * u.meter / u.second) * y * (ndim(height * u.meter) - y)) / ndim(height * u.meter)**2

# unit vector that is tangent to the inclusion surface (evaluated at said surface)
inclusion_unit_tvec = inclusion_unit_rvec.dot(pipemesh.N.j)*pipemesh.N.i - inclusion_unit_rvec.dot(pipemesh.N.i)*pipemesh.N.j

# %%
v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=Vdeg)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=Pdeg, continuous = Pcont)
rho = uw.discretisation.MeshVariable("rho", pipemesh, 1, degree=1, varsymbol=r"{\rho}")
grad_v_dot_tvec = uw.discretisation.MeshVariable("delUt", pipemesh, pipemesh.dim, degree=Vdeg)

# %%
#passive_swarm = uw.swarm.Swarm(mesh=pipemesh)

if infile is None:
    pass
#    passive_swarm.populate(
#        fill_param=1,
#    )
#
#    swarm_add_x = ndim(0.01 * u.meter)
#    swarm_add_y = ndim(0.195 * u.meter)
#
#    # add new points at the inflow
#    #FIXME: these are located at a single point - ask about purpose
#    npoints = 100
#    passive_swarm.dm.addNPoints(npoints)
#    with passive_swarm.access(passive_swarm.particle_coordinates):
#        for i in range(npoints):
#            passive_swarm.particle_coordinates.data[-1 : -(npoints + 1) : -1, :] = np.array(
#                [swarm_add_x, swarm_add_y] + ndim(0.01 * u.meter) * np.random.random((npoints, 2))
#            )
else:

    if uw.mpi.rank == 0:
        print(f"Reading: {infile}")

    v_soln.read_timestep(data_filename = infile, data_name = "U", index = maxsteps, outputPath = outdir)
    p_soln.read_timestep(data_filename = infile, data_name = "P", index = maxsteps, outputPath = outdir)
    #passive_swarm.read_timestep(base_filename = infile, swarm_id = "passive_swarm", index = maxsteps, outputPath = outdir)

# %%
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

navier_stokes = uw.systems.NavierStokesSLCN(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    rho=ndim(fluid_rho * u.kilogram / u.meter**3),
    verbose=True,
    solver_name="navier_stokes",
    order=order,
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Constant visc
navier_stokes.constitutive_model.Parameters.viscosity = ndim(dyn_visc*u.pascal*u.second)


navier_stokes.penalty = 0
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions

navier_stokes.add_dirichlet_bc(
    (0.0, 0.0),
    "inclusion",
)
navier_stokes.add_dirichlet_bc((0.0, 0.0), "top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
navier_stokes.add_dirichlet_bc((Vb, 0.0), "left")

navier_stokes.tolerance = tol

# if model == 2:  # Steady state !
#     # remove the d/dt term ... replace the time dependence with the
#     # steady state advective transport term
#     # to lean towards steady state solutions

#     navier_stokes.UF0 = -(
#         navier_stokes.rho * (v_soln.sym - v_soln_1.sym) / navier_stokes.delta_t
#     )

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
# tested: 0.1, 0.3
delta_t = ndim(delta_t * u.second)

# if infile is None:
#     navier_stokes.delta_t = delta_t
#     navier_stokes.solve(
#         timestep= delta_t, verbose=False,
#     )  # Stokes-like initial flow


# %%
ts = 0
# dt_ns = ndim(0.01 * u.second)
# delta_t_cfl = navier_stokes.estimate_dt()
# delta_t = min(delta_t_cfl[-1], dt_ns)

timeVal =  np.zeros(maxsteps + 1)*np.nan      # time values

# history terms for debugging
#DuDt_x_min  = np.zeros([maxsteps + 1, order])*np.nan
#DuDt_y_min  = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_00_min = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_01_min = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_10_min = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_11_min = np.zeros([maxsteps + 1, order])*np.nan
#
#DuDt_x_max  = np.zeros([maxsteps + 1, order])*np.nan
#DuDt_y_max  = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_00_max = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_01_max = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_10_max = np.zeros([maxsteps + 1, order])*np.nan
#DFDt_11_max = np.zeros([maxsteps + 1, order])*np.nan

elapsed_time = 0.0

if uw.mpi.rank == 0:
    print(delta_t)

# %%
for step in range(0, maxsteps):

    if uw.mpi.rank == 0:
        print(f"Timestep: {step}")

    delta_t_cfl = delta_t
    #delta_t_cfl = navier_stokes.estimate_dt()

    #if step % save_every == 0:
    #    delta_t = min(delta_t_cfl[-1], dt_ns)


    navier_stokes.solve(timestep = delta_t, zero_init_guess=True)

    # update passive swarm
    #passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    elapsed_time += delta_t
    timeVal[step] = elapsed_time

    if uw.mpi.rank == 0:
        print("Timestep {}, t {}, dt {}, dt_s {}".format(ts, elapsed_time, delta_t, delta_t_cfl))

    #if ts % save_every == 0 and ts > 0:
    #    pipemesh.write_timestep(
    #        outfile,
    #        meshUpdates=True,
    #        meshVars=[p_soln, v_soln],
    #        outputPath=outdir,
    #        index =ts,
    #    )

    #    with open(outdir + f"/{outfile}.pkl", "wb") as f:
    #        pickle.dump([timeVal], f)

    # update timestep
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

