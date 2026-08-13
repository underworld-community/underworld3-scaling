# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"

#import nest_asyncio
#nest_asyncio.apply()

import numpy as np
from pathlib import Path

# %%
# Set some things
import sympy
from sympy import Piecewise

import underworld3 as uw
from underworld3.systems import Stokes # should we also include other systems?
from underworld3 import function
from underworld3 import timing

# %%
# order         = int(os.getenv("UW_ORDER","2"))
n_els           = int(os.getenv("UW_RESOLUTION",16))
dim             = int(os.getenv("UW_DIM",2)) # FIXME: add option for running 3D model
scaling_type    = int(os.getenv("SCALING_TYPE",1))
uw_name         = str(os.getenv("UW_NAME", ""))
uw_model        = str(os.getenv("UW_MODEL", ""))

# %%
# solver options - verify options to use in underworld3
tol  = float(os.getenv("UW_SOL_TOLERANCE",1.e-5))
#itol  = float(os.getenv("UW_SOL_TOLERANCE",1.e-10))
# otol  = float(os.getenv("UW_SOL_TOLERANCE",1.e-10))*10
# penalty  = float(os.getenv("UW_PENALTY",-1.))
max_its  = int(os.getenv("UW_MAX_ITS",-1))

# %%
# soln_name = str(os.getenv("UW_MODEL","SolDB3d"))

# do_IO  = bool(int(os.getenv("UW_ENABLE_IO","0")))

# jobid = str(os.getenv("PBS_JOBID",os.getenv("SLURM_JOB_ID","0000000")))

# picklename = str(os.getenv("PICKLENAME","None"))

# %%
show_plot = False
debug = True
do_timing = True
refinement = None

# BC setting
mode_use = 1 # 0 for the old version, 1 for the one with sympy.oo

# solC parameters
# FIXME: allow testing of other geodynamic benchmark problems

eta_0 = 1.0
f_0 = 1.0

x_c = 0.5

# %% [markdown]
# ### Implement SolC first, but handle other benchmarks next

# helper function for cleaning up


# %%
if do_timing:
    timing.reset()
    timing.start()

mesh = uw.meshing.UnstructuredSimplexBox(regular=False,
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / n_els,
    qdegree=3, refinement = refinement
)

#mesh.dm.view()

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2, varsymbol = r"V")
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, varsymbol = r"P")
t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree = 3, continuous = True, varsymbol = r"{T}")

v_soln_rd = uw.discretisation.MeshVariable("U2", mesh, mesh.dim, degree=2, varsymbol = r"V2") # declared on the original variable

swarm = uw.swarm.Swarm(mesh=mesh)
Mat = uw.swarm.SwarmVariable("M", swarm, 1, proxy_degree = 3)

Mat_rd = uw.swarm.SwarmVariable("M2", swarm, 1, proxy_degree = 3) # declared on the original variable

swarm.populate(fill_param= 5)

from copy import deepcopy

with swarm.access():
    before = deepcopy(swarm.data[...])

# v = stokes.Unknowns.u
# p = stokes.Unknowns.p

# should this be ViscoelasticPlasticFlowModel?
stokes = uw.systems.Stokes( mesh,
                            velocityField = v_soln,
                            pressureField = p_soln,
                            solver_name="stokes",)

viscosity = 1
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

stokes.tolerance = tol
stokes.petsc_options["snes_max_it"] = max_its
stokes.petsc_options["ksp_max_it"] = max_its

# %%
# free slip.
# note with petsc we always need to provide a vector of correct cardinality.
if mode_use == 0:
    stokes.add_dirichlet_bc((0.0,), "Left", (0,))
    stokes.add_dirichlet_bc((0.0,), "Right", (0,))
    stokes.add_dirichlet_bc((0.0,), "Top", (1,))
    stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))
elif mode_use == 1:
    stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")

# %%
# stokes.petsc_options["snes_monitor"]= None
# stokes.petsc_options["ksp_monitor"] = None

# # %%f
# # FIXME: need to understand the solver options better

# stokes.petsc_options["snes_type"] = "newtonls"
# stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
# stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

# stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
# stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
# stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
# stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
# stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %%

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1 and show_plot:

    import pyvista as pv
    import underworld3 as uw
    import underworld3.visualisation

    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, stokes.bodyforce[1])
    pvmesh.point_data["Vmag"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))

    velocity_points = uw.visualisation.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=3.0, opacity=1, show_scalar_bar=False)

    pl.show(cpos="xy")

# %% [markdown]
# ### Advection diffusion

# # %%
# Advection-Diffusion
k = 1.0
h = 0.0

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5

adv_diff.add_dirichlet_bc(0.0, "Bottom")
adv_diff.add_dirichlet_bc(1.0, "Top")

adv_diff.tolerance = tol
adv_diff.petsc_options["snes_max_it"] = max_its
adv_diff.petsc_options["ksp_max_it"] = max_its

# %%
x,y = mesh.X

stokes.bodyforce = sympy.Matrix(
    [
        0,
        Piecewise(
            (f_0, x > x_c),
            (0.0, True),
        ),
    ]
)
#print(stokes.bodyforce)

#print(stokes.petsc_options)


# initial solve of stokes system
if uw.mpi.rank == 0 and debug:
    print("1. Attempting stokes solve ... ")
stokes.solve(zero_init_guess = True)
if uw.mpi.rank == 0 and debug:
    print("1. Solved stokes ... ")
    print(f"mode:{mode_use}")

if uw.mpi.rank == 0:
    with mesh.access():
        print(v_soln.data.min())
        print(v_soln.data.max())

# set to a fixed velocity field if using max_its
if max_its >= 0:
    # define velocity sympy function
    v_fn = sympy.Matrix(
    [
        Piecewise(
            (-1, ((x >= 0.5) & (y >= 0.5))),
            (0, ((x <= 0.5) & (y >= 0.5))),
            (1, ((x <= 0.5) & (y <= 0.5))),
            (0, ((x >= 0.5) & (y <= 0.5))),
            (0.0, True),
        ),
        Piecewise(
            (0, ((x >= 0.5) & (y >= 0.5))),
            (-1, ((x <= 0.5) & (y >= 0.5))),
            (0, ((x <= 0.5) & (y <= 0.5))),
            (1, ((x >= 0.5) & (y <= 0.5))),
            (0.0, True),
        ),
    ]
)
    with mesh.access(v_soln):
        v_soln.data[...] = uw.function.evaluate(v_fn, v_soln.coords).reshape(-1, 2)

# initialize temperature field
with mesh.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(y, t_soln.coords).reshape(-1, 1)

# # %%
# # Advection-Diffusion
if uw.mpi.rank == 0 and debug:
    print("2. Attempting adv_diff solve ... ")
adv_diff.solve(timestep=0.01 * stokes.estimate_dt(), zero_init_guess=True)
if uw.mpi.rank == 0 and debug:
    print("2. Solved adv_diff ... ")

if uw.mpi.rank == 0:
    with mesh.access():
        print(v_soln.data.min())
        print(v_soln.data.max())
        print(t_soln.data.min())
        print(t_soln.data.max())

# %%
if uw.mpi.rank == 0 and debug:
    print("3. Attempting swarm advection ... ")
swarm.advection(v_soln.fn, 1 * stokes.estimate_dt())
if uw.mpi.rank == 0 and debug:
    print("3. Solved swarm advection ... ")

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1 and show_plot:

    import pyvista as pv
    #import underworld3 as uw
    import uw.visualisation

    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = uw.visualisation.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["Vmag"] = uw.visualisation.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
    )

    pl.show(cpos="xy")

# %% [markdown]
### Swarm tests

# %%
# %%
if uw.mpi.size == 1 and show_plot:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    # point sources at cell centres
    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    spoints = vis.swarm_to_pv_cloud(swarm)
    spoint_cloud = pv.PolyData(spoints)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_points(spoint_cloud,color="Black",
                  render_points_as_spheres=False,
                  point_size=5, opacity=0.66
                )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    pl.show()


# %% [markdown]
# ### I/O test

# %%
### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
#outputPath = f"../output/timing-tests/"
if scaling_type == 1:
    #outputPath = "../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-SolC-2D-weak/"
    outputPath = f"../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-2D-weak-{uw_name}-{uw_model}/"
elif scaling_type == 2:
    #outputPath = "../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-SolC-2D-strong/"
    outputPath = f"../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-2D-strong-{uw_name}-{uw_model}-{n_els}/"

### write test
if uw.mpi.rank==0:
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

#### save the mesh and associated variables
if uw.mpi.rank == 0 and debug:
    print("4. Attempting save mesh using petsc_save_checkpoint ... ")
mesh.petsc_save_checkpoint(index = 0,
                            meshVars=[v_soln],  # only save one variable for tests
                            #meshVars=[v_soln, p_soln, t_soln],
                            outputPath = outputPath + f"timing_els_{n_els}")
if uw.mpi.rank == 0 and debug:
    print("4. Saved mesh using petsc_save_checkpoint ... ")

#### save the swarm
if uw.mpi.rank == 0 and debug:
    print("5. Attempting save swarm using petsc_save_checkpoint ... ")
swarm.petsc_save_checkpoint('swarm', 0, outputPath + f"timing_els_{n_els}")
if uw.mpi.rank == 0 and debug:
    print("5. Saved swarm using petsc_save_checkpoint ... ")

# mesh_rd = uw.discretisation.Mesh(outputPath + f"timing_els_{n_els}_ref_{refinement}_step_{0:05}.h5")

# %%
### read tests

''' Comment for now since this does not complete when using higher resolution in Gadi '''
# mesh
if uw.mpi.rank == 0 and debug:
   print("6. Attempting save mesh using write_timestep ... ")
mesh.write_timestep(f"timing_els_{n_els}", meshUpdates=False, outputPath=outputPath, index = 0)
if uw.mpi.rank == 0 and debug:
    print("6. Saved mesh using write_timestep ... ")

if uw.mpi.rank == 0 and debug:
   print("7. Attempting read mesh ... ")

mesh_rd = uw.discretisation.Mesh(f"{outputPath}/timing_els_{n_els}.mesh.00000.h5")

if uw.mpi.rank == 0 and debug:
    print("7. Read mesh ... ")

assert np.fabs(mesh.get_min_radius() - mesh_rd.get_min_radius()) < 1.0e-5

# variable
if uw.mpi.rank == 0 and debug:
   print("8. Attempting save mesh variable using write_timestep ... ")
mesh.write_timestep(f"timing_els_{n_els}", meshUpdates = False, meshVars=[v_soln], outputPath=outputPath, index=0)
if uw.mpi.rank == 0 and debug:
    print("8. Saved mesh variable using write_timestep ... ")

if uw.mpi.rank == 0 and debug:
   print("9. Attempting read mesh variable ... ")

v_soln_rd.read_timestep(f"timing_els_{n_els}", "U", 0, outputPath = outputPath)

if uw.mpi.rank == 0 and debug:
    print("9. Read mesh variable ... ")

with mesh.access():
    assert np.allclose(v_soln.data, v_soln_rd.data)

''' Comment for now since this does not complete when using Gadi '''
# swarm
if uw.mpi.rank == 0 and debug:
    print("10. Attempting save swarm variable using write_timestep ... ")

swarm.write_timestep(f"timing_els_{n_els}", "swarm", swarmVars = [], outputPath = outputPath, index=0, force_sequential = True)

if uw.mpi.rank == 0 and debug:
    print("10. Saved swarm variable using write_timestep ... ")

#swarm_rd = uw.swarm.Swarm(mesh = mesh_rd)

#if uw.mpi.rank == 0 and debug:
#    print("11. Attempting read swarm variable ... ")

#swarm_rd.read_timestep(f"timing_els_{n_els}", "swarm", 0, outputPath=outputPath)

#if uw.mpi.rank == 0 and debug:
#    print("11. Read swarm variable ... ")

# swarm variable
#swarm.write_timestep(f"timing_els_{n_els}", "swarm", swarmVars = [Mat], outputPath = outputPath, index=0)

#with swarm.access(Mat_rd):
#    Mat_rd.read_timestep(f"timing_els_{n_els}", "swarm", "M", 0, outputPath = outputPath)

#with swarm.access():
#    assert np.allclose(Mat.data, Mat_rd.data)

# %%
if do_timing:
    timing.stop()

uw.mpi.barrier()
if uw.mpi.rank == 0:
    for fname_end in ("*.h5", "*.xdmf", "*.xmf", "*.pbin"):
        for filename in Path(outputPath).glob(fname_end):
            os.remove(filename)

# %%

if do_timing:
    #module_timing_data_orig = uw.timing.get_data(group_by="line")
    module_timing_data_orig = uw.timing.get_data(group_by="routine")
    if uw.mpi.rank == 0:
        print(module_timing_data_orig)

    # write out data
    #filename = "Res_{}_Nproc_{}_JobID_{}".format(res,uw.mpi.size,jobid)
    filename = "Res_{}_Nproc_{}".format(n_els,uw.mpi.size)
    import json
    if module_timing_data_orig:
        module_timing_data = {}
        for key,val in module_timing_data_orig.items():
            module_timing_data[key[0]] = val
        #module_timing_data["Other_timing"] = other_timing
        #module_timing_data["Other_data"]   = { "res":res, "nproc":uw.mpi.size, "vrms":vrms, "prms":prms }
        with open(f"{outputPath}/{filename}.json", 'w') as fp:
            json.dump(module_timing_data, fp,sort_keys=True, indent=4)

    uw.timing.print_table(group_by="routine", output_file=f"{outputPath}/{filename}.txt", display_fraction = 1.00)

    if uw.mpi.rank == 0 and debug:
        print(module_timing_data)


