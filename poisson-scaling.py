# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"

import sympy
import underworld3 as uw
from underworld3 import timing
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--scaling", type = str, default = "none")
parser.add_argument("--res", type = int, default = 8)
parser.add_argument("--tol", type = float, default = 1e-8)
parser.add_argument("--job", type = int, default = 0)
parser.add_argument("--idx", type = int, default = 0)
parser.add_argument("--maxits", type = int, default = 10)

args = parser.parse_args()

scaling = args.scaling
res     = args.res
tol     = args.tol
job     = args.job
it      = args.idx
max_it  = args.maxits

# %%
timing.reset()
timing.start()

# this works but not sure how to pass a float value
#scaling = uw.options.getString("scaling", default = "none")
#res     = uw.options.getInt("res", default=8)
#tol     = uw.options.getInt("tol", default = 1e-8)
#job     = uw.options.getInt("job", default = 0) # equivalent to nprocs if strong, multiplier if weak
#it    = uw.options.getInt("idx", default = 0)

if uw.mpi.rank == 0:
    print("Parameters passed:")
    print(f"scaling     : {scaling}")
    print(f"resolution  : {res}")
    print(f"tolerance   : {tol}")
    print(f"job         : {job}")
    print(f"iteration   : {it}")
    print(f"max iteration: {max_it}")

height  = 1.
width   = 1.
depth   = 1.

qdeg    = 3
udeg    = 2


is_nonlinear = True
do_vis       = False

# output dir
dir_name = str(os.getenv("NAME", "out"))
outdir = f"/scratch/el06/jg0883/{dir_name}/poisson_out/{scaling}_qdeg{qdeg}_udeg{udeg}_tol{tol}_res{res}_job{job}_iter{it}"
if uw.mpi.rank == 0:
    print(outdir)
    os.makedirs(outdir, exist_ok=True)

meshbox = uw.meshing.UnstructuredSimplexBox(minCoords = (0., 0., 0.),
                                            maxCoords = (height, width, depth),
                                            cellSize = 1/res,
                                            qdegree = qdeg,
                                            filename=f'{outdir}/mesh.msh')
#meshbox = uw.meshing.StructuredQuadBox(elementRes = (res, res, res),
#                                       minCoords = (0., 0., 0.),
#                                       maxCoords = (height, width, depth),
#                                       qdegree = qdeg,
#                                       filename = f'{outdir}/mesh.msh')

if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')
meshbox.dm.view()
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')

u_sol = uw.discretisation.MeshVariable("U", meshbox, 1, degree = udeg)
u_ana = uw.discretisation.MeshVariable("u", meshbox, 1, degree = udeg)

# %%
if uw.mpi.size == 1 and do_vis:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    #pvmesh.point_data["P"]      = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    pl = pv.Plotter(window_size=(1000, 750), notebook = True, off_screen = True)

    pl.add_mesh(
        pvmesh,
        #cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        #scalars="P",
        use_transparency=False,
        opacity=1,
        line_width = 0.0,
    )



    # if outdir is not None and fname is not None:
    pl.camera_position = "xy"
    #     pl.screenshot(
    #         filename=f"{outdir}/{fname}",
    #         window_size=(2560, 1280),
    #         return_img=False,
    #     )
    pl.show()



# %%
# calculate the source value with sympy
x, y,z = meshbox.N.x, meshbox.N.y, meshbox.N.z

# %%
from sympy.vector import CoordSys3D, gradient, divergence

# u_ana_expr = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y) * sympy.sin(sympy.pi * z)

R = CoordSys3D("R")
u_ana_expr = sympy.sin(sympy.pi * R.x) * sympy.sin(sympy.pi * R.y) * sympy.sin(sympy.pi * R.z)

grad_u_ana_expr = gradient(u_ana_expr)
if is_nonlinear:
    final_expr = - divergence( grad_u_ana_expr * (1 + u_ana_expr**2))
else:
    final_expr = - divergence(grad_u_ana_expr)

final_expr = final_expr.simplify()

# replace with variables representing the mesh coordinates
final_expr = final_expr.subs(R.x, x)
final_expr = final_expr.subs(R.y, y)
final_expr = final_expr.subs(R.z, z)

u_ana_expr = u_ana_expr.subs(R.x, x)
u_ana_expr = u_ana_expr.subs(R.y, y)
u_ana_expr = u_ana_expr.subs(R.z, z)

src_expr = final_expr

if uw.mpi.size == 1 and do_vis:
    display(u_ana_expr)
    display(grad_u_ana_expr)
    print("final source expression: ")
    display(src_expr)

# %%
poisson = uw.systems.Poisson(meshbox, u_Field = u_sol, verbose = True)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
if is_nonlinear:
    poisson.constitutive_model.Parameters.diffusivity = 1 + u_sol.sym[0]**2
else:
    poisson.constitutive_model.Parameters.diffusivity = 1

poisson.f = src_expr

poisson.add_dirichlet_bc(0., "Back")
poisson.add_dirichlet_bc(0., "Front")
poisson.add_dirichlet_bc(0., "Bottom")
poisson.add_dirichlet_bc(0., "Top")
poisson.add_dirichlet_bc(0., "Left")
poisson.add_dirichlet_bc(0., "Right")

poisson.tolerance = tol

if max_it > 0:
    poisson.petsc_options["ksp_max_it"] = max_it
    poisson.petsc_options["snes_max_it"] = max_it

# %%
poisson.solve()

# %%
with meshbox.access(u_ana):
    u_ana.data[:, 0] = uw.function.evaluate(u_ana_expr, u_ana.coords)

# %%
if uw.mpi.size == 1 and do_vis:
    import numpy as np
    import matplotlib.pyplot as plt

    # compare by plotting results with constant z
    z_use = 0.5
    x_use = np.linspace(0., 1., 50)
    y_use = np.linspace(0., 1., 50)

    xx, yy = np.meshgrid(x_use, y_use)

    xyz_eval = np.zeros([xx.flatten().shape[0], 3])
    xyz_eval[:, 0] = xx.flatten()
    xyz_eval[:, 1] = yy.flatten()
    xyz_eval[:, 2] = z_use

    u_ana_eval = uw.function.evaluate(u_ana_expr, xyz_eval)
    u_sol_eval = uw.function.evaluate(u_sol.sym, xyz_eval)

    fig, axs = plt.subplots(1, 2)
    out = axs[0].scatter(xx.flatten(), yy.flatten(), c = u_ana_eval, vmin = 0, vmax = 1, s = 15)
    axs[1].scatter(xx.flatten(), yy.flatten(), c = u_sol_eval, vmin = 0, vmax = 1, s = 15)

    for ax in axs:
        ax.set_aspect("equal")

# %%
import math

def calculate_u_rel_norm():

    # sympy functions corresponding to integrals
    u_diff = (u_sol.sym - u_ana.sym)**2
    u_ana_sq = u_ana.sym**2

    u_diff_integ = math.sqrt(uw.maths.Integral(meshbox, u_diff).evaluate())
    u_ana_sq_integ = math.sqrt(uw.maths.Integral(meshbox, u_ana_sq).evaluate())
    u_norm = u_diff_integ / u_ana_sq_integ

    return u_norm

# %%
rel_norm = 100 * calculate_u_rel_norm()

if uw.mpi.rank == 0:
    print(f"Relative norm (%): {rel_norm}")

# %%
timing.stop()

module_timing_data_orig = uw.timing.get_data(group_by="line_routine")
if uw.mpi.rank == 0:
    print(module_timing_data_orig)

filename = f"timing"

import json
if module_timing_data_orig:
    module_timing_data = {}
    for key, val in module_timing_data_orig.items():
        module_timing_data[key[0]] = val

    with open(f"{outdir}/{filename}.json", "w") as fp:
        json.dump(module_timing_data, fp, sort_keys = True, indent = 4)

uw.timing.print_table(group_by = "line_routine", output_file = f"{outdir}/{filename}.txt", display_fraction = 1.00)


