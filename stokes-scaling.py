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
#     display_name: uw3-mamba-run
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# ##### Case 4: No slip boundaries and smooth density distribution
# <!--
#     1. Works fine (i.e., bc produce results)
# -->
# Adapted for scaling tests for the JOSS paper.
#
# Full implementation [here](https://github.com/underworldcode/underworld3-documentation/blob/main/Notebooks/Examples-Spherical-Stokes/Ex_Stokes_Spherical_Benchmark_Kramer.py).
#

# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

from mpi4py import MPI
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy

import assess
import h5py
import sys
import argparse

# %%
if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()

    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

# %%
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

if uw.mpi.rank == 0:
    print("Parameters passed:")
    print(f"scaling     : {scaling}")
    print(f"resolution  : {res}")
    print(f"tolerance   : {tol}")
    print(f"job         : {job}")
    print(f"iteration   : {it}")
    print(f"max iteration: {max_it}")

# %%
# mesh options
r_o = 2.22
r_int = 2.0
r_i = 1.22

refine = None

cellsize = 1/res

# %%
# compute analytical solution
analytical = True
visualize = False
timing = True

# %%
if timing:
    uw.timing.reset()
    uw.timing.start()

# %%
# specify the case
case = uw.options.getString('case', default = 'case4')

# spherical harmonic fn degree (l) and order (m)
l = uw.options.getInt("l", default=2)
m = uw.options.getInt("m", default=1)
k = l+1 # power

# %%
# fem stuff

# stokes_tol = uw.options.getReal("stokes_tol", default = 1e-5)
stokes_tol = tol

vdegree  = uw.options.getInt("vdegree", default = 2)
pdegree = uw.options.getInt("pdegree", default = 1)
pcont = uw.options.getBool("pcont", default = True)
pcont_str = str(pcont).lower()

vel_penalty = uw.options.getReal("vel_penalty", default = 1e8)
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))

# %%
# choosing boundary condition and density perturbation type
freeslip, noslip, delta_fn, smooth = False, False, False, False

# only case 4
noslip, smooth = True, True

# %%
# output dir
dir_name = str(os.getenv("NAME", "out"))
output_dir = f"/scratch/el06/jg0883/{dir_name}/stokes_out/{scaling}_vdeg{vdegree}_pdeg{pdegree}_tol{tol}_res{res}_job{job}_iter{it}"

if uw.mpi.rank == 0:
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ### Analytical Solution

# %%
if analytical:
    '''
    For smooth density distribution only single solution exists in the domain.
    But for sake of code optimization I am creating two solution here.
    '''
    soln_above = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    soln_below = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)

# %% [markdown]
# ### Create Mesh

# %%
mesh = uw.meshing.SphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize,
                                    qdegree=max(pdegree, vdegree),
                                    filename=f'{output_dir}/mesh.msh', refinement=refine)
#mesh = uw.meshing.CubedSphere(radiusOuter=r_o,
#                              radiusInner=r_i,
#                              numElements = res,
#                              simplex = False,
#                              qdegree = max(pdegree, vdegree),
#                              filename = f"{output_dir}/mesh.msh")

# %%
if uw.mpi.size == 1 and visualize:
    vis.plot_mesh(mesh, save_png=True, dir_fname=output_dir+'mesh.png', title='', clip_angle=135, cpos='yz')

# %%
# print mesh size in each cpu
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')
mesh.dm.view()
if uw.mpi.rank == 0:
    print('-------------------------------------------------------------------------------')

# %%
# mesh variables
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.data.shape[1], degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.data.shape[1], degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable('RHO_a', mesh, 1, degree=pdegree, continuous=True)

    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.data.shape[1], degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)

# %%
norm_v = uw.discretisation.MeshVariable("N", mesh, mesh.data.shape[1], degree=pdegree, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)
    norm_v.data[:,2] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[2], norm_v.coords)

# %%
# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw =sympy.Piecewise((2*sympy.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2]<0),
                        (mesh.CoordinateSystem.xR[2], True)
                       )

# Null space in velocity expressed in x,y,z coordinates
v_theta_phi_fn_xyz = sympy.Matrix(((0,1,1), (-1,0,1), (-1,-1,0))) * mesh.CoordinateSystem.N.T

# %%
if analytical:
    with mesh.access(v_ana, p_ana,):

        def get_ana_soln(_var, _r_int, _soln_above, _soln_below):
            # get analytical solution into mesh variables
            r = uw.function.evalf(r_uw, _var.coords)
            for i, coord in enumerate(_var.coords):
                if r[i]>_r_int:
                    _var.data[i] = _soln_above(coord)
                else:
                    _var.data[i] = _soln_below(coord)

        # velocities
        get_ana_soln(v_ana, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # pressure
        get_ana_soln(p_ana, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

# %%
# plotting analytical velocities
clim, vmag, vfreq = [0., 0.001], 5e2, 75

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_ana, vector_name='v_ana', cmap=cmc.lapaz.resampled(21), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_ana.png', clip_angle=135, show_arrows=False, cpos='yz')

    vis.save_colorbar(colormap=cmc.lapaz.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf',
                      output_path=output_dir, fname='v_ana')

# %%
clim = [-0.1, 0.1]
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_ana.sym, 'p_ana', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, clip_angle=135,
                    dir_fname=output_dir+'p_ana.png', cpos='yz')

    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
                      output_path=output_dir, fname='p_ana')

# %%
# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# %%
# defining rho fn and bodyforce term
y_lm_real = sympy.sqrt((2*l + 1)/(4*sympy.pi) * sympy.factorial(l - m)/sympy.factorial(l + m)) * sympy.cos(m*phi_uw) * sympy.assoc_legendre(l, m, sympy.cos(th_uw))

gravity_fn = -1.0 * unit_rvec # gravity

if smooth:
    rho = ((r_uw/r_o)**k) * y_lm_real
    stokes.bodyforce = rho*gravity_fn

# %%
if analytical:
    with mesh.access(rho_ana):
        rho_ana.data[:] = np.c_[uw.function.evaluate(rho, rho_ana.coords)]

# %%
# boundary conditions
v_diff =  v_uw.sym - v_ana.sym
stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Upper.name)
stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Lower.name)

# %%
# plotting analytical rho
clim = [-0.4, 0.4]

if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(mesh, -rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31), clim=clim, save_png=True,
                    dir_fname=output_dir+'rho_ana.png', clip_angle=135, cpos='yz')

    vis.save_colorbar(colormap=cmc.roma.resampled(31), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
                      cb_orient='horizontal', cb_axis_label='Rho', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
                      output_path=output_dir, fname='rho_ana')

# %%
# Stokes settings
stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# mg, multiplicative - very robust ... similar to gamg, additive
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# set max iterations for scaling tests
if max_it > 0:
    stokes.petsc_options["ksp_max_it"] = max_it
    stokes.petsc_options["snes_max_it"] = max_it

# %%
stokes.solve(verbose=True, debug=False)

# %%

if timing:
    uw.timing.stop()

    module_timing_data_orig = uw.timing.get_data(group_by="routine")
    if uw.mpi.rank == 0:
        print(module_timing_data_orig)

    filename = f"timing"

    import json

    if module_timing_data_orig:
        module_timing_data = {}
        for key, val in module_timing_data_orig.items():
            module_timing_data[key[0]] = val

        with open(f"{output_dir}/{filename}.json", "w") as fp:
            json.dump(module_timing_data, fp, sort_keys = True, indent = 4)

    uw.timing.print_table(group_by = "routine", output_file = f"{output_dir}/{filename}.txt", display_fraction = 1.00)


# %%
# Null space evaluation
I0 = uw.maths.Integral(mesh, v_theta_phi_fn_xyz.dot(v_uw.sym))
norm = I0.evaluate()

I0.fn = v_theta_phi_fn_xyz.dot(v_theta_phi_fn_xyz)
vnorm = I0.evaluate()
# print(norm/vnorm, vnorm)

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_phi_fn_xyz, v_uw.coords) / vnorm
    v_uw.data[...] -= dv

## %%
## compute error
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):

        def get_error(_var_err, _var_uw, _r_int, _soln_above, _soln_below):
            # get error in numerical solution
            r = uw.function.evalf(r_uw, _var_err.coords)
            for i, coord in enumerate(_var_err.coords):
                if r[i]>_r_int:
                    _var_err.data[i] = _var_uw.data[i] - _soln_above(coord)
                else:
                    _var_err.data[i] = _var_uw.data[i] - _soln_below(coord)

        # error in velocities
        get_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # error in pressure
        get_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
#
## %%
## computing L2 norm
if analytical:
    with mesh.access(v_err, p_err, p_ana, v_ana):
        v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
        v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
        v_err_l2 = np.sqrt(v_err_I.evaluate())/np.sqrt(v_ana_I.evaluate())

        p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
        p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
        p_err_l2 = np.sqrt(p_err_I.evaluate())/np.sqrt(p_ana_I.evaluate())

        if uw.mpi.rank == 0:
            print('Relative error in velocity in the L2 norm: ', v_err_l2)
            print('Relative error in pressure in the L2 norm: ', p_err_l2)
