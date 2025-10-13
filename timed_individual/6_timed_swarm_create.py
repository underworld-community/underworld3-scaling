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
#     display_name: uw3-dev
#     language: python
#     name: python3
# ---

# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"

#import nest_asyncio
#nest_asyncio.apply()

import numpy as np
from pathlib import Path

import underworld3 as uw
from underworld3.systems import Stokes # should we also include other systems?
from underworld3 import function
from underworld3 import timing

# %% [markdown]
# ### Testbench for Mesh petsc_save_checkpoint

# %%
# order         = int(os.getenv("UW_ORDER","2"))
n_els           = int(os.getenv("UW_RESOLUTION",16))
dim             = int(os.getenv("UW_DIM",2)) # FIXME: add option for running 3D model
scaling_type    = int(os.getenv("SCALING_TYPE",1))

# %%
# solver options - verify options to use in underworld3
tol  = float(os.getenv("UW_SOL_TOLERANCE",1.e-5))
#itol  = float(os.getenv("UW_SOL_TOLERANCE",1.e-10))
# otol  = float(os.getenv("UW_SOL_TOLERANCE",1.e-10))*10
# penalty  = float(os.getenv("UW_PENALTY",-1.))
max_its  = int(os.getenv("UW_MAX_ITS",-1))

# %%
show_plot = False
debug = True
do_timing = True
refinement = None

# %%
if do_timing:
    timing.reset()
    timing.start()

mesh = uw.meshing.UnstructuredSimplexBox(regular=False,
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / n_els,
    qdegree=3, refinement = refinement
)

#mesh.dm.view()

# swarm variables
swarm = uw.swarm.Swarm(mesh=mesh)

# %% [markdown]
# ### I/O test

# %%
### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
#outputPath = f"../output/timing-tests/"
if scaling_type == 1:
    #outputPath = "/scratch/el06/jg0883/uw3-scaling/output/timing-mesh-petsc_save_checkpoint-2D-weak/"
    outputPath = "../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-swarm-create-2D-weak/"
elif scaling_type == 2:
    outputPath = "../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-swarm-create-2D-strong/"

### write test
if uw.mpi.rank==0:
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

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
    module_timing_data_orig = uw.timing.get_data(group_by="routine")

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


