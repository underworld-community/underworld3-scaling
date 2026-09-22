"""Is read_checkpoint returning zeros in parallel, or genuinely wrong values?"""
import os
import numpy as np
from mpi4py import MPI
import underworld3 as uw

out = os.environ["PROBE_OUT"]; os.makedirs(out, exist_ok=True)
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.,0.,0.), maxCoords=(1.,1.,1.), cellSize=1/8, qdegree=2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1, continuous=True)
z = mesh.N.z
T.data[:, 0] = uw.function.evaluate(1.0 - z, T.coords).reshape(-1)

mesh.write_timestep("probe", meshUpdates=True, meshVars=[T],
                    outputPath=out, index=0, create_xdmf=False, petsc_reload=True)

Trd = uw.discretisation.MeshVariable("T_rd", mesh, 1, degree=1, continuous=True)
Trd.read_checkpoint(os.path.join(out, "probe.mesh.T.00000.h5"), data_name="T")

a, b = np.asarray(T.data[:,0]), np.asarray(Trd.data[:,0])
loc = np.array([a.min(), a.max(), b.min(), b.max(),
                float(np.abs(b).sum()), float(np.count_nonzero(b)), float(b.size)])
g = MPI.COMM_WORLD.gather(loc, root=0)
if uw.mpi.rank == 0:
    print(f"ranks={uw.mpi.size}")
    for r, v in enumerate(g):
        print(f"  rank{r}: original[{v[0]:.3f},{v[1]:.3f}]  reloaded[{v[2]:.3f},{v[3]:.3f}]  "
              f"sum|reloaded|={v[4]:.3f}  nonzero={int(v[5])}/{int(v[6])}")
