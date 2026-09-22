"""Is eval_nodal's cost real work, or a collective absorbing earlier skew?

Barrier immediately before each timed region, so no stage can inherit imbalance
from what preceded it. If the nodal/offnode gap survives barriers it is real work;
if it collapses, the earlier measurement was charging skew to the first collective.
Per-rank times are printed, since a max-across-ranks number hides imbalance.
"""
import os, time
import numpy as np
from mpi4py import MPI
import underworld3 as uw

res = int(os.environ.get("PROBE_RES", "24"))
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.,0.,0.), maxCoords=(1.,1.,1.), cellSize=1.0/res, qdegree=2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1, continuous=True)
x, y, z = mesh.N.x, mesh.N.y, mesh.N.z
T.data[:, 0] = uw.function.evaluate(1.0 - z, T.coords).reshape(-1)

nodal = np.asarray(T.coords)
h = 0.5 / res
offset = nodal + np.array([h, h, h])
offset = np.clip(offset, 1e-6, 1.0 - 1e-6)

uw.function.evaluate(T.sym[0], offset)          # untimed warm-up (JIT)

def timed(label, coords):
    MPI.COMM_WORLD.Barrier()                     # everyone starts together
    t0 = time.perf_counter()
    uw.function.evaluate(T.sym[0], coords)
    t = time.perf_counter() - t0
    allt = MPI.COMM_WORLD.gather(t, root=0)
    if uw.mpi.rank == 0:
        a = np.array(allt)
        print(f"{label:<10} max={a.max():.4f} min={a.min():.4f} mean={a.mean():.4f} "
              f"imbalance={a.max()/a.min():.1f}x")

if uw.mpi.rank == 0:
    print(f"ranks={uw.mpi.size} res={res} pts/rank~{nodal.shape[0]}")
timed("nodal", nodal)
timed("offnode", offset)
timed("nodal2", nodal)      # repeat: is the first one special?
