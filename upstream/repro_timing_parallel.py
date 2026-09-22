"""Prototype of the two proposed additions, tested against unequal ranks.

1. uw.timing.stage(name, barrier=True) -- context manager emitting <name> and <name>_sync
2. uw.timing.get_summary_parallel()    -- max/mean/min/n_working instead of rank-local
"""
import time, contextlib
import underworld3 as uw
from petsc4py import PETSc
from mpi4py import MPI


@contextlib.contextmanager
def stage(name, barrier=False):
    """Push a log stage; optionally follow it with a TIMED barrier in <name>_sync.

    stage.pop() is a LOCAL call, so a stage measures its own duration, never the
    wall-clock span. barrier=True separates work from arrival-spread by construction.
    """
    st = PETSc.Log.Stage(name); st.push()
    try:
        yield st
    finally:
        st.pop()
        if barrier:
            sync = PETSc.Log.Stage(f"{name}_sync"); sync.push()
            MPI.COMM_WORLD.Barrier()
            sync.pop()


def get_summary_parallel(comm=MPI.COMM_WORLD, min_time=0.0):
    """Reduce across ranks. max sets the pace; n_working separates work from waiting."""
    local = {e[0]: e[2] for e in uw.timing.get_summary(min_time=0.0)["events"]}
    names = sorted(set().union(*comm.allgather(set(local))))
    vec = [local.get(n, 0.0) for n in names]
    gathered = comm.gather(vec, root=0)
    if comm.rank != 0:
        return None
    out = []
    for i, n in enumerate(names):
        col = [g[i] for g in gathered]
        mx = max(col)
        if mx < min_time:
            continue
        out.append({"event": n, "max": mx, "mean": sum(col)/len(col), "min": min(col),
                    "n_working": sum(1 for v in col if v > 0.5*mx), "nranks": len(col)})
    return sorted(out, key=lambda r: -r["max"])


uw.timing.start()
with stage("unequal_phase", barrier=True):
    time.sleep(0.1 if uw.mpi.rank == 0 else 2.0)

with stage("balanced_phase", barrier=True):
    time.sleep(1.0)

ev = uw.timing.create_event("Mesh.rank0_only")
ev.begin()
if uw.mpi.rank == 0:
    time.sleep(1.5)
ev.end()

res = get_summary_parallel(min_time=0.01)
if uw.mpi.rank == 0:
    print(f"\n{'event':<28}{'max':>8}{'mean':>8}{'min':>8}{'nwork':>7}   verdict")
    for r in res:
        v = "SERIAL" if r["n_working"] == 1 else ("imbalanced" if r["max"]/max(r["min"],1e-9) > 2 else "balanced")
        print(f"{r['event']:<28}{r['max']:>8.2f}{r['mean']:>8.2f}{r['min']:>8.2f}"
              f"{r['n_working']:>7}   {v}")
