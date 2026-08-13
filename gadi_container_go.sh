#!/bin/bash -l

# Container-based launcher for UW3 scaling scripts.
# Mirrors gadi_baremetal_go.sh but runs each rank inside the Singularity container.
#
# Extra Python packages (e.g. assess) not bundled in the image are installed once:
#   pip install --target=/scratch/m18/jg0883/uw3-extra-pkgs assess
# and injected at runtime via PYTHONPATH below.

module load singularity
module load openmpi/4.1.7

export CONTAINER=/g/data/m18/software/containers/underworld3-gadi_v3.1.0.sif
export EXTRA_PKGS=/scratch/el06/jg0883/uw3-extra-pkgs

# Host-MPI injection for InfiniBand RDMA on Gadi (mirrors UW2 Gadi recipe).
#
# The UW3 container is built on Rocky Linux 8.10 with yum-installed system OpenMPI,
# giving libmpi.so.40 — same SONAME as Gadi's host openmpi/4.1.7.
# Injecting the host's libmpi via LD_LIBRARY_PATH makes Python/PETSc use the
# host's OpenMPI, which runs outside Singularity's cgroup and has full RDMA access.
#
# Key finding: on Gadi compute nodes, /lib64/liblustreapi.so.1 is a symlink that
# resolves to /half-root/usr/lib64/liblustreapi.so.1.0.0. The real system library
# tree lives under /half-root/. Binding /half-root makes these symlinks resolvable
# inside the container. This is identical to UW2's -B /half-root/ approach.
#
# Full dependency chain of /apps/openmpi/4.1.7/lib/libmpi.so.40 (from ldd):
#   /half-root/usr/lib64/ + /half-root/lib64/  — real system libs (liblustreapi,
#                                                libxpmem, libibverbs, librdmacm, etc.)
#   /opt/pbs/default/lib/libpbs.so.0           — PBS Pro runtime
#   /apps/openmpi-mofed5.8-pbs2021.1/4.1.7/lib/ — MOFED OpenMPI runtime
#   /apps/ucx/1.17.0/lib/                      — UCX transport
#   /apps/ucc/1.3.0/lib/                       — UCC collective library
#   /apps/hcoll/4.8.3228/lib/                  — HCollective
export HOST_MPI_LIB=/apps/openmpi/4.1.7/lib
export HOST_OMPI_RT=/apps/openmpi-mofed5.8-pbs2021.1/4.1.7/lib
export HOST_UCX=/apps/ucx/1.17.0/lib
export HOST_UCC=/apps/ucc/1.3.0/lib
export HOST_HCOLL=/apps/hcoll/4.8.3228/lib

env
cat ${UW_MODEL}.py

echo ""
echo "---------- Running Job ----------"
echo ${PWD}
echo ${UW_MODEL}
echo ${TYPE}
echo ${UW_RESOLUTION}
echo ${UW_SOL_TOLERANCE}
echo ${JOB_IDX}
echo ${RUN_IDX}
echo ""

# Rank placement. Default (empty) lets OpenMPI pack densely — 48 ranks to a node —
# which is what PBS memory limits are enforced against and what makes memory bandwidth
# per rank vary across a weak-scaling sweep (1, 8, 27, 48, 48 ranks/node for i^3 job
# indices). Set UW_MPI_MAP="--map-by node" to distribute round-robin instead, holding
# ranks/node roughly constant so placement stops confounding the scaling measurement.
# Requires PBSTASK_MULT>1 to provide the extra nodes.
mpiexec ${UW_MPI_MAP:-} -n ${NTASKS} singularity exec \
    --bind /half-root \
    --bind /opt/pbs/default/lib \
    ${CONTAINER} \
    bash -c \
    "LD_LIBRARY_PATH=${HOST_MPI_LIB}:${HOST_OMPI_RT}:${HOST_UCX}:${HOST_UCC}:${HOST_HCOLL}:/opt/pbs/default/lib:/half-root/usr/lib64:/half-root/lib64:\${LD_LIBRARY_PATH} \
     PYTHONPATH=${EXTRA_PKGS}:\${PYTHONPATH} \
     python3 ${UW_MODEL}.py \
     -uw_scaling ${TYPE} \
     -uw_res ${UW_RESOLUTION} \
     -uw_tol ${UW_SOL_TOLERANCE} \
     -uw_maxits ${UW_MAX_ITS} \
     -uw_nsteps ${UW_NSTEPS:-10} \
     -uw_inner_rtol ${UW_INNER_RTOL:-0} \
     -uw_init ${UW_INIT:-zero} \
     -uw_job ${JOB_IDX} \
     -uw_idx ${RUN_IDX} \
     -uw_integrator ${UW_VEP_INTEGRATOR} \
     -uw_memprobe ${UW_MEMPROBE} \
     -uw_re ${UW_NS_RE}"
