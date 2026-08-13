#!/bin/bash -l

# the follow load the full software stack and running environment on gadi
source /home/157/jg0883/install-scripts/gadi_install_shared.sh
env
#cat timed_model_${UW_MODEL}.py
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

export TIME_LAUNCH_MPI=`date +%s%N | cut -b1-13`
# See gadi_container_go.sh for why UW_MPI_MAP exists — default packs 48 ranks/node,
# "--map-by node" distributes round-robin so placement stops confounding scaling runs.
mpiexec ${UW_MPI_MAP:-} -x LD_PRELOAD=libmpi.so -n ${NTASKS} bash -c \
    "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py \
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

#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 timed_model_2D.py"

# profiling runs - for petsc tests

#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 timed_model_${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.txt:ascii_flamegraph"

#mpiexec -n ${NTASKS} -x LD_PRELOAD=libmpi.so bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.txt"

#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.xml:ascii_xml"

# create flame graph
#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.txt:ascii_flamegraph"
