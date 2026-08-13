#!/bin/bash
source params.sh

if (($SCALING_TYPE==1)); then
   export TYPE="Weak"
elif (($SCALING_TYPE==2)); then
   export TYPE="Strong"
else
   echo "Scaling type must be 1 (weak) or 2 (strong)."
   exit 1
fi

## find the BATCH environment ##
#################################
if qstat --version &> /dev/null ; then
   BATCH_SYS="PBS"
   SITE="Gadi"
elif squeue --version &> /dev/null ; then
   BATCH_SYS="SLURM"
   SITE="Setonix"
else
   echo "Can't determine batch system"
   exit 1
fi
echo "Batch system is $BATCH_SYS"
#################################

# Iterate over every model in UW_MODELS (falls back to UW_MODEL if unset)
MODELS_TO_RUN="${UW_MODELS:-$UW_MODEL}"

for model in ${MODELS_TO_RUN}
do
    export UW_MODEL="${model}"
    export NAME="${TYPE}_${UW_NAME}_DIM${UW_DIM}_BASE${SCALING_BASE}_TOL${UW_SOL_TOLERANCE}_MAXITS${UW_MAX_ITS}_MODEL_${UW_MODEL}_${SITE}"

    mkdir -p ${NAME}
    cp *.sh ${NAME}
    cp *.py ${NAME}
    # Copy analysis scripts if present
    [ -d analysis ] && cp -r analysis ${NAME}/

    pushd ${NAME} > /dev/null

    for i in ${JOBS}
    do
        export JOB_IDX="${i}"

        for j in ${RUN_INDICES}
        do
            export RUN_IDX="${j}"

            if (($SCALING_TYPE==1)); then   # weak
                export UW_RESOLUTION="$((${SCALING_BASE} * ${i}))"
                if (($UW_DIM==2)); then
                    export NTASKS="$((${i}*${i}))"
                else
                    export NTASKS="$((${i}*${i}*${i}))"
                fi
            else                            # strong
                export UW_RESOLUTION=${SCALING_BASE}
                export NTASKS=${i}
            fi

            # Whitelist of variables forwarded into the PBS/Slurm job. A variable set in
            # params.sh but MISSING here is silently unset inside the job, so the
            # launcher's ${VAR:-default} falls back and the run quietly uses the default
            # — it does not fail. Add every new params.sh variable here.
            export EXPORTVARS="TYPE,UW_RESOLUTION,NTASKS,UW_DIM,UW_SOL_TOLERANCE,UW_MAX_ITS,UW_MODEL,SCALING_TYPE,UW_NAME,JOB_IDX,RUN_IDX,NAME,OUTPUT_BASE,MESH_CACHE,UW_VEP_INTEGRATOR,UW_MEMPROBE,UW_NS_RE,UW_NSTEPS,UW_INNER_RTOL,UW_MPI_MAP,UW_INIT"

            if [ $BATCH_SYS == "PBS" ] ; then
                PBSTASKS=`python3 <<<"print((int(${NTASKS}/48) + (${NTASKS} % 48 > 0))*48*${PBSTASK_MULT})"`
                NNODES=$((PBSTASKS / 48))
                # Per-node memory depends on the queue. PBS enforces this PER NODE, and
                # mpiexec packs ranks densely (48 to a node), so what matters is
                # ranks_per_node x GB_per_rank, NOT the job total. BASE=10 Stokes needs
                # ~6.7 GB/rank, so a full node wants ~320 GB and cannot fit in normal's
                # 192 GB — which is why those runs were SIGKILLed at well under their
                # requested total. PBSTASK_MULT does not help: it adds nodes but leaves
                # the packing unchanged.
                MEM_PER_NODE=192
                [ "${QUEUE}" = "hugemem" ] && MEM_PER_NODE=1470
                MEMORY="$((NNODES * MEM_PER_NODE))GB"
                GADI_LAUNCHER="gadi_baremetal_go.sh"
                [ "${USE_CONTAINER:-0}" = "1" ] && GADI_LAUNCHER="gadi_container_go.sh"
                CMD="qsub -v ${EXPORTVARS} -N ${UW_MODEL} -l storage=gdata/m18+scratch/m18+scratch/el06,ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} ${GADI_LAUNCHER}"
                echo ${CMD}
                ${CMD}
            else
                export QUEUE="work"
                export OUTNAME="Model_${UW_MODEL}_Res_${UW_RESOLUTION}_Nproc_${NTASKS}_JobID_%j.out"
                CMD="sbatch --export=IMAGE,${EXPORTVARS} --job-name=${UW_MODEL} --output=${OUTNAME} --ntasks=${NTASKS} --time=${WALLTIME} --account=${ACCOUNT} --partition=${QUEUE} setonix_baremetal_go.sh"
                echo ${CMD}
                ${CMD}
            fi
        done
    done

    popd > /dev/null
done
