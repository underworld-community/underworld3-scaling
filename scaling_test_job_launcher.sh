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

export NAME="${TYPE}_${UW_NAME}_DIM${UW_DIM}_BASE${SCALING_BASE}_TOL${UW_SOL_TOLERANCE}_MAXITS${UW_MAX_ITS}_MODEL_${UW_MODEL}"
MEM_ARRAY=($SS_MEMORY)

## find the BATCH environment ##
#################################
if qstat --version &> /dev/null ; then
   BATCH_SYS="PBS"
   export NAME="${NAME}_Gadi"
elif squeue --version &> /dev/null ; then
   BATCH_SYS="SLURM"
   export NAME="${NAME}_Setonix"
else
   echo "Can't determine batch system"
   exit 1
fi

echo "Batch system is $BATCH_SYS"
#################################
mkdir -p ${NAME}
#cp -rf .meshes ${NAME} # only used for NS scaling test
cp *.sh ${NAME}
cp *.py ${NAME}
cd ${NAME}

#for j in $(seq 1 ${RUN_INDICES}) # different way to loop

ITER=0 # counts the job number
for i in ${JOBS}
do
    export JOB_IDX="${i}"
    echo ${JOB_IDX}

    for j in ${RUN_INDICES}
    do
        export RUN_IDX="${j}"
        echo ${RUN_IDX}

        if (($SCALING_TYPE==1)); then   # weak
            export UW_RESOLUTION="$((${SCALING_BASE} * ${i}))"

            if (($UW_DIM==2)); then
                export NTASKS="$((${i}*${i}))" # 2D problem
            else
                export NTASKS="$((${i}*${i}*${i}))"
            fi
        else                            # strong
            export UW_RESOLUTION=${SCALING_BASE}
            export NTASKS=${i}
        fi
        export EXPORTVARS="TYPE,UW_RESOLUTION,NTASKS,UW_DIM,UW_SOL_TOLERANCE,UW_MAX_ITS,UW_MODEL,SCALING_TYPE,UW_NAME,JOB_IDX,RUN_IDX,NAME"
        if [ $BATCH_SYS == "PBS" ] ; then
            PBSTASKS=`python3 <<<"print((int(${NTASKS}/48) + (${NTASKS} % 48 > 0))*48*${PBSTASK_MULT})"`  # round up to nearest 48 as required by nci
            #PBSTASKS=`python3 <<<"print(int(${NTASKS}))"`
            if (($SCALING_TYPE==1)); then     # WEAK SCALING

                MEMORY=${MEM_ARRAY[${ITER}]}
                #CMD="qsub -v ${EXPORTVARS} -N ${NAME} -l storage=gdata/m18+gdata/q97,ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} gadi_container_go.sh"
                CMD="qsub -v ${EXPORTVARS} -N ${NAME} -l storage=gdata/m18+scratch/el06,ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} gadi_baremetal_go.sh"
                echo ${CMD}
                ${CMD}

            else                             # STRONG SCALING
                MEMORY=${MEM_ARRAY[${ITER}]}
                CMD="qsub -v ${EXPORTVARS} -N ${NAME} -l storage=gdata/m18+scratch/el06,ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} gadi_baremetal_go.sh"
                echo ${CMD}
                ${CMD}

                # for running one job at a time
                #MEMORY="$((2*${PBSTASKS}))GB" # for 2D
                # if (($ITER==0)); then
                #     CMD="qsub -v ${EXPORTVARS} -N ${NAME} -l storage=scratch/el06,ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} gadi_baremetal_go.sh"
                #     echo ${CMD}
                #     jobid=$(${CMD})
                # else # wait for the previous job to finish
                #     CMD="qsub -W depend=afterany:${jobid} -v ${EXPORTVARS} -N ${NAME} -l storage=scratch/el06,ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} gadi_baremetal_go.sh"
                #     echo ${CMD}
                #     jobid=$(${CMD})
                # fi
            fi
        else
            if [[ "$QUEUE" == "express" ]] ; then
                export QUEUE="work"
            else
                export QUEUE="work"
            fi
            export OUTNAME="Res_"${UW_RESOLUTION}"_Nproc_"${NTASKS}"_JobID_"%j".out"

            # Container
            #CMD="sbatch --export=IMAGE,${EXPORTVARS} --job-name=${NAME} --output=${OUTNAME} --ntasks=${NTASKS} --time=${WALLTIME} --account=${ACCOUNT} --partition=${QUEUE} setonix_container_go.sh"

            # Baremetal
            CMD="sbatch --export=IMAGE,${EXPORTVARS} --job-name=${NAME} --output=${OUTNAME} --ntasks=${NTASKS} --time=${WALLTIME} --account=${ACCOUNT}  setonix_baremetal_go.sh"
            echo ${CMD}
            ${CMD}
        fi
    done
    ITER=$(expr $ITER + 1)
done
