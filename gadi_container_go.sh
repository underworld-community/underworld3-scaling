#!/bin/bash -l

# load Singularity
module load singularity
module load openmpi/4.1.4

export singularityDir=/home/157/jg0883/uw3-container

# Define the container to use
export containerImage=$singularityDir/underworld3_0.9-x86_64.sif

#source /home/jovyan/uw3/pypathsetup.sh
#cd /home/jovyan/uw3/underworld3/
#source pypathsetup.sh
#cd -
#ls -lrt

env
cat timed_model.py
echo ""
echo "---------- Running Job ----------"
echo ""
export TIME_LAUNCH_MPI=`date +%s%N | cut -b1-13`
mpiexec -n $PBS_NCPUS singularity exec $containerImage bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 timed_model.py"
