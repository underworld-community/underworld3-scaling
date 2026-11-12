#!/bin/bash -l
#PBS -N mesh_generate
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/el06+gdata/m18
#PBS -P el06
#PBS -j oe

# Gmsh runs single-process — no MPI needed, so we use singularity exec directly.
# Gadi charges max(ncpus, mem/4GB): 32 GB is enough up to box res~168; the res=240
# box (~14M tets) needed 64 GB and ~2 h on its own, and the BASE-10 stokes res=100
# shell (job 10) needed the hugemem queue at 1000 GB / 8 h.

cd ~/uw3-scaling-scripts
source params.sh

export CONTAINER=${UW_CONTAINER:?set UW_CONTAINER in params.sh}
export EXTRA_PKGS=/scratch/el06/jg0883/uw3-extra-pkgs

module load singularity

# One line per mesh family this round, "model:base:jobs" — independent of params.sh,
# whose BASE only describes the box models.
MESH_SETS=(
    "poisson-scaling:24:1 2 3 4 5 7 10"
    "stokes-scaling:5:1 2 3 4 5 7 10"
)

echo "=== Mesh pre-generation ==="
echo "MESH_CACHE = ${MESH_CACHE}"
echo "CONTAINER  = ${CONTAINER}"

for spec in "${MESH_SETS[@]}"; do
    IFS=: read -r MODEL BASE JOBS <<< "${spec}"
    echo ""
    echo "--- ${MODEL}  BASE=${BASE}  JOBS=${JOBS}"
    singularity exec \
        --bind /half-root \
        --bind /opt/pbs/default/lib \
        ${CONTAINER} \
        bash -c \
        "PYTHONPATH=${EXTRA_PKGS}:\${PYTHONPATH} \
         python3 generate_meshes.py \
         --models '${MODEL}' \
         --base   ${BASE} \
         --jobs   '${JOBS}' \
         --cache  ${MESH_CACHE}"
done

echo "=== Done ==="
