#!/bin/bash -l
#PBS -N mesh_generate
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/el06+gdata/m18
#PBS -P el06
#PBS -j oe

# Gmsh runs single-process — no MPI needed, so we use singularity exec directly.
# 64 GB covers the largest box mesh in the campaign (res=240, ~14M tets).
# Walltime 6 h is generous; expect ~2-3 h for the full set at BASE=24.

cd ~/uw3-scaling-scripts
source params.sh

export CONTAINER=/g/data/m18/software/containers/underworld3-gadi_v3.1.0.sif
export EXTRA_PKGS=/scratch/el06/jg0883/uw3-extra-pkgs

module load singularity

# Campaign-specific overrides (independent of whatever params.sh is set to)
SCALING_BASE=24
JOBS="1 2 3 4 5"
MODELS="poisson-scaling"   # one box model is enough; box mesh is shared by all rectangular models

echo "=== Mesh pre-generation ==="
echo "SCALING_BASE = ${SCALING_BASE}"
echo "JOBS         = ${JOBS}"
echo "MESH_CACHE   = ${MESH_CACHE}"
echo "Models       = ${MODELS}"
echo ""

singularity exec \
    --bind /half-root \
    --bind /opt/pbs/default/lib \
    ${CONTAINER} \
    bash -c \
    "PYTHONPATH=${EXTRA_PKGS}:\${PYTHONPATH} \
     python3 generate_meshes.py \
     --models '${MODELS}' \
     --base   ${SCALING_BASE} \
     --jobs   '${JOBS}' \
     --cache  ${MESH_CACHE}"

echo "=== Done ==="
