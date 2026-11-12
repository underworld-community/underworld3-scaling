#!/bin/bash
# Dense-swarm advection layer over params.sh:  ./scaling_test_job_launcher.sh params-swarm.sh
# No solve: tolerance and maxits are accepted by the script and unused. Same box meshes as
# poisson (BASE 24); UW_FILL sets particles per cell, UW_NSTEPS the advection steps.
source "$(dirname "${BASH_SOURCE[0]}")/params.sh"

export UW_MODELS="swarm-scaling"
export UW_MODEL="swarm-scaling"
export JOBS="1 2 3 4 5 7"
export RUN_INDICES="1 2"
export WALLTIME="03:00:00"
export UW_FILL=3
export UW_NSTEPS=10
