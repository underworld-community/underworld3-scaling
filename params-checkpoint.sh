#!/bin/bash
# Checkpoint I/O + evaluate layer over params.sh:  ./scaling_test_job_launcher.sh params-checkpoint.sh
# No solve: tolerance and maxits are accepted by the script and unused. Same box meshes as
# poisson (BASE 24). Round 2026-11: the six points that have a v3.1.0 pair (evalbarrier).
source "$(dirname "${BASH_SOURCE[0]}")/params.sh"

export UW_MODELS="checkpoint-scaling"
export UW_MODEL="checkpoint-scaling"
export JOBS="1 2 3 4 5 7"
export RUN_INDICES="1 2"
export WALLTIME="01:00:00"
