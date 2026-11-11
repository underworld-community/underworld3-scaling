#!/bin/bash
# Stokes layer over params.sh:  ./scaling_test_job_launcher.sh params-stokes.sh
# Fixed-tolerance protocol (see README): BASE 5, tol 1e-8, up to 100 outer iterations.
source "$(dirname "${BASH_SOURCE[0]}")/params.sh"

export UW_MODELS="stokes-scaling"
export UW_MODEL="stokes-scaling"
export SCALING_BASE=5
export UW_SOL_TOLERANCE=1e-8
export UW_MAX_ITS=100
export WALLTIME="01:00:00"
