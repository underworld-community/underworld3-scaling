#!/bin/bash
export JOBS="8 9"                   # 3D multipliers or procs
export SS_MEMORY="192GB 1920GB"     # memory for EACH job
export RUN_INDICES="1 2 3"         # the indices to use to use as label. This is for averaging
export WALLTIME="01:00:00"

export SCALING_TYPE=1               # 1=weak, 2=strong
export SCALING_BASE=3
export PBSTASK_MULT=1               # multiplier for requested PBSTASKS so you can increase memory request (e.g. use 384 GB, but you only need < 48 procs)

export UW_DIM=3                     # 2 or 3

export UW_SOL_TOLERANCE=1e-6        # tolerance to use
export UW_MODEL="stokes-scaling"    # filename (w/o extension) to use in scaling

export UW_NAME="uw3_May2025"        # uw version name

export PICKLENAME=None              # "None" to disable conv testing
###

export ACCOUNT="m18"
export QUEUE="normal"               # normal or express

# Test style - UW_MAX_ITS (+ve, recommended >100): Fixed work, (-ve): Accuracy (UW_SOL_TOLERANCE is used)
export UW_MAX_ITS=50                # ensure that your model handles this

# unused May 2025
export UW_PENALTY=-1.               # set to negative value to disable penalty
export UW_ENABLE_IO="0"             # Jan 2024 - not used
###
