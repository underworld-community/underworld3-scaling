## uw3-scaling-scripts

To use the scaling scripts:

1. prepare model script as needed (see examples)
2. params.sh contains important parameters that must be set accordingly:
	- JOBS:
		- If doing strong scaling, set to the processors to test.
		- If doing weak scaling, set to resolution multiplier (ensure that elem or dof per proc is constant).
	- SS_MEMORY: list of memory to request for each entry in JOBS
	- RUN_INDICES: indices to use as label for each entry in JOBS. You can use this to do several trials for each JOB (e.g., "1 2 3" for three trials, "1" for just one trial)
	- WALLTIME: the wall time for ALL the jobs
	- SCALING_TYPE: se to 1 for weak scaling, 2 for strong scaling
	- SCALING_BASE:
		- if doing strong scaling, the resolution you wish to use. All JOBS will use the same resolution value.
		- if doing weak scaling, the base resolution you wish to use. For example, SCALING_BASE is set to 32 and JOBS is "1 2 3", then the script with run models with resolution equal to 32, 64, and 96.
		- make sure your model handles this together with the JOBS variable.
	- PBSTASKS_MULT: mutiplier for requested PBSTASKS so that you can increase memory request. For example, in Gadi ,when you want to run with < 48 procs but wish to request for 384 GB of memory. If you don't want this kind of scenario, just set PBSTASKS_MULT to 1.
	- UW_DIM: dimension of the problem
	- UW_SOL_TOLERANCE: the solver tolerance you wish to use. Ensure that your model script can read this value.
	- UW_MODEL: the filename of the model you wish to test without the ".py" extension.
	- UW_NAME: underworld version name. This is just used for naming the output directory.
	- PICKLENAME: set to None as this is not used
	- ACCOUNT: HPC account you wish to use
	- QUEUE: the HPC queue you wish to use
	- UW_MAX_ITS:
		- set to positive integer to have consistent number of iterations for each test (i.e. ignores tolerance)
		- set to negative integer if you wish to ignore
		- make sure your model handles this
3. Run ./scaling_test_job_launcher.sh
4. Check the outputs and plot them.
