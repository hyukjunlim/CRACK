#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # Cores per node
#SBATCH --partition=snu-gpu1     # Partition Name
#SBATCH --job-name=mpflow     # Default job name
#SBATCH --time=10-00:00:00          # Runtime: 10 days
#SBATCH -o output/%x.%N.%j.out   # STDOUT with job name in output directory
#SBATCH -e output/%x.%N.%j.err   # STDERR with job name in output directory
#SBATCH --gres=gpu:a6000:1       # Request 1 A5000 GPU

# # Check if a script was provided as an argument
# if [ $# -eq 0 ]; then
#     echo "Error: No Python script specified"
#     echo "Usage: sbatch run_slurm.sh <python_script.py> [job_name]"
#     exit 1
# fi

# # Get the Python script from the first argument
# PYTHON_SCRIPT=$1

# # Set job name from second argument or use script name as job name
# if [ $# -ge 2 ]; then
#     # A job name was provided as the second argument
#     SLURM_JOB_NAME=$2
# else
#     # Use the script name (without .py) as the job name
#     SLURM_JOB_NAME=$(basename ${PYTHON_SCRIPT} .py)
# fi

# mkdir output
mkdir -p output

# Update the job name
scontrol update job $SLURM_JOB_ID name=$SLURM_JOB_NAME

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running script: ${PYTHON_SCRIPT}"
echo "Job name: ${SLURM_JOB_NAME}"

module purge

StartTime=$(date +%s)
# cd $SLURM_SUBMIT_DIR

# Run the specified Python script
sh scripts/train/oc20/s2ef/equiformer_v2/153M_exp.sh

EndTime=$(date +%s)

# Calculate and display runtime
RUNTIME=$((EndTime - StartTime))
echo "Run time:"
echo "${RUNTIME} sec"
echo "$(echo "scale=2; ${RUNTIME}/60" | bc) min"
echo "$(echo "scale=2; ${RUNTIME}/3600" | bc) hour"
echo "Job completed at: $(date)"

