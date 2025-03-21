#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # Cores per node
#SBATCH --partition=snu-gpu1     # Partition Name
#SBATCH --job-name=eqv2     # Default job name
#SBATCH --time=07:00:00          # Runtime: 7 hours
#SBATCH -o output/%x.%N.%j.out   # STDOUT with job name in output directory
#SBATCH -e output/%x.%N.%j.err   # STDERR with job name in output directory
#SBATCH --gres=gpu:A5000:1       # Request 1 A5000 GPU

python train_flow_matching.py

# # Get the Python script from the first argument
# PYTHON_SCRIPT="train_flow_matching"
# SLURM_JOB_NAME="train_flow_matching"

# # Update the job name
# scontrol update job $SLURM_JOB_ID name=$SLURM_JOB_NAME

# # Create output directory if it doesn't exist
# mkdir -p output

# echo "Running on host: $(hostname)"
# echo "Job started at: $(date)"
# echo "Running script: ${PYTHON_SCRIPT}"
# echo "Job name: ${SLURM_JOB_NAME}"

# module purge

# StartTime=$(date +%s)

# # Run the specified Python script
# python ${PYTHON_SCRIPT}.py

# EndTime=$(date +%s)

# # Calculate and display runtime
# RUNTIME=$((EndTime - StartTime))
# echo "Run time:"
# echo "${RUNTIME} sec"
# echo "$(echo "scale=2; ${RUNTIME}/60" | bc) min"
# echo "$(echo "scale=2; ${RUNTIME}/3600" | bc) hour"
# echo "Job completed at: $(date)"

