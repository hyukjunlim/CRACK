#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # Cores per node
#SBATCH --partition=snu-gpu1     # Partition Name
#SBATCH --job-name=eqv2     # Default job name
#SBATCH --time=07:00:00          # Runtime: 7 hours
#SBATCH -o output/%x.%N.%j.out   # STDOUT with job name in output directory
#SBATCH -e output/%x.%N.%j.err   # STDERR with job name in output directory
#SBATCH --gres=gpu:1       # Request 1 A5000 GPU

# Get the Python script from the first argument
PYTHON_SCRIPT="main_oc20"
SLURM_JOB_NAME="main_oc20"

# Update the job name
scontrol update job $SLURM_JOB_ID name=$SLURM_JOB_NAME

# Create output directory if it doesn't exist
mkdir -p output

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running script: ${PYTHON_SCRIPT}"
echo "Job name: ${SLURM_JOB_NAME}"

module purge

StartTime=$(date +%s)
cd $SLURM_SUBMIT_DIR

# Run the specified Python script
# python ${PYTHON_SCRIPT}
sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@2M_g@1.sh

EndTime=$(date +%s)

# Calculate and display runtime
RUNTIME=$((EndTime - StartTime))
echo "Run time:"
echo "${RUNTIME} sec"
echo "$(echo "scale=2; ${RUNTIME}/60" | bc) min"
echo "$(echo "scale=2; ${RUNTIME}/3600" | bc) hour"
echo "Job completed at: $(date)"

