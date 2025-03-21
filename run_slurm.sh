#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # Cores per node
#SBATCH --partition=snu-gpu1     # Partition Name
#SBATCH --job-name=flow_matching     # Default job name
#SBATCH --time=07:00:00          # Runtime: 7 hours
#SBATCH -o output/%x.%N.%j.out   # STDOUT with job name in output directory
#SBATCH -e output/%x.%N.%j.err   # STDERR with job name in output directory
#SBATCH --gres=gpu:1       # Request 1 A5000 GPU

nvidia-smi

python train_flow_matching.py