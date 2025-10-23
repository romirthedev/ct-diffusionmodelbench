#!/bin/bash
#SBATCH -J LLaDA_FineTune      # Job name
#SBATCH -o LLaDA_FineTune.o%j   # Output file name
#SBATCH -e LLaDA_FineTune.e%j   # Error file name
#SBATCH -p gh-dev               # Queue name (changed to gh-dev for GPU support)
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of tasks (cores requested)
#SBATCH -t 02:00:00             # Run time (hh:mm:ss)


# Load necessary modules
module load gcc
module load cuda
module load python3

# Set Hugging Face Cache Directory
export HF_HOME=$SCRATCH/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Activate your virtual environment
source /scratch/10936/romirpatel/ct-diffusionmodelbench/venv/bin/activate

# Navigate to the directory containing your train.py script
cd /scratch/10936/romirpatel/ct-diffusionmodelbench/Training

# Run your Python training script
/scratch/10936/romirpatel/ct-diffusionmodelbench/venv/bin/python train.py