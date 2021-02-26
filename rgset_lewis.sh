#!/bin/bash

####################### Batch Headers #########################
#SBATCH -p Lewis
#SBATCH -J train_lit_set_model_with_pl
#SBATCH -t 0-02:00
#SBATCH --partition Gpu
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node 24
#SBATCH --mem 64G
#SBATCH --nodes 1
#SBATCH --signal=SIGUSR1@90
###############################################################

# Remote project path
export PROJDIR=/home/$USER/data/RGSET

# Configure Conda for SBATCH script environment
module load miniconda3
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Load CUDA module for DGL
module load cuda/cuda-10.1.243

# Run training script
cd "$PROJDIR"/project || exit
python3 lit_set.py