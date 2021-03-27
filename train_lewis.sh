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
export PROJDIR=/home/$USER/data/Equivariant-GNNs
export DGLBACKEND=pytorch # Required to override default ~/.dgl config directory which is read-only

# Configure Conda for SBATCH script environment
module load miniconda3
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Load CUDA module for DGL
module load cuda/cuda-10.1.243

# Run training script
cd "$PROJDIR"/project || exit

START=$(date +%s) # Capture script start time in seconds since Unix epoch
echo "Script started at $(date)"

# Execute script
python3 lit_set.py --num_layers 2 --num_channels 32 --num_nearest_neighbors 3 --batch_size 4 --lr 0.001 --num_epochs 25 num_workers 28

END=$(date +%s) # Capture script end time in seconds since Unix epoch
echo "Script finished at $(date)"

# Calculate and output time elapsed during script execution
((diff = END - START))
((seconds = diff))
((minutes = seconds / (60)))
((hours = minutes / (24)))
echo "Script took $seconds second(s) to execute"
echo "Script took $minutes minute(s) to execute"
echo "Script took $hours hour(s) to execute"
