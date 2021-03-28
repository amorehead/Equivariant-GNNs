#!/bin/bash
####################### BSUB Headers ############################################################
#BSUB -J train_lit_set_model_with_pl
#BSUB -P bip198
#BSUB -W 00:10
#BSUB -nnodes 2
#BSUB -q batch
#BSUB -alloc_flags "gpumps NVME"
#BSUB -o /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Lab_Repositories/Equivariant-GNNs/job%J.out
#BSUB -e /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Lab_Repositories/Equivariant-GNNs/job%J.out
#BSUB --signal=SIGUSR1@90
#################################################################################################

# Remote project path and DGL backend override
export USER=acmwhb
export PROJID=bip198
export PROJDIR=$MEMBERWORK/$PROJID/Repositories/Lab_Repositories/Equivariant-GNNs
export DGLBACKEND=pytorch # Required to override default ~/.dgl config directory which is read-only

# Configure OMP for PyTorch
export OMP_PLACES=threads

# Configure Conda for BSUB script environment
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Configure WandB logger for local configuration storage and proxy access on compute nodes
export WANDB_CONFIG_DIR=/gpfs/alpine/scratch/$USER/$PROJID/ # For local reading and writing of WandB files
export WANDB_CACHE_DIR=/gpfs/alpine/scratch/$USER/$PROJID/  # For logging checkpoints as artifacts
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'

# Run training script
cd "$PROJDIR"/project || exit

START=$(date +%s) # Capture script start time in seconds since Unix epoch
echo "Script started at $(date)"

# Execute script
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_set.py --num_layers 2 --num_channels 32 --num_nearest_neighbors 3 --batch_size 4 --lr 0.001 --num_epochs 25 num_workers 28 --tb_log_dir /mnt/bb/$USER/tb_log --ckpt_dir /mnt/bb/$USER/checkpoints

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

# Copying leftover items from NVMe drive back to GPFS
echo "Copying log files and best checkpoint(s) back to GPFS..."
jsrun -n 1 cp -r /mnt/bb/$USER/tb_log /gpfs/alpine/scratch/$USER/$PROJID/Repositories/Lab_Repositories/Equivariant-GNNs/project/tb_log
jsrun -n 1 cp -r /mnt/bb/$USER/checkpoints /gpfs/alpine/scratch/$USER/$PROJID/Repositories/Lab_Repositories/Equivariant-GNNs/project/checkpoints
echo "Done copying log files and best checkpoint(s) back to GPFS"