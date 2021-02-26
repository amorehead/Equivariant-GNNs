#!/bin/bash

####################### BSUB Headers ############################################################
#BSUB -P bip198
#BSUB -W 00:30
#BSUB -nnodes 1
#BSUB -q batch
#BSUB -alloc_flags "gpumps"
#BSUB -J train_lit_set_model_with_pl
#BSUB -o /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Lab_Repositories/Equivariant-GNNs/job%J.out
#BSUB -e /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Lab_Repositories/Equivariant-GNNs/job%J.out
#BSUB --signal=SIGUSR1@90
#################################################################################################

# Remote project path and DGL backend override
export PROJDIR=$MEMBERWORK/bip198/Repositories/Lab_Repositories/Equivariant-GNNs
export DGLBACKEND=pytorch  # Required to override default ~/.dgl config directory which is read-only

# Configure Conda for BSUB script environment
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Configure Weights and Biases (Wandb) for local configuration storage and proxy access on compute nodes:
export WANDB_CONFIG_DIR=.
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'

# Run training script
cd "$PROJDIR"/project || exit
jsrun -r1 -g6 -a6 -c42 -bpacked:7 python lit_set.py
