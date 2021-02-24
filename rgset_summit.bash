#!/bin/bash

####################### BSUB Headers ############################################################
#BSUB -P bip198
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -q batch
#BSUB -alloc_flags "gpumps"
#BSUB -J train_lit_set_model_with_pl
#BSUB -o /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Lab_Repositories/RGSET/job%J.out
#BSUB -e /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Lab_Repositories/RGSET/job%J.out
#BSUB --signal=SIGUSR1@90
#################################################################################################

# Remote project path
export PROJDIR=$MEMBERWORK/bip198/Repositories/Lab_Repositories/RGSET

# Configure Conda for BSUB script environment
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Run training script
cd "$PROJDIR"/project || exit
jsrun -r1 -g1 -a6 -c42 -bpacked:7 python lit_set.py
