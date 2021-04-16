#!/bin/bash
# Begin LSF directives
#BSUB -P BIF132
#BSUB -J test
#BSUB -o tst.o%J
#BSUB -W 00:30
#BSUB -nnodes 4
#BSUB -alloc_flags "nvme smt4"
# End LSF directives and begin shell commands

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
module load open-ce/1.1.3-py38-0
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
CMD="python -u test.py"
jsrun -n${NODES} -a6 -c42 -g6 -r1 --bind=proportional-packed:7 --launch_distribution=packed ./launch.sh "$CMD"
