#!/bin/bash
#SBATCH --job-name=hw4_q4_rtx8000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --output=%x.out
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
--overlay /scratch/app7533/hw3/overlay-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "source /ext3/env.sh;
cd /scratch/app7533/app7533_hw4;
python 4.py"
