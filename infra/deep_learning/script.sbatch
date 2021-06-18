#!/bin/bash
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=johannes.kiechle@tum.de
#SBATCH -N 1
#SBATCH -p dgx-1-v100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH -o DI-LABxVW.out
#SBATCH -e DI-LABxVW.err 
srun enroot start --root --mount ./mount_dir:/workspace/mount_dir ./containers/deep_learning.sqsh horovodrun -n 4 python3 ./TUM-DI-Lab/src/deep_learning/main.py