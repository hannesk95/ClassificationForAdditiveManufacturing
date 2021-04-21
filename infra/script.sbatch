#!/bin/bash
#SBATCH -N 1
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o enroot_test.out
#SBATCH -e enroot_test.err
 
srun --container-mounts=./data-test:/mnt/data-test \
     --container-image='path/to/container-image' \
     python3 example.py