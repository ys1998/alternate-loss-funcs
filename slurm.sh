#!/bin/bash
#SBATCH --time=360
#SBATCH --gres=gpu:4,gpu_mem:1500M
#SBATCH --cpus-per-task=48
#SBATCH --output=slurm-%j.out

## Write your command here
