#!/bin/bash
#SBATCH --time=360
#SBATCH --gres=gpu:4,gpu_mem:2500M
#SBATCH --output=logs/slurm-%j.out

## Write your command here

python main.py --mode train --data_dir wiki/ --dataset wiki --save_dir save/ --best_dir save_best --config_file config/sgd.yml --lm ngram-lm --loss_mode mixed --mixed_constant 0.4 --job_id wiki_int_l2_0 --device gpu
