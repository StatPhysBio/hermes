#!/bin/bash
#SBATCH --job-name=so3_convnet_from_scratch_pro_05_noise=0.5_seed=10006
#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvisan01@uw.edu
#SBATCH --export=all
#SBATCH -e ./slurm_logs/so3_convnet_from_scratch_pro_05_noise=0.5_seed=10006.err
#SBATCH -o ./slurm_logs/so3_convnet_from_scratch_pro_05_noise=0.5_seed=10006.out

python train_and_eval.py \
            --model_dir runs/so3_convnet_from_scratch_pro_05_noise=0.5_seed=10006 \
            --training_config config/so3_convnet_noise_prop_05_6.yaml
