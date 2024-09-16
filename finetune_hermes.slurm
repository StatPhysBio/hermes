#!/bin/bash
#SBATCH --job-name=finetune_$SLURM_ARRAY_TASK_ID
#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e slurm_logs/finetune_$SLURM_ARRAY_TASK_ID.err
#SBATCH -o slurm_logs/finetune_$SLURM_ARRAY_TASK_ID.log
#SBATCH --array=0-9

python finetune_hermes.py \
            -c /gscratch/spe/gvisan01/hermes/training_data/finetuning/cdna117k/configs/test.yaml \
            -i $SLURM_ARRAY_TASK_ID
