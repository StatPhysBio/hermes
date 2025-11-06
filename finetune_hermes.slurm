#!/bin/bash
#SBATCH --job-name=finetune_$SLURM_ARRAY_TASK_ID
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e slurm_logs/finetune_$SLURM_ARRAY_TASK_ID.err
#SBATCH -o slurm_logs/finetune_$SLURM_ARRAY_TASK_ID.log
#SBATCH --array=0-9

python finetune_hermes.py \
            -c /gscratch/stf/gvisan01/hermes/training_data/finetuning/cdna117k/configs/hermes_py_000_ft_casp12_perc0p5_relaxed_pred.yaml \
            -i $SLURM_ARRAY_TASK_ID
