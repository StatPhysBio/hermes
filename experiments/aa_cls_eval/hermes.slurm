#!/bin/bash
#SBATCH --job-name=aa_cls_eval
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/aa_cls_eval.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/aa_cls_eval.log

bash hermes.sh
