#!/bin/bash
#SBATCH --job-name=t2837
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=44G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/t2837.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/t2837.log

bash hermes.sh
