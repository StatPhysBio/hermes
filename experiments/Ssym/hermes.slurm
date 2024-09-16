#!/bin/bash
#SBATCH --job-name=ssym
#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/ssym.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/ssym.log

bash hermes.sh
