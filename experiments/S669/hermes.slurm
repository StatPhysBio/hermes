#!/bin/bash
#SBATCH --job-name=s669
#SBATCH --account=stf
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/s669.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/s669.log

bash hermes.sh

