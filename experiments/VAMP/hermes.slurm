#!/bin/bash
#SBATCH --job-name=vamp
#SBATCH --account=stf
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=44G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/vamp.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/vamp.log

bash hcnn.sh
