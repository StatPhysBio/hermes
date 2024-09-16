#!/bin/bash
#SBATCH --job-name=TIM_SULSO_b0
#SBATCH --account=stf
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=44G
#SBATCH -e ./__slurm_logs/TIM_SULSO_b0.err
#SBATCH -o ./__slurm_logs/TIM_SULSO_b0.out

cd TIM_SULSO_b0
bash hermes.sh
