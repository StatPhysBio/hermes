#!/bin/bash
#SBATCH --job-name=install
#SBATCH --account=spe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --export=all
#SBATCH -e slurm_logs/install.err
#SBATCH -o slurm_logs/install.log

yes | conda install zernikegrams=1.0.11 pytorch-cuda=12.4 -c statphysbio -c conda-forge -c bioconda -c pytorch -c nvidia
yes | pip install -e .

