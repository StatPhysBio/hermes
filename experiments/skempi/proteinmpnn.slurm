#!/bin/bash
#SBATCH --job-name=skempi
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e proteinmpnn_skempi.err
#SBATCH -o proteinmpnn_skempi.log

bash proteinmpnn_skempi.sh
