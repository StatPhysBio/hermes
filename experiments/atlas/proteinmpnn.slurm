#!/bin/bash
#SBATCH --job-name=atlas
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e proteinmpnn_atlas.err
#SBATCH -o proteinmpnn_atlas.log

bash proteinmpnn_atlas.sh
