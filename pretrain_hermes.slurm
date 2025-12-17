#!/bin/bash
#SBATCH --job-name=so3_convnet_12p5_nb_noise
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvisan01@uw.edu
#SBATCH --export=all
#SBATCH -e ./slurm_logs/so3_convnet_12p5_nb_noise.err
#SBATCH -o ./slurm_logs/so3_convnet_12p5_nb_noise.out
#SBATCH --array=0-9

python pretrain_hermes.py \
        -c /gscratch/stf/gvisan01/hermes/training_data/pretraining/config/so3_convnet_12p5_nb.yaml \
        -i $SLURM_ARRAY_TASK_ID \
        -o /gscratch/stf/gvisan01/hermes/trained_models/hermes_py_000_12p5_nb \
        -n 1
