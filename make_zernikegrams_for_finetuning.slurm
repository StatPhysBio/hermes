#!/bin/bash
#SBATCH --job-name=zernikegrams
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/zernikegrams.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/zernikegrams.log

python -u make_zernikegrams_for_finetuning.py \
                -m hermes_py_000 \
                -n 0 \
                -pd /gscratch/stf/gvisan01/casp12/pdbs \
                -o /gscratch/stf/gvisan01/hermes/training_data/finetuning/casp12_relaxed_pred/premade_zernikegrams \
                -pr /gscratch/stf/gvisan01/hermes/training_data/finetuning/casp12_relaxed_pred/pdb_to_residues.json

