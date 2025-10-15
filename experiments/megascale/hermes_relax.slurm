#!/bin/bash
#SBATCH --job-name=megascale
#SBATCH --account=stf
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=16G
#SBATCH --export=all
#SBATCH -e /gscratch/scrubbed/gvisan01/slurm_logs/megascale.err
#SBATCH -o /gscratch/scrubbed/gvisan01/slurm_logs/megascale.log
#SBATCH --array=0-159


# model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_000_untrained_small_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_050_untrained_small_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'
# model_version_list='hermes_bp_050_ft_cdna117k_ddg_st' # hermes_py_050_ft_cdna117k_w_boot_ddg_st_squared'
model_version_list='hermes_py_050'

base_dir='./'
output_dir=$base_dir'results_relaxed_nrep1_ens1_nowt_side12_bb0/'

for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../mutation_effect_prediction_with_hermes_with_relaxation.py \
                        --model_version $model_version \
                        --csv_file $base_dir'test_targets.csv' \
                        --folder_with_pdbs /gscratch/spe/gvisan01/hermes/training_data/finetuning/mega_thermompnn_neg/pdbs \
                        --output_dir $output_dir \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --relax_wt 0 \
                        --nrepeats 1 \
                        --num_relaxations_ensemble 1 \
                        --num_splits 160 \
                        --sidechain_flexible_distance_threshold 12.0 \
                        --backbone_flexible_distance_threshold 0.0 \
                        --split_idx $SLURM_ARRAY_TASK_ID

done


# python -u ../../mutation_effect_prediction_with_hermes_with_relaxation.py \
#                     --model_version hermes_py_050 \
#                     --csv_file './test_targets.csv' \
#                     --folder_with_pdbs /gscratch/spe/gvisan01/hermes/training_data/finetuning/mega_thermompnn_neg/pdbs \
#                     --output_dir ./results_relaxed_nrep1_ens1_nowt_side12_bb0/ \
#                     --wt_pdb_column pdbid \
#                     --mutant_column variant \
#                     --mutant_chain_column chainid \
#                     --mutant_split_symbol"=|" \
#                     --relax_wt 0 \
#                     --nrepeats 1 \
#                     --num_relaxations_ensemble 1 \
#                     --num_splits 20000 \
#                     --sidechain_flexible_distance_threshold 12.0 \
#                     --backbone_flexible_distance_threshold 0.0 \
#                     --split_idx 0
