

# model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'
# model_version_list='hermes_py_000_untrained_small_ft_cdna117k_ddg_st hermes_py_050_untrained_small_ft_cdna117k_ddg_st'

# model_version_list='hermes_py_000_ft_cdna117k_relaxed_pred hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_relaxed_pred hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st'
# model_version_list='hermes_py_000_ft_mega_thermompnn_ddg_st hermes_py_050_ft_mega_thermompnn_ddg_st'
# model_version_list='hermes_py_050_untrained_ft_mega_thermompnn_ddg_st hermes_py_050_untrained_small_ft_mega_thermompnn_ddg_st'
model_version_list='hermes_py_000_untrained_ft_mega_thermompnn_ddg_st hermes_py_000_untrained_small_ft_mega_thermompnn_ddg_st'


for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../mutation_effect_prediction_with_hermes.py \
                        --model_version $model_version \
                        --csv_file Ssym_dir/ssym_dir_ddg_experimental.csv \
                        --folder_with_pdbs Ssym_dir/pdbs/ \
                        --output_dir Ssym_dir/results/ \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure 0
    
    python -u correlations.py \
                --system_name Ssym_dir \
                --model_version $model_version \
                --use_mt_structure 0


    python -u ../../mutation_effect_prediction_with_hermes.py \
                        --model_version $model_version \
                        --csv_file Ssym_inv/ssym_inv_ddg_experimental.csv \
                        --folder_with_pdbs Ssym_inv/pdbs/ \
                        --output_dir Ssym_inv/results/ \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure 0
    
    python -u correlations.py \
                --system_name Ssym_inv \
                --model_version $model_version \
                --use_mt_structure 0


    # python -u ../../mutation_effect_prediction_with_hermes.py \
    #                     --model_version $model_version \
    #                     --csv_file Ssym_naive_inv/ssym_naive_inv_ddg_experimental.csv \
    #                     --folder_with_pdbs Ssym_naive_inv/pdbs/ \
    #                     --output_dir Ssym_naive_inv/results/ \
    #                     --wt_pdb_column pdbid \
    #                     --mutant_column variant \
    #                     --mutant_chain_column chainid \
    #                     --mutant_split_symbol"=|" \
    #                     --use_mt_structure 0
    
    # python -u correlations.py \
    #             --system_name Ssym_naive_inv \
    #             --model_version $model_version \
    #             --use_mt_structure 0


    # python -u ../../mutation_effect_prediction_with_hermes.py \
    #                     --model_version $model_version \
    #                     --csv_file Ssym_dir_with_mut_structs/ssym_dir_with_mut_structs_ddg_experimental.csv \
    #                     --folder_with_pdbs Ssym_dir_with_mut_structs/pdbs/ \
    #                     --output_dir Ssym_dir_with_mut_structs/results/ \
    #                     --wt_pdb_column pdbid \
    #                     --mt_pdb_column mt_pdbid \
    #                     --mutant_column variant \
    #                     --mutant_chain_column chainid \
    #                     --mutant_split_symbol"=|" \
    #                     --use_mt_structure 1
    
    # python -u correlations.py \
    #             --system_name Ssym_dir_with_mut_structs \
    #             --model_version $model_version \
    #             --use_mt_structure 1

done
