

model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st hermes_py_000_ft_cdna117k_relaxed_pred hermes_py_050_ft_cdna117k_relaxed_pred hermes_bp_000_ft_cdna117k_relaxed_pred hermes_bp_050_ft_cdna117k_relaxed_pred'

use_mt_structure='0'

base_dir='./'

for model_version in $model_version_list
    do

    # python -u ../../mutation_effect_prediction_with_hermes.py \
    #                     --model_version $model_version \
    #                     --csv_file $base_dir'skempi_v2_cleaned_NO_1KBH.csv' \
    #                     --folder_with_pdbs /gscratch/stf/gvisan01/skempi/pyrosetta_mutated_pdbs \
    #                     --output_dir $base_dir'results/' \
    #                     --wt_pdb_column PDB_filename \
    #                     --mt_pdb_column PDB_filename_pyrosetta_mutant \
    #                     --mutant_column mutant \
    #                     --mutant_chain_column mutant_chain \
    #                     --mutant_split_symbol"=|" \
    #                     --use_mt_structure $use_mt_structure
    
    python -u correlations.py \
                        --model_version $model_version \
                        --use_mt_structure $use_mt_structure

done

