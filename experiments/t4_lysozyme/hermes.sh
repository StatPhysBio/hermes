

model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'


use_mt_structure_list='0 1'

base_dir='./'

for model_version in $model_version_list
    do
    for use_mt_structure in $use_mt_structure_list
        do

        python -u ../../mutation_effect_prediction_with_hermes.py \
                            --model_version $model_version \
                            --csv_file $base_dir'T4_mutant_ddG_standardized.csv' \
                            --folder_with_pdbs $base_dir'pdbs' \
                            --output_dir $base_dir'results/' \
                            --dms_column ddG \
                            --wt_pdb_column wt_pdb \
                            --mt_pdb_column mt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column mutant_chain \
                            --use_mt_structure $use_mt_structure
        
        python -u correlations.py \
                            --model_version $model_version \
                            --use_mt_structure $use_mt_structure
    
    done
done