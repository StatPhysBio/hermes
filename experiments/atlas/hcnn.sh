

model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'


use_mt_structure_list='0 1'

pdb_dir='/gscratch/spe/gvisan01/tcr_pmhc/pdbs/ATLAS/'

base_dir='./'

for model_version in $model_version_list
    do
    for use_mt_structure in $use_mt_structure_list
        do

        python -u ../../mutation_effect_prediction_with_hermes.py \
                            --model_version $model_version \
                            --csv_file $base_dir'ATLAS_cleaned.csv' \
                            --folder_with_pdbs $pdb_dir \
                            --output_dir $base_dir'results/' \
                            --dms_column '[ ddg ]' \
                            --wt_pdb_column wt_pdb \
                            --mt_pdb_column mt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column chain \
                            --mutant_split_symbol"=|" \
                            --use_mt_structure $use_mt_structure
        
        python -u correlations.py \
                    --model_version $model_version \
                    --use_mt_structure $use_mt_structure
        
    done
done

