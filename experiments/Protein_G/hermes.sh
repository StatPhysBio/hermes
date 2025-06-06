


model_version_list='hermes_bp_000 hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_ros_ddg_st hermes_py_050 hermes_py_050_ft_ros_ddg_st'


use_mt_structure='0'

base_dir='./'
output_dir=$base_dir'results/'

for model_version in $model_version_list
    do

    python -u ../../mutation_effect_prediction_with_hermes.py \
                        --model_version $model_version \
                        --csv_file $base_dir'protein_g_ddg_experimental.csv' \
                        --folder_with_pdbs $base_dir'pdbs/' \
                        --output_dir $output_dir \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure $use_mt_structure \
                        --dms_column score
    
    python -u correlations.py \
                        --model_version $model_version \
                        --use_mt_structure $use_mt_structure \
                        --system_name protein_g

done

