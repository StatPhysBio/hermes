
model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'

systems='output_KKA2_KLEPN_Mikkelsen2014__1nd4 output_KKA2_KLEPN_Mikkelsen2014__AF-P00552-F1-model_v4'

dms_columns='Ami11_avg Ami12_avg Ami14_avg Ami18_avg G41811_avg G41812_avg G41814_avg Kan11_avg Kan12_avg Kan14_avg Kan18_avg Neo11_avg Neo12_avg Neo14_avg Neo18_avg Paro11_avg Paro12_avg Paro14_avg Paro18_avg Ribo11_avg Ribo12_avg Ribo14_avg Ribo18_avg'

pdb_dir='./pdbs/'

base_dir='./'

for system in $systems
    do
    for model_version in $model_version_list
        do

        python -u ../../../mutation_effect_prediction_with_hermes.py \
                            --model_version $model_version \
                            --csv_file $base_dir$system'.csv' \
                            --folder_with_pdbs $pdb_dir \
                            --output_dir $base_dir'results/' \
                            --wt_pdb_column wt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column chain \
                            --use_mt_structure 0 \
                            --dms_column $dms_columns


        python -u ../correlations.py \
                        --model_version $model_version \
                        --use_mt_structure 0 \
                        --system_name $system \
                        --dms_column $dms_columns
    
    done
done