
model_version_list='hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st'

systems='output_DLG4_RAT_Ranganathan2012__1qt3 output_DLG4_RAT_Ranganathan2012__5hdy'

dms_columns='CRIPT Tm2F'

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