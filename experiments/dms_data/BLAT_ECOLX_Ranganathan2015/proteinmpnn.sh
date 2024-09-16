

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/
pdb_dir='./pdbs/'
base_dir='./'

model_version_list='v_48_030 v_48_020 v_48_002'

systems='output_BLAT_ECOLX_Ranganathan2015__AF-P62593-F1-model_v4'

dms_columns='0 39 156 625 2500 cefotaxime 0_1 0_2 10_1 39_1 39_2 156_1 156_2 625_1 625_2 2500_1 2500_2 km vmax'

for system in $systems
    do
    for model_version in $model_version_list
        do

        python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
                        --csv_file $base_dir$system'.csv' \
                        --folder_with_pdbs $pdb_dir \
                        --output_dir $base_dir'proteinmpnn_'$model_version \
                        --use_mt_structure 0 \
                        --model_name $model_version \
                        --num_seq_per_target 10 \
                        --batch_size 10 \
                        --wt_pdb_column wt_pdb \
                        --mutant_column mutant \
                        --mutant_chain_column chain \
        
        python -u ../correlations.py \
                        --model_version proteinmpnn_$model_version \
                        --use_mt_structure 0 \
                        --system_name $system \
                        --dms_column $dms_columns
    
    done
done


