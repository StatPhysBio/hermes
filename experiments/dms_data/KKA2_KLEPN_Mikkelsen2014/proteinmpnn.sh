

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/
pdb_dir='./pdbs/'
base_dir='./'

model_version_list='v_48_030 v_48_020 v_48_002'

systems='output_KKA2_KLEPN_Mikkelsen2014__1nd4 output_KKA2_KLEPN_Mikkelsen2014__AF-P00552-F1-model_v4'

dms_columns='Ami11_avg Ami12_avg Ami14_avg Ami18_avg G41811_avg G41812_avg G41814_avg Kan11_avg Kan12_avg Kan14_avg Kan18_avg Neo11_avg Neo12_avg Neo14_avg Neo18_avg Paro11_avg Paro12_avg Paro14_avg Paro18_avg Ribo11_avg Ribo12_avg Ribo14_avg Ribo18_avg'

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


