

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/

base_dir='/gscratch/spe/gvisan01/hermes/experiments/megascale/'

model_version_list='v_48_030 v_48_020 v_48_002'

for model_version in $model_version_list
    do

    # python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
    #                 --csv_file $base_dir'test_targets.csv' \
    #                 --folder_with_pdbs /gscratch/spe/gvisan01/hermes/training_data/finetuning/mega_thermompnn_neg/pdbs \
    #                 --output_dir $base_dir'results/proteinmpnn_'$model_version \
    #                 --use_mt_structure 0 \
    #                 --model_name $model_version \
    #                 --num_seq_per_target 10 \
    #                 --batch_size 10 \
    #                 --wt_pdb_column pdbid \
    #                 --mutant_column variant \
    #                 --mutant_chain_column chainid

    python -u correlations.py \
                        --model_version proteinmpnn_$model_version
    
    # python -u correlations_stratified.py \
    #                     --model_version proteinmpnn_$model_version \
    #                     --use_mt_structure 0 \
    #                     --system_name T2837

done


