

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/

base_dir='/gscratch/spe/gvisan01/protein_holography-web/experiments/skempi/'

model_version_list='v_48_030 v_48_020 v_48_002'

for model_version in $model_version_list
    do

    # python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
    #                 --csv_file $base_dir'skempi_v2_cleaned_NO_1KBH.csv' \
    #                 --folder_with_pdbs /gscratch/stf/gvisan01/skempi/pyrosetta_mutated_pdbs \
    #                 --output_dir $base_dir'proteinmpnn_'$model_version \
    #                 --use_mt_structure 0 \
    #                 --model_name $model_version \
    #                 --num_seq_per_target 10 \
    #                 --batch_size 10 \
    #                 --wt_pdb_column PDB_filename \
    #                 --mt_pdb_column PDB_filename_pyrosetta_mutant \
    #                 --mutant_column mutant \
    #                 --mutant_chain_column mutant_chain \
    #                 --mutant_split_symbol"=|" \
    
    python -u correlations.py \
                        --model_version 'proteinmpnn_'$model_version \
                        --use_mt_structure 0

done


