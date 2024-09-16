

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/

base_dir='/gscratch/spe/gvisan01/protein_holography-web/experiments/t4_lysozyme/'

model_version_list='v_48_030 v_48_020 v_48_002'

use_mt_structure_list='0 1'

for use_mt_structure in $use_mt_structure_list
    do

    for model_version in $model_version_list
        do

        python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
                        --csv_file $base_dir'T4_mutant_ddG_standardized.csv' \
                        --folder_with_pdbs $base_dir'pdbs/' \
                        --output_dir $base_dir'proteinmpnn_'$model_version \
                        --use_mt_structure 0 \
                        --model_name $model_version \
                        --num_seq_per_target 10 \
                        --batch_size 10 \
                        --dms_column ddG \
                        --dms_label None \
                        --wt_pdb_column wt_pdb \
                        --mt_pdb_column mt_pdb \
                        --mutant_column mutant \
                        --mutant_chain_column mutant_chain \
                        --use_mt_structure $use_mt_structure

        python -u correlations.py \
                            --model_version 'proteinmpnn_'$model_version \
                            --use_mt_structure $use_mt_structure

    done

done

