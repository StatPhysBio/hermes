

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/

base_dir='/gscratch/stf/gvisan01/hermes/experiments/antigens/ha/'

model_version_list='v_48_030 v_48_002'

for model_version in $model_version_list
    do

    python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
                    --csv_file $base_dir'mutations_all.csv' \
                    --folder_with_pdbs $base_dir'pdbs/' \
                    --output_dir $base_dir'results/proteinmpnn_'$model_version \
                    --use_mt_structure 0 \
                    --model_name $model_version \
                    --num_seq_per_target 10 \
                    --batch_size 10 \
                    --wt_pdb_column pdb \
                    --mutant_column mutation \
                    --mutant_chain_column chain

done


