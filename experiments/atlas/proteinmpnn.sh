

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/

base_dir='/gscratch/spe/gvisan01/protein_holography-web/experiments/atlas/'

pdb_dir='/gscratch/spe/gvisan01/tcr_pmhc/pdbs/ATLAS/'

model_version_list='v_48_030 v_48_020 v_48_002'

use_mt_structure_list='0 1'

for model_version in $model_version_list
    do
    for use_mt_structure in $use_mt_structure_list
        do

        # python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
        #                 --csv_file $base_dir'ATLAS_cleaned.csv' \
        #                 --folder_with_pdbs $pdb_dir \
        #                 --output_dir $base_dir'proteinmpnn_'$model_version \
        #                 --use_mt_structure $use_mt_structure \
        #                 --model_name $model_version \
        #                 --num_seq_per_target 10 \
        #                 --batch_size 10 \
        #                 --wt_pdb_column wt_pdb \
        #                 --mt_pdb_column mt_pdb \
        #                 --mutant_column mutant \
        #                 --mutant_chain_column chain \
        #                 --mutant_split_symbol"=|"
        
        python -u correlations.py \
                            --model_version 'proteinmpnn_'$model_version \
                            --use_mt_structure $use_mt_structure
    done
done


