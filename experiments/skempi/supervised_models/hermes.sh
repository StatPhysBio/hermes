

model_version_list='hermes_bp_000_ft_skempi hermes_bp_050_ft_skempi hermes_py_000_ft_skempi hermes_py_050_ft_skempi'


use_mt_structure='0'

base_dir='./'

datasets='easy medium hard'

folds='fold1 fold2 fold3'

for model_version in $model_version_list
    do

    for dataset in $datasets
        do
        # for fold in $folds
        #     do

        #     python -u ../../../mutation_effect_prediction_with_hermes.py \
        #                         --model_version $model_version'_'$dataset'_'$fold'_ddg_bi' \
        #                         --csv_file $base_dir'/'$dataset'/'$fold'/test_targets.csv' \
        #                         --folder_with_pdbs /gscratch/stf/gvisan01/skempi/pyrosetta_mutated_pdbs \
        #                         --output_dir $base_dir'/'$dataset'/'$fold'/results' \
        #                         --wt_pdb_column pdbid \
        #                         --mt_pdb_column PDB_filename_pyrosetta_mutant \
        #                         --mutant_column variant \
        #                         --mutant_chain_column chainid \
        #                         --mutant_split_symbol"=|" \
        #                         --use_mt_structure $use_mt_structure

        # done

        python -u correlations.py \
                            --dataset $dataset \
                            --model_version_base $model_version \
                            --use_mt_structure $use_mt_structure
        
    done
done

