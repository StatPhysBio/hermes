

model_version_list='hermes_py_050' # 'hermes_py_050 hermes_py_000'


base_dir='./'
output_dir=$base_dir'results_relaxed_nrep1_ens1_nowt_side12_bb0/'

for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../concat_split_dfs.py \
                        --model_version $model_version \
                        --csv_file $base_dir'test_targets.csv' \
                        --output_dir $output_dir \

done

