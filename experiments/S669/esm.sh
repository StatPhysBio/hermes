

model_version_list='esm_1v_masked_marginals esm_1v_wt_marginals'


use_mt_structure='0'

base_dir='./'
output_dir=$base_dir

for model_version in $model_version_list
    do
    
    python -u correlations.py \
                        --model_version $model_version \
                        --use_mt_structure $use_mt_structure \
                        --system_name s669

done

