


# model_version_list='hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'
# model_version_list='hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st'
model_version_list='hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st'

parallelism=8

results_dir='./results_all_sites/csv_files/'
mkdir -p $results_dir

for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../run_hermes_on_pdbfiles.py \
                    -m $model_version \
                    -r logits \
                    -pd ./pdbs \
                    -o $results_dir$model_version'.csv' \
                    -pp $parallelism

done

