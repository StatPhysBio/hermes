
model_version_list='esm_1v_wt_marginals proteinmpnn_v_48_002 proteinmpnn_v_48_020 proteinmpnn_v_48_030 hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st hermes_py_000_untrained_small_ft_cdna117k_ddg_st hermes_py_050_untrained_small_ft_cdna117k_ddg_st'


use_mt_structure='0'

for model_version in $model_version_list
    do

    python -u compute_antisymmetry_score.py \
                --model_version $model_version \
                --use_mt_structure $use_mt_structure

done

cd antisymmetry_scores
python -u pool_scores.py
cd ..