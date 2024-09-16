
import os
import numpy as np
import pandas as pd

models = 'esm_1v_wt_marginals proteinmpnn_v_48_002 proteinmpnn_v_48_020 proteinmpnn_v_48_030 hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st hermes_py_000_untrained_small_ft_cdna117k_ddg_st hermes_py_050_untrained_small_ft_cdna117k_ddg_st'.split()

r2_scores = []
for model in models:
    with open(f'ssym_antisymmetry_score_{model}-use_mt_structure=0.txt', 'r') as f:
        antisymmetry_score = float(f.read())
    r2_scores.append(antisymmetry_score)

df = pd.DataFrame({'model': models, 'r2_score_antisymmetry': r2_scores})
df.to_csv('ssym_antisymmetry_scores.csv', index=False)
