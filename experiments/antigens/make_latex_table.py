
import os
import numpy as np
import pandas as pd
import re

HERMES_MODEL_TO_LATEX_NAME = {
    'proteinmpnn_v_48_002': '\proteinmpnn',
    'proteinmpnn_v_48_030': '\proteinmpnnNoise',

    'hermes_bp_000': '\HERMESBp',
    'hermes_bp_000_ft_ros_ddg_st': '\HERMESBpFtRos',
    'hermes_bp_000_ft_cdna117k_ddg_st': '\HERMESBpFtCdna',

    'hermes_bp_050': '\HERMESBpNoise',
    'hermes_bp_050_ft_ros_ddg_st': '\HERMESBpNoiseFtRos',
    'hermes_bp_050_ft_cdna117k_ddg_st': '\HERMESBpNoiseFtCdna',

    'hermes_py_000': '\HERMESPy',
    'hermes_py_000_ft_ros_ddg_st': '\HERMESPyFtRos',
    'hermes_py_000_ft_cdna117k_ddg_st': '\HERMESPyFtCdna',

    'hermes_py_000_ft_cdna117k_relaxed_pred': '\HERMESPyRelaxed',
    'hermes_py_050_ft_cdna117k_relaxed_pred': '\HERMESPyNoiseRelaxed',

    'hermes_py_050': '\HERMESPyNoise',
    'hermes_py_050_ft_ros_ddg_st': '\HERMESPyNoiseFtRos',
    'hermes_py_050_ft_cdna117k_ddg_st': '\HERMESPyNoiseFtCdna',

    'hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': '\HERMESPyFtRelaxedFtCdna',
    'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': '\HERMESPyNoiseFtRelaxedFtCdna',
}

def annotate_table(latex_str: str) -> str:
    """
    Annotate LaTeX table cells of the form n1/n2 according to the rules:

    - prepend \yellowl if n1 > n2 and n1 <= 3
    - prepend \greenl  if n1 < n2 and n1 > 3
    - prepend \green   if n1 < n2 and n1 <= 3
    - leave unchanged otherwise
    """

    def replacer(match):
        n1, n2 = int(match.group(1)), int(match.group(2))
        if n1 > n2 and n1 <= 3:
            return r"\yellowl " + match.group(0)
        elif n1 < n2 and (n1 > 3 and n1 <= 6):
            return r"\greenl " + match.group(0)
        elif n1 < n2 and n1 <= 3:
            return r"\green " + match.group(0)
        else:
            return match.group(0)

    return re.sub(r"\b(\d+)/(\d+)\b", replacer, latex_str)


models = ['proteinmpnn_v_48_030', 'hermes_py_050', 'hermes_py_050_ft_cdna117k_relaxed_pred', 'hermes_py_050_ft_cdna117k_ddg_st', 'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st']

## initialized csv file
final_df_dict = {'antigen name': [], 'stabilized name': [], 'mutation': []}
for model in models:
    final_df_dict[HERMES_MODEL_TO_LATEX_NAME[model]] = []

for antigen_name in ['rsv_f', 'ha', 'hmpv_f', 'denv_e']:

    results_dir = f'{antigen_name}/results'

    # get dfs
    dfs = {}
    for model in models:
        dfs[model] = pd.read_csv(os.path.join(results_dir, f'{model}.csv'))
    
    ## sort the rows of each df stabily, first by mutation, then by label
    for model in models:
        dfs[model] = dfs[model].sort_values(by=['mutation'], ascending=[True]).reset_index(drop=True)
        dfs[model] = dfs[model].sort_values(by=['label'], ascending=[True]).reset_index(drop=True)
        print()
        print(dfs[model])
        print()

    for i, model in enumerate(models):

        # get df for this model
        df = dfs[model]

        if i == 0: # the things we want to add only once
            final_df_dict['antigen name'].extend([antigen_name] * len(df))
            final_df_dict['stabilized name'].extend(df['label'].tolist())
            final_df_dict['mutation'].extend(df['mutation'].tolist())

        final_df_dict[HERMES_MODEL_TO_LATEX_NAME[model]].extend([f'{row["rank_mt"]}/{row["rank_wt"]}' for _, row in df.iterrows()])

for key in final_df_dict:
    print(key, len(final_df_dict[key]))

final_df = pd.DataFrame(final_df_dict)
final_df.to_csv('antigen_table.csv', index=False)
print(annotate_table(final_df.to_latex(index=False, escape=True)))

