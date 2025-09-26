
import os
import numpy as np
import pandas as pd
import re

BLOSUM62 = {
    'C': {'C': 9, 'S': -1, 'T': -1, 'A': 0, 'G': -3, 'P': -3, 'D': -3, 'E': -4, 'Q': -3, 'N': -3, 'H': -3, 'R': -3, 'K': -3, 'M': -1, 'I': -1, 'L': -1, 'V': -1, 'W': -2, 'Y': -2, 'F': -2},
    'S': {'C': -1, 'S': 4, 'T': 1, 'A': 1, 'G': 0, 'P': -1, 'D': 0, 'E': 0, 'Q': 0, 'N': 1, 'H': -1, 'R': -1, 'K': 0, 'M': -1, 'I': -2, 'L': -2, 'V': -2, 'W': -3, 'Y': -2, 'F': -2},
    'T': {'C': -1, 'S': 1, 'T': 5, 'A': 0, 'G': -2, 'P': -1, 'D': -1, 'E': -1, 'Q': -1, 'N': 0, 'H': -2, 'R': -1, 'K': -1, 'M': -1, 'I': -1, 'L': -1, 'V': 0, 'W': -2, 'Y': -2, 'F': -2},
    'A': {'C': 0, 'S': 1, 'T': 0, 'A': 4, 'G': 0, 'P': -1, 'D': -2, 'E': -1, 'Q': -1, 'N': -2, 'H': -2, 'R': -1, 'K': -1, 'M': -1, 'I': -1, 'L': -1, 'V': 0, 'W': -3, 'Y': -2, 'F': -2},
    'G': {'C': -3, 'S': 0, 'T': -2, 'A': 0, 'G': 6, 'P': -2, 'D': -1, 'E': -2, 'Q': -2, 'N': 0, 'H': -2, 'R': -2, 'K': -2, 'M': -3, 'I': -4, 'L': -4, 'V': -3, 'W': -2, 'Y': -3, 'F': -3},
    'P': {'C': -3, 'S': -1, 'T': -1, 'A': -1, 'G': -2, 'P': 7, 'D': -1, 'E': -1, 'Q': -1, 'N': -1, 'H': -2, 'R': -2, 'K': -1, 'M': -2, 'I': -3, 'L': -3, 'V': -2, 'W': -4, 'Y': -3, 'F': -4},
    'D': {'C': -3, 'S': 0, 'T': -1, 'A': -2, 'G': -1, 'P': -1, 'D': 6, 'E': 2, 'Q': 0, 'N': 1, 'H': -1, 'R': -2, 'K': -1, 'M': -3, 'I': -3, 'L': -4, 'V': -3, 'W': -4, 'Y': -3, 'F': -3},
    'E': {'C': -4, 'S': 0, 'T': -1, 'A': -1, 'G': -2, 'P': -1, 'D': 2, 'E': 5, 'Q': 2, 'N': 0, 'H': 0, 'R': 0, 'K': 1, 'M': -2, 'I': -3, 'L': -3, 'V': -2, 'W': -3, 'Y': -2, 'F': -3},
    'Q': {'C': -3, 'S': 0, 'T': -1, 'A': -1, 'G': -2, 'P': -1, 'D': 0, 'E': 2, 'Q': 5, 'N': 0, 'H': 0, 'R': 1, 'K': 1, 'M': 0, 'I': -3, 'L': -2, 'V': -2, 'W': -2, 'Y': -1, 'F': -3},
    'N': {'C': -3, 'S': 1, 'T': 0, 'A': -2, 'G': 0, 'P': -2, 'D': 1, 'E': 0, 'Q': 0, 'N': 6, 'H': 1, 'R': 0, 'K': 0, 'M': -2, 'I': -3, 'L': -3, 'V': -3, 'W': -4, 'Y': -2, 'F': -3},
    'H': {'C': -3, 'S': -1, 'T': -2, 'A': -2, 'G': -2, 'P': -2, 'D': -1, 'E': 0, 'Q': 0, 'N': 1, 'H': 8, 'R': 0, 'K': -1, 'M': -2, 'I': -3, 'L': -3, 'V': -3, 'W': -2, 'Y': 2, 'F': -1},
    'R': {'C': -3, 'S': -1, 'T': -1, 'A': -1, 'G': -2, 'P': -2, 'D': -2, 'E': 0, 'Q': 1, 'N': 0, 'H': 0, 'R': 5, 'K': 2, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'W': -3, 'Y': -2, 'F': -3},
    'K': {'C': -3, 'S': 0, 'T': -1, 'A': -1, 'G': -2, 'P': -1, 'D': -1, 'E': 1, 'Q': 1, 'N': 0, 'H': -1, 'R': 2, 'K': 5, 'M': -1, 'I': -3, 'L': -2, 'V': -2, 'W': -3, 'Y': -2, 'F': -3},
    'M': {'C': -1, 'S': -1, 'T': -1, 'A': -1, 'G': -3, 'P': -2, 'D': -3, 'E': -2, 'Q': 0, 'N': -2, 'H': -2, 'R': -1, 'K': -1, 'M': 5, 'I': 1, 'L': 2, 'V': 1, 'W': -1, 'Y': -1, 'F': 0},
    'I': {'C': -1, 'S': -2, 'T': -1, 'A': -1, 'G': -4, 'P': -3, 'D': -3, 'E': -3, 'Q': -3, 'N': -3, 'H': -3, 'R': -3, 'K': -3, 'M': 1, 'I': 4, 'L': 2, 'V': 3, 'W': -3, 'Y': -1, 'F': 0},
    'L': {'C': -1, 'S': -2, 'T': -1, 'A': -1, 'G': -4, 'P': -3, 'D': -4, 'E': -3, 'Q': -2, 'N': -3, 'H': -3, 'R': -2, 'K': -2, 'M': 2, 'I': 2, 'L': 4, 'V': 1, 'W': -2, 'Y': -1, 'F': 0},
    'V': {'C': -1, 'S': -2, 'T': 0, 'A': 0, 'G': -3, 'P': -2, 'D': -3, 'E': -2, 'Q': -2, 'N': -3, 'H': -3, 'R': -3, 'K': -2, 'M': 1, 'I': 3, 'L': 1, 'V': 4, 'W': -3, 'Y': -1, 'F': -1},
    'W': {'C': -2, 'S': -3, 'T': -2, 'A': -3, 'G': -2, 'P': -4, 'D': -4, 'E': -3, 'Q': -2, 'N': -4, 'H': -2, 'R': -3, 'K': -3, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'W': 11, 'Y': 2, 'F': 1},
    'Y': {'C': -2, 'S': -2, 'T': -2, 'A': -2, 'G': -3, 'P': -3, 'D': -3, 'E': -2, 'Q': -1, 'N': -2, 'H': 2, 'R': -2, 'K': -2, 'M': -1, 'I': -1, 'L': -1, 'V': -1, 'W': 2, 'Y': 7, 'F': 3},
    'F': {'C': -2, 'S': -2, 'T': -2, 'A': -2, 'G': -3, 'P': -4, 'D': -3, 'E': -3, 'Q': -3, 'N': -3, 'H': -1, 'R': -3, 'K': -3, 'M': 0, 'I': 0, 'L': 0, 'V': -1, 'W': 1, 'Y': 3, 'F': 6}
}

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
    'hermes_py_000_ft_mega_thermompnn_ddg_st': '\HERMESPyFtMega',

    'hermes_py_000_ft_cdna117k_relaxed_pred': '\HERMESPyRelaxed',
    'hermes_py_050_ft_cdna117k_relaxed_pred': '\HERMESPyNoiseRelaxed',

    'hermes_py_050': '\HERMESPyNoise',
    'hermes_py_050_ft_ros_ddg_st': '\HERMESPyNoiseFtRos',
    'hermes_py_050_ft_cdna117k_ddg_st': '\HERMESPyNoiseFtCdna',
    'hermes_py_050_ft_mega_thermompnn_ddg_st': '\HERMESPyNoiseFtMega',

    'hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': '\HERMESPyFtRelaxedFtCdna',
    'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': '\HERMESPyNoiseFtRelaxedFtCdna',

    'thermompnn': '\Thermompnn'
}

def annotate_table(latex_str: str) -> str:
    """
    Annotate LaTeX table cells of the form n1/n2 according to the rules:

    - prepend \yellowl if n1 > n2 and n1 <= 3
    - prepend \greenl  if n1 < n2 and n1 > 3 and n1 <= 6
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


models = ['proteinmpnn_v_48_030', 'thermompnn', 'hermes_py_050', 'hermes_py_050_ft_cdna117k_relaxed_pred', 'hermes_py_050_ft_cdna117k_ddg_st', 'hermes_py_050_ft_mega_thermompnn_ddg_st']

## initialized csv file
final_df_dict = {'antigen name': [], 'stabilized name': [], 'mutation': [], 'B62 score': []}

for model in models:
    final_df_dict[HERMES_MODEL_TO_LATEX_NAME[model]] = []

for antigen_name in ['rsv_f', 'ha', 'hmpv_f', 'denv_e', 'sars_cov_2']:

    results_dir = f'{antigen_name}/results'

    # get dfs
    dfs = {}
    for model in models:
        dfs[model] = pd.read_csv(os.path.join(results_dir, f'{model}.csv'))
    
    ## sort the rows of each df stabily, first by mutation, then by label
    for model in models:
        dfs[model]['resnum'] = dfs[model]['mutation'].apply(lambda mut: int(mut[1:-1]))
        dfs[model] = dfs[model].sort_values(by=['resnum'], ascending=[True]).reset_index(drop=True)
        dfs[model] = dfs[model].drop('resnum', axis=1)
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
            final_df_dict['B62 score'].extend([BLOSUM62[mut[0]][mut[-1]] for mut in df['mutation'].tolist()])

        final_df_dict[HERMES_MODEL_TO_LATEX_NAME[model]].extend([f'{row["rank_mt"]}/{row["rank_wt"]}' for _, row in df.iterrows()])

for key in final_df_dict:
    print(key, len(final_df_dict[key]))

final_df = pd.DataFrame(final_df_dict)
final_df.to_csv('antigen_table.csv', index=False)
print(annotate_table(final_df.to_latex(index=False, escape=True)))

