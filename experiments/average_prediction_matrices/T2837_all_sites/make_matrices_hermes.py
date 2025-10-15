
import os
import numpy as np
import pandas as pd

AMINOACIDS = 'GPCAVILMFYWSTNQRHKDE'

model_version_list = ['hermes_py_000',
                      'hermes_py_050',
                      'hermes_py_000_ft_cdna117k_relaxed_pred',
                      'hermes_py_050_ft_cdna117k_relaxed_pred',
                      'hermes_py_000_ft_cdna117k_ddg_st',
                      'hermes_py_050_ft_cdna117k_ddg_st',
                      'hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st',
                      'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st',
                      'hermes_py_000_ft_mega_thermompnn_ddg_st',
                      'hermes_py_050_ft_mega_thermompnn_ddg_st',
                      ]

os.makedirs('all/matrices', exist_ok=True)

for model_version in model_version_list:

    df = pd.read_csv(f'/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/csv_files/{model_version}.csv')

    matrix_num = np.zeros((20, 20))
    matrix_denom = np.zeros((20, 20))

    for i, row in df.iterrows():

        aa_wt = row['resname']

        for aa_mt in AMINOACIDS:

            score = row[f'logit_{aa_mt}'] - row[f'logit_{aa_wt}']

            matrix_num[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += score
            matrix_denom[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += 1

    matrix = matrix_num / matrix_denom

    matrix_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
    matrix_df.to_csv(f'all/matrices/{model_version}.csv')


    ## make them split by surface vs core based on SASA

    info_df = pd.read_csv('/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/structural_info.csv')

    df = pd.merge(df, info_df, on=['pdb', 'chain', 'resname', 'resnum', 'insertion_code'])
    
    sasa = df['sasa'].values

    masks = {}
    masks['core'] = sasa <= 1
    masks['in_between'] = np.logical_and(sasa > 1, sasa < 3)
    masks['surface'] = sasa >= 3

    for region in masks:

        os.makedirs(f'{region}/matrices', exist_ok=True)

        curr_df = df[masks[region]]

        matrix_num = np.zeros((20, 20))
        matrix_denom = np.zeros((20, 20))

        for i, row in curr_df.iterrows():

            aa_wt = row['resname']

            for aa_mt in AMINOACIDS:

                score = row[f'logit_{aa_mt}'] - row[f'logit_{aa_wt}']

                matrix_num[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += score
                matrix_denom[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += 1

        matrix = matrix_num / matrix_denom

        matrix_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
        matrix_df.to_csv(f'{region}/matrices/{model_version}.csv')


