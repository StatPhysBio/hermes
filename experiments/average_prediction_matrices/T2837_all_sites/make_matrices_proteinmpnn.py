
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

AMINOACIDS = 'GPCAVILMFYWSTNQRHKDE'

model_version_list = ['proteinmpnn_v_48_002',
                      'proteinmpnn_v_48_030']

os.makedirs('all/matrices', exist_ok=True)

for model_version in model_version_list:

    df = pd.read_csv(f'/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/proteinmpnn_output/{model_version}/zero_shot_predictions/proteinmpnn_input-num_seq_per_target=10-use_mt_structure=0.csv')

    matrix_num = np.zeros((20, 20))
    matrix_denom = np.zeros((20, 20))

    for i, row in tqdm(df.iterrows(), total=len(df)):

        aa_wt = row['mutant'][0]
        aa_mt = row['mutant'][-1]
        score = row['log_p_mt__minus__log_p_wt']
        # print('-'+aa_wt+'-', aa_mt, score)

        if not np.isnan(score):
            matrix_num[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += score
            matrix_denom[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += 1

    matrix = matrix_num / matrix_denom

    matrix_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
    matrix_df.to_csv(f'all/matrices/{model_version}.csv')


    ## make them split by surface vs core based on SASA

    info_df = pd.read_csv('/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/structural_info.csv')

    df['resnum'] = df['mutant'].apply(lambda x: int(x[1:-1]))
    df = pd.merge(df, info_df, on=['pdb', 'chain', 'resnum'])
    
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

        for i, row in tqdm(curr_df.iterrows(), total=len(curr_df)):

            aa_wt = row['mutant'][0]
            aa_mt = row['mutant'][-1]
            score = row['log_p_mt__minus__log_p_wt']

            if not np.isnan(score):
                matrix_num[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += score
                matrix_denom[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += 1

        matrix = matrix_num / matrix_denom

        matrix_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
        matrix_df.to_csv(f'{region}/matrices/{model_version}.csv')


