
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

AMINOACIDS = 'GPCAVILMFYWSTNQRHKDE'

os.makedirs('all/matrices', exist_ok=True)

csv_dir = '/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/thermompnn'

matrix_num = np.zeros((20, 20))
matrix_denom = np.zeros((20, 20))

for csv_file in os.listdir(csv_dir):
    df = pd.read_csv(os.path.join(csv_dir, csv_file))

    for i, row in df.iterrows():
        aa_wt = row['wildtype']
        aa_mt = row['mutation']
        score = -row['ddG_pred']
        matrix_num[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += score
        matrix_denom[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += 1

matrix = matrix_num / matrix_denom

matrix_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
matrix_df.to_csv(f'all/matrices/thermompnn.csv')


## make them split by surface vs core based on SASA

info_df = pd.read_csv('/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/structural_info.csv')

for region in ['core', 'in_between', 'surface']:

    os.makedirs(f'{region}/matrices', exist_ok=True)

    matrix_num = np.zeros((20, 20))
    matrix_denom = np.zeros((20, 20))

    for csv_file in os.listdir(csv_dir):
        df = pd.read_csv(os.path.join(csv_dir, csv_file))

        # min_resnum is the smallest residue in info_df with the same pdb as in df
        info_ = info_df[info_df['pdb'] == df['pdb'].values[0]]
        min_resnum = info_['resnum'].min()
        df['resnum'] = df['position'] + min_resnum
        df = pd.merge(df, info_df, on=['pdb', 'chain', 'resnum'])
        
        sasa = df['sasa'].values

        masks = {}
        masks['core'] = sasa <= 1
        masks['in_between'] = np.logical_and(sasa > 1, sasa < 3)
        masks['surface'] = sasa >= 3

        curr_df = df[masks[region]]

        for i, row in curr_df.iterrows():
            aa_wt = row['wildtype']
            aa_mt = row['mutation']
            score = -row['ddG_pred']
            matrix_num[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += score
            matrix_denom[AMINOACIDS.index(aa_wt), AMINOACIDS.index(aa_mt)] += 1

    matrix = matrix_num / matrix_denom

    matrix_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
    matrix_df.to_csv(f'{region}/matrices/thermompnn.csv')



