
import os
import numpy as np
import pandas as pd

AMINOACIDS = 'ACDEFGHIKLMNPQRSTVWY'

idx_to_resnum = np.arange(9, 502)


if __name__ == '__main__':

    df = pd.read_csv('mutations.csv')
    df_preds = pd.read_csv('results_all_sites/ThermoMPNN_inference_7VDF.csv')

    df_preds['resnum'] = df_preds['position'].map(lambda x: idx_to_resnum[x])
    df_preds.rename(columns={'mutation': 'mutant'}, inplace=True)
    df_preds['mutation'] = df_preds['wildtype'] + df_preds['resnum'].map(str) + df_preds['mutant']

    data_to_add = {'rank_wt': [], 'rank_mt': []}
    for aa in AMINOACIDS:
        data_to_add[f'ddG_pred_{aa}'] = []

    for mutation in df['mutation']:
        resnum = int(mutation[1:-1])
        wt = mutation[0]
        mt = mutation[-1]
        df_subset = df_preds[df_preds['resnum'] == resnum]

        aas = []
        ddgs = []
        for i, row in df_subset.iterrows():
            data_to_add[f'ddG_pred_{row["mutant"]}'].append(row['ddG_pred'])
            aas.append(row['mutant'])
            ddgs.append(row['ddG_pred'])
        rank_wt = (np.array(ddgs) < df_subset[df_subset['mutant'] == wt]['ddG_pred'].values[0]).sum() + 1
        rank_mt = (np.array(ddgs) < df_subset[df_subset['mutant'] == mt]['ddG_pred'].values[0]).sum() + 1
        data_to_add['rank_wt'].append(rank_wt)
        data_to_add['rank_mt'].append(rank_mt)
    
    for key in data_to_add:
        df[key] = data_to_add[key]
    
    df.to_csv('results/thermompnn.csv', index=False)

    
