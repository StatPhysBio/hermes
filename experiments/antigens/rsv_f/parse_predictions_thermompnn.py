
import os
import numpy as np
import pandas as pd
import json

AMINOACIDS = 'ACDEFGHIKLMNPQRSTVWY'

idx_to_resnum = np.arange(26, 514)

def get_percentile_bucket_str(x, perc_list, value_list):
    # If below min
    if x <= value_list[0]:
        return f"≤ {perc_list[0]}th"
    # If above max
    if x >= value_list[-1]:
        return f"≥ {perc_list[-1]}th"
    
    # Find index where it belongs
    idx = np.searchsorted(value_list, x) - 1
    p0, p1 = perc_list[idx], perc_list[idx + 1]
    
    # Return bucket string
    return f"{p0}th - {p1}th"

if __name__ == '__main__':

    df = pd.read_csv('mutations.csv')
    df_preds = pd.read_csv('results_all_sites/ThermoMPNN_inference_4jhw_trimer.csv')

    perc_list = [50, 60, 70, 75, 80, 85, 90, 95, 97.5, 99]
    percentile_file = f'/gscratch/stf/gvisan01/hermes/experiments/average_prediction_matrices/T2837_all_sites/all/distributions/thermompnn__percentiles.json'
    with open(percentile_file, 'r') as f:
        percentile_data = json.load(f)

    df_preds['resnum'] = df_preds['position'].map(lambda x: idx_to_resnum[x])
    df_preds.rename(columns={'mutation': 'mutant'}, inplace=True)
    df_preds['mutation'] = df_preds['wildtype'] + df_preds['resnum'].map(str) + df_preds['mutant']

    data_to_add = {'rank_wt': [], 'rank_mt': [], 'delta_logit': [], 'perc_bucket': []}
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

        aas = np.array(aas)
        ddgs = np.array(ddgs)
        delta_logit = -(ddgs[np.where(aas == mt)[0][0]] - ddgs[np.where(aas == wt)[0][0]]) # higher is better

        # compute percentile bucket
        perc_value_list = [percentile_data[wt][mt][str(perc)] for perc in perc_list]
        perc_bucket = get_percentile_bucket_str(delta_logit, perc_list, perc_value_list)

        data_to_add['rank_wt'].append(rank_wt)
        data_to_add['rank_mt'].append(rank_mt)
        data_to_add['delta_logit'].append(delta_logit)
        data_to_add['perc_bucket'].append(perc_bucket)


    
    for key in data_to_add:
        df[key] = data_to_add[key]
    
    os.makedirs('./results', exist_ok=True)
    df.to_csv('results/thermompnn.csv', index=False)

    
