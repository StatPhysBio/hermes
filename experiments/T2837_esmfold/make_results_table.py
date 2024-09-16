
import os, sys
import json
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues


sys.path.append('..')
from get_full_table import METADATA_COLUMNS, HERMES_MODELS, PROTEINMPNN_MODELS, ESM_MODELS


system_name = 'T2837_esmfold_ddg_experimental'

df = None

for dataset in ['myoglobin', 'p53', 's669', 'ssym', 't2226', 'overall']:

    correlations_values_in_table = {}

    num_measurements_trace = []

    for model in HERMES_MODELS + PROTEINMPNN_MODELS + ESM_MODELS:

        try:
            if model in HERMES_MODELS + ESM_MODELS:
                with open(f'results/{model}/{system_name}-{model}-use_mt_structure=0_correlations.json', 'r') as f:
                    correlations = json.load(f)
            elif model in PROTEINMPNN_MODELS:
                with open(f'results/{model}/{system_name}-num_seq_per_target=10-use_mt_structure=0_correlations.json', 'r') as f:
                    correlations = json.load(f)
        except:
            print(f'WARNING: {model} not found for {dataset}')
            continue
        
        pr = -correlations[dataset]['pearson'][0] # flip correlation so higher is better
        pr_std_error = np.nan
        pr_pval = correlations[dataset]['pearson'][1]
        sr = -correlations[dataset]['spearman'][0] # flip correlation so higher is better
        sr_std_error = np.nan
        sr_pval = correlations[dataset]['spearman'][1]
        num_measurements = correlations[dataset]['count']
        num_structures = np.nan

        num_measurements_trace.append(num_measurements)

        correlations_values_in_table[model + ' - Pearsonr'] = pr
        correlations_values_in_table[model + ' - Pearsonr std error'] = pr_std_error
        correlations_values_in_table[model + ' - Pearsonr p-value'] = pr_pval
        correlations_values_in_table[model + ' - Spearmanr'] = sr
        correlations_values_in_table[model + ' - Spearmanr std error'] = sr_std_error
        correlations_values_in_table[model + ' - Spearmanr p-value'] = sr_pval

        correlations_values_in_table[model + ' - precision'] = correlations[dataset]['precision']
        correlations_values_in_table[model + ' - recall'] = correlations[dataset]['recall']
        correlations_values_in_table[model + ' - auroc'] = correlations[dataset]['auroc']
        correlations_values_in_table[model + ' - accuracy'] = correlations[dataset]['accuracy']

    if len(set(num_measurements_trace)) > 1:
        print('WARNING: Number of measurements for each model is not the same')
        print(num_measurements_trace)

    metadata_values = [f'T2837_esmfold - {dataset}', False, '-ddG stability', num_measurements_trace[-1], num_structures, True, False, 'overall']

    metatadata_in_table = dict(zip(METADATA_COLUMNS, metadata_values))

    if df is None:
        df = pd.DataFrame(metatadata_in_table, index=[0])
        df = pd.concat([df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)
    else:
        curr_df = pd.DataFrame(metatadata_in_table, index=[0])
        curr_df = pd.concat([curr_df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)
        df = pd.concat([df, curr_df], axis=0)


print(df)
df.to_csv(f'{system_name}-results_table.csv', index=False)





    












