
import os, sys
import json
import numpy as np
import pandas as pd


sys.path.append('..')
from get_full_table import METADATA_COLUMNS, HERMES_MODELS, PROTEINMPNN_MODELS, ESM_MODELS


system_name = 'T4_mutant_ddG_standardized'

df = None

for use_mt_structure in [0, 1]:

    correlations_values_in_table = {}

    num_measurements_trace = []

    for model in HERMES_MODELS + PROTEINMPNN_MODELS + ESM_MODELS:

        try:

            if model in HERMES_MODELS + ESM_MODELS:
                with open(f'results/{model}/{system_name}-{model}-use_mt_structure={use_mt_structure}_correlations.json', 'r') as f:
                    correlations = json.load(f)
            elif model in PROTEINMPNN_MODELS:
                with open(f'results/{model}/{system_name}-num_seq_per_target=10-use_mt_structure={use_mt_structure}_correlations.json', 'r') as f:
                    correlations = json.load(f)
            
            pr = -correlations['2LZM']['pearson'][0] # flip correlation so higher is better
            pr_pval = correlations['2LZM']['pearson'][1] # flip correlation so higher is better

            sr = -correlations['2LZM']['spearman'][0]
            sr_pval = correlations['2LZM']['spearman'][1]

            num_measurements = correlations['2LZM']['count']
            num_measurements_trace.append(num_measurements)

            correlations_values_in_table[model + ' - Pearsonr'] = pr
            correlations_values_in_table[model + ' - Pearsonr p-value'] = pr_pval
            correlations_values_in_table[model + ' - Spearmanr'] = sr
            correlations_values_in_table[model + ' - Spearmanr p-value'] = sr_pval
        
        except:

            correlations_values_in_table[model + ' - Pearsonr'] = np.nan
            correlations_values_in_table[model + ' Pearsonr p-value'] = np.nan
            correlations_values_in_table[model + ' Spearmanr'] = np.nan
            correlations_values_in_table[model + ' Spearmanr p-value'] = np.nan

    if len(set(num_measurements_trace)) > 1:
        print('WARNINGL Number of measurements for each model is not the same')

    metadata_values = ['T4 Lysozyme', bool(use_mt_structure), '-ddG stability', num_measurements_trace[-1], 1, True, False, 'per structure']

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





    












