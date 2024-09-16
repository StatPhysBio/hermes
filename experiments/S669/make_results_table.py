
import os, sys
import json
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues


sys.path.append('..')
from get_full_table import METADATA_COLUMNS, HERMES_MODELS, PROTEINMPNN_MODELS, ESM_MODELS


system_name = 's669_ddg_experimental'

df = None

for curr_comp in ['per structure', 'overall']:

    correlations_values_in_table = {}

    num_measurements_trace = []

    for model in HERMES_MODELS + PROTEINMPNN_MODELS + ESM_MODELS:

        if model in HERMES_MODELS + ESM_MODELS:
            with open(f'results/{model}/{system_name}-{model}-use_mt_structure=0_correlations.json', 'r') as f:
                correlations = json.load(f)
        elif model in PROTEINMPNN_MODELS:
            with open(f'results/{model}/{system_name}-num_seq_per_target=10-use_mt_structure=0_correlations.json', 'r') as f:
                correlations = json.load(f)
        
        if curr_comp == 'per structure':
            pr_trace, pr_pval_trace, sr_trace, sr_pval_trace, num_trace = [], [], [], [], []
            for struct in correlations.keys():
                if struct != 'overall' and correlations[struct]['count'] >= 10:
                    pr_trace.append(-correlations[struct]['pearson'][0])
                    pr_pval_trace.append(correlations[struct]['pearson'][1])
                    sr_trace.append(-correlations[struct]['spearman'][0])
                    sr_pval_trace.append(correlations[struct]['spearman'][1])
                    num_trace.append(correlations[struct]['count'])
            pr = np.mean(pr_trace)
            pr_std_error = np.std(pr_trace) / np.sqrt(len(num_trace))
            pr_pval = combine_pvalues(pr_pval_trace, method='fisher')[1]
            sr = np.mean(sr_trace)
            sr_std_error = np.std(sr_trace) / np.sqrt(len(num_trace))
            sr_pval = combine_pvalues(sr_pval_trace, method='fisher')[1]
            num_measurements = np.sum(num_trace)
            num_structures = len(num_trace)
        else:
            pr = -correlations['overall']['pearson'][0] # flip correlation so higher is better
            pr_std_error = np.nan
            pr_pval = correlations['overall']['pearson'][1] # flip correlation so higher is better
            sr = -correlations['overall']['spearman'][0]
            sr_std_error = np.nan
            sr_pval = correlations['overall']['spearman'][1]
            num_measurements = correlations['overall']['count']
            num_structures = 94

        num_measurements_trace.append(num_measurements)

        correlations_values_in_table[model + ' - Pearsonr'] = pr
        correlations_values_in_table[model + ' - Pearsonr std error'] = pr_std_error
        correlations_values_in_table[model + ' - Pearsonr p-value'] = pr_pval
        correlations_values_in_table[model + ' - Spearmanr'] = sr
        correlations_values_in_table[model + ' - Spearmanr std error'] = sr_std_error
        correlations_values_in_table[model + ' - Spearmanr p-value'] = sr_pval

    if len(set(num_measurements_trace)) > 1:
        print('WARNING: Number of measurements for each model is not the same')
        print(num_measurements_trace)

    metadata_values = ['S669', False, '-ddG stability', num_measurements_trace[-1], num_structures, True, False, curr_comp]

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





    












