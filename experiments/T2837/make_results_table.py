
import os, sys
import json
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues


sys.path.append('..')
from get_full_table import METADATA_COLUMNS, HERMES_MODELS, PROTEINMPNN_MODELS, ESM_MODELS, SUBSTITUTION_MATRICES, OTHER_MODELS


system_name = 'T2837_ddg_experimental'

df = None

stratified_results = ['core', 'in-between', 'surface', 'alpha-helix', 'beta-sheet', 'loop']
datasets = ['myoglobin', 'p53', 's669', 'ssym', 't2226', 'overall'] # 'per-structure',

for dataset in stratified_results + datasets:

    correlations_values_in_table = {}

    num_measurements_trace = []
    num_structures_trace = []

    for model in HERMES_MODELS + PROTEINMPNN_MODELS + ESM_MODELS + SUBSTITUTION_MATRICES + OTHER_MODELS:

        if dataset in stratified_results:
            try:
                if model in HERMES_MODELS + ESM_MODELS + SUBSTITUTION_MATRICES + OTHER_MODELS:
                    name = f'results/{model}/{system_name}-{model}-use_mt_structure=0_correlations_stratified.json'
                    if not os.path.exists(name):
                        name = name.replace('correlations_stratified', 'correlations_stratrified')
                    
                    with open(name, 'r') as f:
                        correlations = json.load(f)

                elif model in PROTEINMPNN_MODELS:
                    with open(f'results/{model}/{system_name}-num_seq_per_target=10-use_mt_structure=0_correlations_stratrified.json', 'r') as f:
                        correlations = json.load(f)
                    
            except Exception as e:
                print(f'WARNING: Could not find correlations for {model} and {dataset}')
                continue
        else:
            try:
                if model in HERMES_MODELS + ESM_MODELS + SUBSTITUTION_MATRICES + OTHER_MODELS:
                    with open(f'results/{model}/{system_name}-{model}-use_mt_structure=0_correlations.json', 'r') as f:
                        correlations = json.load(f)
                elif model in PROTEINMPNN_MODELS:
                    with open(f'results/{model}/{system_name}-num_seq_per_target=10-use_mt_structure=0_correlations.json', 'r') as f:
                        correlations = json.load(f)
            except:
                print(f'WARNING: Could not find correlations for {model} and {dataset}')
                continue
        
        if dataset not in correlations:
            print(f'WARNING: Could not find correlations for {model} and {dataset}')
            continue
        
        pr = -correlations[dataset]['pearson'][0] # flip correlation so higher is better
        pr_pval = correlations[dataset]['pearson'][-1]
        sr = -correlations[dataset]['spearman'][0] # flip correlation so higher is better
        sr_pval = correlations[dataset]['spearman'][-1]
        num_measurements = correlations[dataset]['count']

        try:
            num_structures = correlations[dataset]['num_structures']
        except Exception as e:
            print(f'WARNING: Could not find num_structures for {model} and {dataset}')
            raise e

        num_measurements_trace.append(num_measurements)
        num_structures_trace.append(num_structures)


        correlations_values_in_table[model + ' - Pearsonr'] = pr
        correlations_values_in_table[model + ' - Pearsonr p-value'] = pr_pval
        correlations_values_in_table[model + ' - Spearmanr'] = sr
        correlations_values_in_table[model + ' - Spearmanr p-value'] = sr_pval

        correlations_values_in_table[model + ' - precision'] = correlations[dataset]['precision']
        correlations_values_in_table[model + ' - recall'] = correlations[dataset]['recall']
        correlations_values_in_table[model + ' - auroc'] = correlations[dataset]['auroc']
        correlations_values_in_table[model + ' - accuracy'] = correlations[dataset]['accuracy']

        if 'f1_score' in correlations[dataset]:
            correlations_values_in_table[model + ' - f1-score'] = correlations[dataset]['f1_score']
        else:
            correlations_values_in_table[model + ' - f1-score'] = np.nan

        if dataset == 'per-structure':
            correlations_values_in_table[model + ' - Pearsonr std error'] = correlations[dataset]['pearson'][1] / np.sqrt(correlations[dataset]['num_structures'])
            correlations_values_in_table[model + ' - Spearmanr std error'] = correlations[dataset]['spearman'][1] / np.sqrt(correlations[dataset]['num_structures'])
            correlations_values_in_table[model + ' - auroc std error'] = correlations[dataset]['auroc'][1] / np.sqrt(correlations[dataset]['num_structures'])
        else:
            correlations_values_in_table[model + ' - Pearsonr std error'] = np.nan
            correlations_values_in_table[model + ' - Spearmanr std error'] = np.nan
            correlations_values_in_table[model + ' - auroc std error'] = np.nan

    if len(set(num_measurements_trace)) > 1:
        print('WARNING: Number of measurements for each model is not the same')
        print(num_measurements_trace)
        print(num_structures_trace)

    metadata_values = [f'T2837 - {dataset}', False, '-ddG stability', num_measurements_trace[-1], num_structures_trace[-1], True, False, 'per structure' if dataset == 'per-structure' else 'overall']

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


