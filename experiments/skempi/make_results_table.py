
import os, sys
import json
import numpy as np
import pandas as pd


sys.path.append('..')
from get_full_table import METADATA_COLUMNS, HERMES_MODELS, PROTEINMPNN_MODELS, HERMES_MODELS_SUP_SKEMPI


system_name = 'skempi_v2_cleaned_NO_1KBH'

df = None



for curr_comp_in_json, curr_comp in zip(['Per-Structure', 'Overall'],
                                        ['per structure', 'overall']):

    for mut_types, is_single, is_multi in zip(['all_types_of_mutations', 'single_point_mutations', 'multi_point_mutations'],
                                                [True, True, False],
                                                [True, False, True]):    

        correlations_values_in_table = {}

        num_measurements_trace = []
        num_structures_trace = []

        for model in HERMES_MODELS + PROTEINMPNN_MODELS + HERMES_MODELS_SUP_SKEMPI:

            try:
                if model in HERMES_MODELS:
                    with open(f'results/{model}/{system_name}-{model}-use_mt_structure=0-correlations.json', 'r') as f:
                        correlations = json.load(f)
                elif model in PROTEINMPNN_MODELS:
                    with open(f'results/{model}/{system_name}-num_seq_per_target=10-use_mt_structure=0-correlations.json', 'r') as f:
                        correlations = json.load(f)
                elif model in HERMES_MODELS_SUP_SKEMPI:
                    if mut_types != 'single_point_mutations':
                        continue
                    sup_split = model.split('_')[-1]
                    with open(f'supervised_models/{sup_split}/results/{model}/test_targets-{model}-use_mt_structure=0-correlations.json', 'r') as f:
                        correlations = json.load(f)
            except:
                print(f'WARNING: Could not find correlations for {model}')
                continue
            
            pr = -correlations[curr_comp_in_json][mut_types]['Pr'] # flip correlation so higher is better
            pr_pval = correlations[curr_comp_in_json][mut_types]['Pr_pval']

            sr = -correlations[curr_comp_in_json][mut_types]['Sr'] # flip correlation so higher is better
            sr_pval = correlations[curr_comp_in_json][mut_types]['Sr_pval']

            precision = correlations[curr_comp_in_json][mut_types]['precision']
            recall = correlations[curr_comp_in_json][mut_types]['recall']
            f1 = correlations[curr_comp_in_json][mut_types]['f1']
            auroc = correlations[curr_comp_in_json][mut_types]['auroc']
            accuracy = correlations[curr_comp_in_json][mut_types]['accuracy']

            num_measurements = correlations[curr_comp_in_json][mut_types]['num']
            num_measurements_trace.append(num_measurements)

            num_structures = correlations[curr_comp_in_json][mut_types]['num_struc']
            num_structures_trace.append(num_structures)

            correlations_values_in_table[model + ' - Pearsonr'] = pr
            correlations_values_in_table[model + ' - Pearsonr p-value'] = pr_pval
            correlations_values_in_table[model + ' - Spearmanr'] = sr
            correlations_values_in_table[model + ' - Spearmanr p-value'] = sr_pval

            correlations_values_in_table[model + ' - precision'] = precision
            correlations_values_in_table[model + ' - recall'] = recall
            correlations_values_in_table[model + ' - f1'] = f1
            correlations_values_in_table[model + ' - auroc'] = auroc
            correlations_values_in_table[model + ' - accuracy'] = accuracy

            if curr_comp_in_json == 'Per-Structure':
                correlations_values_in_table[model + ' - Pearsonr std error'] = correlations[curr_comp_in_json][mut_types]['Pr_std'] / np.sqrt(correlations[curr_comp_in_json][mut_types]['num_struc'])
                correlations_values_in_table[model + ' - Spearmanr std error'] = correlations[curr_comp_in_json][mut_types]['Sr_std'] / np.sqrt(correlations[curr_comp_in_json][mut_types]['num_struc'])
            else:
                correlations_values_in_table[model + ' - Pearsonr std error'] = np.nan
                correlations_values_in_table[model + ' - Spearmanr std error'] = np.nan

        if len(set(num_measurements_trace)) > 1:
            print('WARNING: Number of measurements for each model is not the same')
            print(num_measurements_trace)
        
        if len(set(num_structures_trace)) > 1:
            print('WARNING: Number of structures for each model is not the same')
            print(num_structures_trace)

        metadata_values = ['SKEMPI', False, '-ddG binding', num_measurements_trace[-1], num_structures_trace[-1], is_single, is_multi, curr_comp]

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





    












