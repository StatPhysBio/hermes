

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import spearmanr, pearsonr, combine_pvalues

import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate correlations between predictions and measurements for SKEMPI dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset of split type.')
    parser.add_argument('--model_version_base', type=str, required=True, help='Model version. without dataset and fold')
    parser.add_argument('--use_mt_structure', type=int, default=0, help='Whether to use the mutant structure or the wildtype structure.')
    args = parser.parse_args()
            
    use_mt_structure = args.use_mt_structure

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    min_num_of_mutants_for_groups = 10
    system = 'SKEMPI'
    system_name_in_csv_file = 'test_targets'
    mutation_column = 'variant'
    pdb_column = 'pdbid'

    old_target_column = 'score'
    target_column = 'ddg'

    prediction_column = 'log_proba_mt__minus__log_proba_wt'

    correlations_dict = {
        'Overall': {},
        'Per-Structure': {}
    }

    df_full_orig = None
    for fold_name in ['fold1', 'fold2', 'fold3']:
        model_version_in_filename = args.model_version_base + '_' + args.dataset + '_' + fold_name + '_ddg_bi'
        df = pd.read_csv(os.path.join(this_file_dir, args.dataset, fold_name, 'results', model_version_in_filename, f'{system_name_in_csv_file}-{model_version_in_filename}-use_mt_structure={use_mt_structure}.csv'))
        if df_full_orig is None:
            df_full_orig = df
        else:
            df_full_orig = pd.concat([df_full_orig, df], ignore_index=True)
    
    # rename target column
    df_full_orig.rename(columns={old_target_column: target_column}, inplace=True)

    # filter out nans and infs
    df_full_orig = df_full_orig.loc[np.isfinite(df_full_orig[prediction_column].values)].reset_index(drop=True)

    
    for num_mut_mode in ['single_point_mutations']:

        df_full = df_full_orig.copy()

        if num_mut_mode == 'single_point_mutations':
            is_single_point_mutation = [len(mutation.split('|')) == 1 for mutation in df_full[mutation_column]]
            df_full = df_full.loc[is_single_point_mutation].reset_index(drop=True)
            single_point_mutation_str = '-single_point_mutations'
        elif num_mut_mode == 'multi_point_mutations':
            is_multi_point_mutation = [len(mutation.split('|')) > 1 for mutation in df_full[mutation_column]]
            df_full = df_full.loc[is_multi_point_mutation].reset_index(drop=True)
            single_point_mutation_str = '-multi_point_mutations'
        elif num_mut_mode == 'all_types_of_mutations':
            single_point_mutation_str = ''
        else:
            raise ValueError(f'num_mut_mode={num_mut_mode} not recognized')
    

        # cross with DSMBind because for some reason DSMBind has less rows
        df_dsmbind = pd.read_csv(os.path.join(this_file_dir, 'skempi_filtered_ddg.csv'))
        df_dsmbind['to_merge'] = [pdb+mut for pdb, mut in zip(df_dsmbind['pdb'], df_dsmbind['mutation'])]
        df_full['to_merge'] = [pdb+mut for pdb, mut in zip(df_full['#Pdb'], df_full['Mutation(s)_cleaned'])]
        df_full = df_full.drop_duplicates(subset=['to_merge']) # drops our duplicates, DSSMBind has no duplicates already
        df_full = df_full.merge(df_dsmbind, on=['to_merge'], how='inner')

        print(len(df_full))


        # exlude nans and infs, keep only one wildtype measurement
        try:
            is_not_inf_nor_nan_mask = np.logical_and(np.isfinite(df_full[target_column]), np.isfinite(df_full[prediction_column]))
        except KeyError:
            print(f"Warning: {target_column} values don't quite match, using {target_column}_x.", file=sys.stderr)
            target_column = target_column + '_x'
            is_not_inf_nor_nan_mask = np.logical_and(np.isfinite(df_full[target_column]), np.isfinite(df_full[prediction_column]))

        if 'is_wt' in df_full.columns:
            is_wt_mask = df_full['is_wt']
            first_wt_mask = np.zeros(df_full.shape[0], dtype=bool)
            first_wt_mask[np.arange(df_full.shape[0])[is_wt_mask][0]] = True
            mask = np.logical_and(is_not_inf_nor_nan_mask, np.logical_or(~is_wt_mask, first_wt_mask))
        else:
            print('Warning: no "is_wt" column in csv file, assuming there are no wildtype duplicates.', file=sys.stderr)
            mask = is_not_inf_nor_nan_mask

        df_full = df_full.loc[mask].reset_index(drop=True)

        print(len(df_full))


        def get_correlations(df, do_group_structures=False):
            if do_group_structures:
                assert min_num_of_mutants_for_groups is not None
                groups = df.groupby(pdb_column)
                pr, pr_pval, sr, sr_pval, num = [], [], [], [], []
                for group_name, group_df in groups:
                    group_df = group_df.reset_index(drop=True)
                    
                    if group_df.shape[0] >= min_num_of_mutants_for_groups:
                        targets, predictions = group_df[target_column].values, group_df[prediction_column].values
                        pr_, pr_pval_ = pearsonr(targets, predictions)
                        sr_, sr_pval_ = spearmanr(targets, predictions)
                        pr.append(pr_)
                        pr_pval.append(pr_pval_)
                        sr.append(sr_)
                        sr_pval.append(sr_pval_)
                        num.append(len(targets))
                return np.mean(pr), np.std(pr), combine_pvalues(pr_pval, method='fisher')[1], np.mean(sr), np.std(sr), combine_pvalues(sr_pval, method='fisher')[1], np.sum(num), len(num)
            else:
                targets, predictions = df[target_column].values, df[prediction_column].values
                pr, pr_pval = pearsonr(targets, predictions)
                sr, sr_pval = spearmanr(targets, predictions)
                num = len(targets)
                return pr, pr_pval, sr, sr_pval, num, len(df.groupby(pdb_column))


        pr, pr_pval, sr, sr_pval, num, num_struc = get_correlations(df_full)
        correlations_dict['Overall'][num_mut_mode] = {'Pr': float(pr), 'Pr_pval': float(pr_pval), 'Sr': float(sr), 'Sr_pval': float(sr_pval), 'num': float(num), 'num_struc': float(num_struc)}

        pr, pr_std, pr_pval, sr, sr_std, sr_pval, num, num_struc = get_correlations(df_full, do_group_structures=True)
        correlations_dict['Per-Structure'][num_mut_mode] = {'Pr': float(pr), 'Pr_std': float(pr_std), 'Pr_pval': float(pr_pval), 'Sr': float(sr), 'Sr_std': float(sr_std), 'Sr_pval': float(sr_pval), 'num': float(num), 'num_struc': float(num_struc)}

    os.makedirs(os.path.join(this_file_dir, args.dataset, 'results', args.model_version_base + '_' + args.dataset), exist_ok=True)
    with open(os.path.join(this_file_dir, args.dataset, 'results', args.model_version_base + '_' + args.dataset, f"{system_name_in_csv_file}-{args.model_version_base + '_' + args.dataset}-use_mt_structure={use_mt_structure}-correlations.json"), 'w') as f:
        json.dump(correlations_dict, f, indent=4)






