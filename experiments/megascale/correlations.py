
import os
from glob import glob
import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr, combine_pvalues
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from utils import get_model_df_and_dir_and_pred_column, get_avg_pred_matrix

import argparse

def center_fermi_transform(y: np.ndarray, beta: float = 0.4, alpha: float = 3.0) -> np.ndarray:
    zero_value = 1 / (1 + np.exp(beta*alpha))
    return y - zero_value

min_num_of_mutants_for_groups = 10

def get_metrics(df, pdb_column, pred_column, target_column, do_group_structures=False):
    if do_group_structures:
        assert min_num_of_mutants_for_groups is not None
        groups = df.groupby(pdb_column)
        pr, pr_pval, sr, sr_pval, num = [], [], [], [], []
        auroc, num_auroc = [], []
        for group_name, group_df in groups:
            group_df = group_df.reset_index(drop=True)
            
            if group_df.shape[0] >= min_num_of_mutants_for_groups:
                targets, predictions = group_df[target_column].values, group_df[pred_column].values
                pr_, pr_pval_ = pearsonr(targets, predictions)
                sr_, sr_pval_ = spearmanr(targets, predictions)
                pr.append(pr_)
                pr_pval.append(pr_pval_)
                sr.append(sr_)
                sr_pval.append(sr_pval_)
                num.append(len(targets))

                targets_binary = np.array(targets) < 0
                if len(np.unique(targets_binary)) > 1:
                    auroc.append(roc_auc_score(targets_binary, predictions))
                    num_auroc.append(len(targets))
                else:
                    auroc.append(np.nan)
                    num_auroc.append(0)
        
        mask = np.isfinite(pr) & np.isfinite(sr)
        mask_auroc = np.isfinite(auroc)

        return {
            'Pr': np.mean(np.array(pr)[mask]),
            'Pr_std': np.std(np.array(pr)[mask]),
            'Pr_pval': combine_pvalues(np.array(pr_pval)[mask], method='fisher')[1],
            'Sr': np.mean(np.array(sr)[mask]),
            'Sr_std': np.std(np.array(sr)[mask]),
            'Sr_pval': combine_pvalues(np.array(sr_pval)[mask], method='fisher')[1],
            'num': int(np.sum(np.array(num)[mask])),
            'num_struc': len(np.array(num)[mask]),
            'AUROC': np.mean(np.array(auroc)[mask_auroc]),
            'AUROC_std': np.std(np.array(auroc)[mask_auroc]),
            'AUROC_num': int(np.sum(np.array(num_auroc)[mask_auroc])),
            'AUROC_num_struc': len(np.array(num_auroc)[mask_auroc])
        }
    
    else:
        targets, predictions = df[target_column].values, df[pred_column].values
        pr, pr_pval = pearsonr(targets, predictions)
        sr, sr_pval = spearmanr(targets, predictions)
        num = len(targets)
        auroc = roc_auc_score(targets < 0, predictions)
        auroc_num = len(targets)
        return {
            'Pr': pr,
            'Pr_pval': pr_pval,
            'Sr': sr,
            'Sr_pval': sr_pval,
            'num': int(num),
            'num_struc': int(len(df.groupby(pdb_column))),
            'AUROC': auroc,
            'AUROC_num': int(auroc_num),
            'AUROC_num_struc': int(len(df.groupby(pdb_column)))
        }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True)
    parser.add_argument('--normalize_by_average_prediction', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_both_norm_and_unnorm_for_classification', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    if args.use_both_norm_and_unnorm_for_classification:
        assert args.normalize_by_average_prediction

    df, directory, df_path, orig_pred_column = get_model_df_and_dir_and_pred_column(args.model_version)

    pdb_column = 'pdbid'
    chain_column = 'chainid'
    mutation_column = 'variant'
    target_column = 'score'


    if args.normalize_by_average_prediction:
        avg_matrix_df = get_avg_pred_matrix(args.model_version)
        avg_preds = []
        for i, row, in df.iterrows():
            variant = row['variant']
            aa_wt = variant[0]
            aa_mt = variant[-1]
            avg_preds.append(avg_matrix_df.loc[aa_wt, aa_mt])
        df[orig_pred_column + '__average'] = avg_preds
        pred_column = orig_pred_column + '__minus_its_average'
        df[pred_column] = df[orig_pred_column] - df[orig_pred_column + '__average']
        df.to_csv(df_path)
    else:
        pred_column = orig_pred_column


    experimental_scores = df[target_column].values
    predicted_scores = df[pred_column].values
    pdbids = df[pdb_column].values
    chainids = df[chain_column].values
    variants = df[mutation_column].values

    mask = np.logical_and(~np.isnan(experimental_scores), ~np.isnan(predicted_scores))
    print(f'Num NaN values: {np.sum(~mask)}/{mask.shape[0]}')
    experimental_scores = experimental_scores[mask]
    predicted_scores = predicted_scores[mask]
    # if 'stability_oracle' not in args.model_version:
    #     datasets = df['dataset'].values[mask]
    pdbids = pdbids[mask]
    chainids = chainids[mask]
    variants = variants[mask]
    experimental_stabilizing = np.array([1 if exp_score <= 0 else 0 for exp_score in experimental_scores])
    experimental_stabilizing_at_05 = np.array([1 if exp_score <= -0.5 else 0 for exp_score in experimental_scores])

    if args.use_both_norm_and_unnorm_for_classification:
        predicted_scores_unnorm = df[orig_pred_column].values[mask]
        print(predicted_scores_unnorm)
        print(predicted_scores)
        predicted_stabilizing = np.array([1 if (pred_score >= 0 and unnorm_pred_score >= 0) else 0 for pred_score, unnorm_pred_score in zip(predicted_scores, predicted_scores_unnorm)])
    else:
        predicted_stabilizing = np.array([1 if pred_score >= 0 else 0 for pred_score in predicted_scores])
    
    prediction_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    predicted_stabilizing_dict = {}
    for threshold in prediction_thresholds:
        predicted_stabilizing_dict[str(threshold).replace('.', '')] = np.array([1 if pred_score >= threshold else 0 for pred_score in predicted_scores])

    df = df[mask]

    # # split by pdbid
    # pdbid_to_experimental_scores = {}
    # pdbid_to_predicted_scores = {}
    # for pdbid, experimental_score, predicted_score in zip(pdbids, experimental_scores, predicted_scores):
    #     if pdbid not in pdbid_to_experimental_scores:
    #         pdbid_to_experimental_scores[pdbid] = []
    #         pdbid_to_predicted_scores[pdbid] = []
    #     pdbid_to_experimental_scores[pdbid].append(experimental_score)
    #     pdbid_to_predicted_scores[pdbid].append(predicted_score)
    
    # # calculate correlations
    correlations = {}
    # for pdbid in pdbid_to_experimental_scores:
    #     if len(pdbid_to_experimental_scores[pdbid]) < 2:
    #         continue
    #     curr_experimental_scores = np.array(pdbid_to_experimental_scores[pdbid])
    #     curr_predicted_scores = np.array(pdbid_to_predicted_scores[pdbid])
    #     correlations[pdbid] = {
    #         'pearson': (pearsonr(curr_experimental_scores, curr_predicted_scores)[0], pearsonr(curr_experimental_scores, curr_predicted_scores)[1]),
    #         'spearman': (spearmanr(curr_experimental_scores, curr_predicted_scores)[0], spearmanr(curr_experimental_scores, curr_predicted_scores)[1]),
    #         'count': len(curr_experimental_scores)
    #     }

    # # split by dataset
    # if 'stability_oracle' not in args.model_version:
    #     datasets_unique = np.unique(datasets)
    #     for dataset in datasets_unique:
    #         mask = datasets == dataset
    #         correlations[dataset] = {
    #             'pearson': (pearsonr(experimental_scores[mask], predicted_scores[mask])[0], pearsonr(experimental_scores[mask], predicted_scores[mask])[1]),
    #             'spearman': (spearmanr(experimental_scores[mask], predicted_scores[mask])[0], spearmanr(experimental_scores[mask], predicted_scores[mask])[1]),
    #             'precision': precision_score(experimental_stabilizing[mask], predicted_stabilizing[mask]),
    #             'recall': recall_score(experimental_stabilizing[mask], predicted_stabilizing[mask]),
    #             'auroc': roc_auc_score(experimental_stabilizing[mask], predicted_scores[mask]),
    #             'accuracy': accuracy_score(experimental_stabilizing[mask], predicted_stabilizing[mask]),
    #             '1/rmse': 1 / np.sqrt(np.mean((-experimental_scores[mask] - predicted_scores[mask])**2)), # need to make sure that the sign is correct
    #             'count': len(experimental_scores[mask]),
    #             'num_structures': len(np.unique(pdbids[mask])),
    #             'num_stabilizing': int(np.sum(experimental_stabilizing[mask])),
    #             'num_destabilizing': len(experimental_scores[mask]) - int(np.sum(experimental_stabilizing[mask]))
    #         }
    #         for threshold in prediction_thresholds:
    #             correlations[dataset][f'precision_at_05_with_{threshold}'.replace('.', '')] = precision_score(experimental_stabilizing_at_05[mask], predicted_stabilizing_dict[str(threshold).replace('.', '')][mask])
    #             correlations[dataset][f'recall_at_05_with_{threshold}'.replace('.', '')] = recall_score(experimental_stabilizing_at_05[mask], predicted_stabilizing_dict[str(threshold).replace('.', '')][mask])
    
    # add overall correlation
    correlations['overall'] = {
        'pearson': (pearsonr(experimental_scores, predicted_scores)[0], pearsonr(experimental_scores, predicted_scores)[1]),
        'spearman': (spearmanr(experimental_scores, predicted_scores)[0], spearmanr(experimental_scores, predicted_scores)[1]),
        'precision': precision_score(experimental_stabilizing, predicted_stabilizing),
        'recall': recall_score(experimental_stabilizing, predicted_stabilizing),
        'f1_score': f1_score(experimental_stabilizing, predicted_stabilizing),
        'precision_at_05': precision_score(experimental_stabilizing_at_05, predicted_stabilizing),
        'recall_at_05': recall_score(experimental_stabilizing_at_05, predicted_stabilizing),
        'auroc': roc_auc_score(experimental_stabilizing, predicted_scores),
        'accuracy': accuracy_score(experimental_stabilizing, predicted_stabilizing),
        '1/rmse': 1 / np.sqrt(np.mean((-experimental_scores - predicted_scores)**2)), # need to make sure that the sign is correct
        'count': len(experimental_scores),
        'num_structures': len(np.unique(pdbids)),
        'num_stabilizing': int(np.sum(experimental_stabilizing)),
        'num_destabilizing': len(experimental_scores) - int(np.sum(experimental_stabilizing))
    }
    for threshold in prediction_thresholds:
        correlations['overall'][f'precision_at_05_with_{threshold}'.replace('.', '')] = precision_score(experimental_stabilizing_at_05, predicted_stabilizing_dict[str(threshold).replace('.', '')])
        correlations['overall'][f'recall_at_05_with_{threshold}'.replace('.', '')] = recall_score(experimental_stabilizing_at_05, predicted_stabilizing_dict[str(threshold).replace('.', '')])

    metrics = get_metrics(df, pdb_column, pred_column, target_column, do_group_structures=True)
    correlations['per-structure'] = {
        'pearson': (float(metrics['Pr']), float(metrics['Pr_std']), float(metrics['Pr_pval'])),
        'spearman': (float(metrics['Sr']), float(metrics['Sr_std']), float(metrics['Sr_pval'])),
        'auroc': float(metrics['AUROC']),
        'count': int(metrics['num']),
        'num_structures': int(metrics['num_struc']),
    }

    # save correlations
    if args.normalize_by_average_prediction:
        if args.use_both_norm_and_unnorm_for_classification:
            with open(df_path.replace('.csv', '_correlations_normalized_by_average_prediction_and_strict_for_classification.json'), 'w') as f:
                json.dump(correlations, f, indent=4)
        else:
            with open(df_path.replace('.csv', '_correlations_normalized_by_average_prediction.json'), 'w') as f:
                json.dump(correlations, f, indent=4)
    else:
        with open(df_path.replace('.csv', '_correlations.json'), 'w') as f:
            json.dump(correlations, f, indent=4)





