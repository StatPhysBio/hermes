
import os
from glob import glob
import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

import argparse

def center_fermi_transform(y: np.ndarray, beta: float = 0.4, alpha: float = 3.0) -> np.ndarray:
    zero_value = 1 / (1 + np.exp(beta*alpha))
    return y - zero_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True, help='HCNN model name')
    parser.add_argument('--use_mt_structure', type=int, default=0)
    parser.add_argument('--system_name', type=str, required=True)
    args = parser.parse_args()

    if 'proteinmpnn' in args.model_version:
        model_version_in_filename = 'num_seq_per_target=10'
        pred_column = 'log_p_mt__minus__log_p_wt'
    elif 'esm_1v' in args.model_version:
        model_version_in_filename = args.model_version
        pred_column = 'avg_pred_ddg'
    else:
        model_version_in_filename = args.model_version
        pred_column = 'log_proba_mt__minus__log_proba_wt'


    df = pd.read_csv(f'./results/{args.model_version}/{args.system_name}_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}.csv')
    experimental_scores = df['score'].values
    predicted_scores = df[pred_column].values
    pdbids = df['pdbid'].values
    chainids = df['chainid'].values
    variants = df['variant'].values

    mask = np.logical_and(~np.isnan(experimental_scores), ~np.isnan(predicted_scores))
    print(f'Num NaN values: {np.sum(~mask)}/{mask.shape[0]}')
    experimental_scores = experimental_scores[mask]
    predicted_scores = predicted_scores[mask]
    datasets = df['dataset'].values[mask]
    pdbids = pdbids[mask]
    chainids = chainids[mask]
    variants = variants[mask]
    experimental_stabilizing = np.array([1 if exp_score <= 0 else 0 for exp_score in experimental_scores])
    predicted_stabilizing = np.array([1 if pred_score >= 0 else 0 for pred_score in predicted_scores])

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

    # split by dataset
    datasets_unique = np.unique(datasets)
    for dataset in datasets_unique:
        mask = datasets == dataset
        correlations[dataset] = {
            'pearson': (pearsonr(experimental_scores[mask], predicted_scores[mask])[0], pearsonr(experimental_scores[mask], predicted_scores[mask])[1]),
            'spearman': (spearmanr(experimental_scores[mask], predicted_scores[mask])[0], spearmanr(experimental_scores[mask], predicted_scores[mask])[1]),
            'precision': precision_score(experimental_stabilizing[mask], predicted_stabilizing[mask]),
            'recall': recall_score(experimental_stabilizing[mask], predicted_stabilizing[mask]),
            'auroc': roc_auc_score(experimental_stabilizing[mask], predicted_scores[mask]),
            'accuracy': accuracy_score(experimental_stabilizing[mask], predicted_stabilizing[mask]),
            'count': len(experimental_scores[mask]),
            'num_stabilizing': int(np.sum(experimental_stabilizing[mask])),
            'num_destabilizing': len(experimental_scores[mask]) - int(np.sum(experimental_stabilizing[mask]))
        }
    
    # add overall correlation
    correlations['overall'] = {
        'pearson': (pearsonr(experimental_scores, predicted_scores)[0], pearsonr(experimental_scores, predicted_scores)[1]),
        'spearman': (spearmanr(experimental_scores, predicted_scores)[0], spearmanr(experimental_scores, predicted_scores)[1]),
        'precision': precision_score(experimental_stabilizing, predicted_stabilizing),
        'recall': recall_score(experimental_stabilizing, predicted_stabilizing),
        'auroc': roc_auc_score(experimental_stabilizing, predicted_scores),
        'accuracy': accuracy_score(experimental_stabilizing, predicted_stabilizing),
        'count': len(experimental_scores),
        'num_stabilizing': int(np.sum(experimental_stabilizing)),
        'num_destabilizing': len(experimental_scores) - int(np.sum(experimental_stabilizing))
    }
    
    # save correlations
    with open(f'./results/{args.model_version}/{args.system_name}_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=4)





