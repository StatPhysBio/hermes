
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

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

    prediction_cutoff_list = ['0', 'all_perc_99', 'all_perc_97.5', 'all_perc_95', 'all_perc_90', 'all_perc_85', 'all_perc_80', 'all_perc_75', 'all_perc_70', 'all_perc_60', 'all_perc_50',
                                   'mut_perc_99', 'mut_perc_97.5', 'mut_perc_95', 'mut_perc_90', 'mut_perc_85', 'mut_perc_80', 'mut_perc_75', 'mut_perc_70', 'mut_perc_60', 'mut_perc_50',
                                   'allOrZero_perc_99', 'allOrZero_perc_97.5', 'allOrZero_perc_95', 'allOrZero_perc_90', 'allOrZero_perc_85', 'allOrZero_perc_80', 'allOrZero_perc_75', 'allOrZero_perc_70', 'allOrZero_perc_60', 'allOrZero_perc_50', 
                                   'mutOrZero_perc_99', 'mutOrZero_perc_97.5', 'mutOrZero_perc_95', 'mutOrZero_perc_90', 'mutOrZero_perc_85', 'mutOrZero_perc_80', 'mutOrZero_perc_75', 'mutOrZero_perc_70', 'mutOrZero_perc_60', 'mutOrZero_perc_50']

    ncols = 1
    nrows = 2
    colsize = 5.5
    rowsize = 3
    fig, axs = plt.subplots(figsize=(ncols*colsize, nrows*rowsize), ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    correlations = {}
    for ax_i, ddg_cutoff in enumerate([0, 0.5]):
        correlations[str(ddg_cutoff)] = {}
        experimental_stabilizing = np.array([1 if exp_score <= -ddg_cutoff else 0 for exp_score in experimental_scores])

        for prediction_cutoff in prediction_cutoff_list:

            if prediction_cutoff == '0':
                predicted_stabilizing = np.array([1 if pred_score >= 0 else 0 for pred_score in predicted_scores])

            elif 'perc' in prediction_cutoff:

                with open(f'/gscratch/stf/gvisan01/hermes/experiments/average_prediction_matrices/T2837_all_sites/all/distributions/{args.model_version}__percentiles.json', 'r') as f:
                    percentiles = json.load(f)
                
                mode, perc = prediction_cutoff.split('_')[0], prediction_cutoff.split('_')[-1]

                if mode == 'all':
                    predicted_stabilizing = np.array([1 if pred_score >= percentiles['all'][perc] else 0 for pred_score in predicted_scores])
                elif mode == 'mut':
                    predicted_stabilizing = []
                    for variant, pred_score in zip(variants, predicted_scores):
                        aa_wt, aa_mt = variant[0], variant[-1]
                        predicted_stabilizing.append(1 if pred_score >= percentiles[aa_wt][aa_mt][perc] else 0)
                    predicted_stabilizing = np.array(predicted_stabilizing)
                elif mode == 'allOrZero':
                    predicted_stabilizing = np.array([1 if pred_score >= min(percentiles['all'][perc], 0) else 0 for pred_score in predicted_scores])
                elif mode == 'mutOrZero':
                    predicted_stabilizing = []
                    for variant, pred_score in zip(variants, predicted_scores):
                        aa_wt, aa_mt = variant[0], variant[-1]
                        predicted_stabilizing.append(1 if pred_score >= min(percentiles[aa_wt][aa_mt][perc], 0) else 0)
                    predicted_stabilizing = np.array(predicted_stabilizing)
                else:
                    raise ValueError()

            else:
                raise ValueError()
        
            correlations[str(ddg_cutoff)][prediction_cutoff] = {
                'pearson': (pearsonr(experimental_scores, predicted_scores)[0], pearsonr(experimental_scores, predicted_scores)[1]),
                'spearman': (spearmanr(experimental_scores, predicted_scores)[0], spearmanr(experimental_scores, predicted_scores)[1]),
                'precision': precision_score(experimental_stabilizing, predicted_stabilizing),
                'recall': recall_score(experimental_stabilizing, predicted_stabilizing),
                'f1_score': f1_score(experimental_stabilizing, predicted_stabilizing),
                'auroc': roc_auc_score(experimental_stabilizing, predicted_scores),
                'accuracy': accuracy_score(experimental_stabilizing, predicted_stabilizing),
                'count': len(experimental_scores),
                'num_stabilizing': int(np.sum(experimental_stabilizing)),
                'num_destabilizing': len(experimental_scores) - int(np.sum(experimental_stabilizing))
            }
        
        ## make diagnostic plot!
        def add_noise(xs):
            return np.array(xs) + np.random.normal(loc=0, scale=0.02, size=len(xs))
        
        fontsize = 14
        mode_list = ['all', 'mut', 'allOrZero', 'mutOrZero']
        perc_list = [50, 60, 70, 75, 80, 85, 90, 95, 97.5, 99]
        mode_to_color_and_marker = {
            'all': ('blue', 'd'),
            'mut': ('orange', 'd'),
            'allOrZero': ('blue', 's'),
            'mutOrZero': ('orange', 's'),
        }
        ax = axs[ax_i]
        for mode in mode_list:
            f1_list = []
            for perc in perc_list:
                prediction_cutoff = f'{mode}_perc_{str(perc)}'
                f1_ = correlations[str(ddg_cutoff)][prediction_cutoff]['f1_score']
                f1_list.append(f1_)
            color, marker = mode_to_color_and_marker[mode]
            ax.scatter(add_noise(perc_list), f1_list, color=color, marker=marker)
        ax.set_xticks(perc_list)
        ax.axhline(correlations[str(ddg_cutoff)]['0']['f1_score'], ls='--', color='black')
        ax.grid(axis='both', ls='--', alpha=0.5)
        ax.set_ylabel('F1 Score', fontsize=fontsize)
        ax.set_title(f'ddG cutoff = {str(ddg_cutoff)}', fontsize=fontsize)
    
        # Create custom legend handles
        handles = [
            Line2D([0], [0], color=color, marker=marker, linestyle='',
                markersize=8, label=mode)
            for mode, (color, marker) in mode_to_color_and_marker.items()
        ]
        handles.append(
            Line2D([0], [0], color='black', linestyle='--', label='Zero')
        )
        ax.legend(handles=handles, loc="best")
    
    plt.tight_layout()
    plt.savefig(f'./results/{args.model_version}/{args.system_name}_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}_correlations_with_percentiles.png')
    plt.close()
    
    # save correlations
    with open(f'./results/{args.model_version}/{args.system_name}_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}_correlations_with_percentiles.json', 'w') as f:
        json.dump(correlations, f, indent=4)







