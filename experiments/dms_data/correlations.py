
import os
from glob import glob
import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr

import argparse

this_file_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True, help='HCNN or ProteinMPNN model name')
    parser.add_argument('--use_mt_structure', type=int, default=0)
    parser.add_argument('--system_name', type=str, required=True)
    parser.add_argument('--dms_column', type=str, nargs='+', required=True)
    args = parser.parse_args()

    if 'proteinmpnn' in args.model_version:
        model_version_in_filename = 'num_seq_per_target=10'
        pred_column = 'log_p_mt__minus__log_p_wt'
    else:
        model_version_in_filename = args.model_version
        pred_column = 'log_proba_mt__minus__log_proba_wt'
    
    system_name_no_extra = args.system_name.split('__')[0][7:] # remove 'output_' and the pdb info

    df = pd.read_csv(os.path.join(this_file_dir, f'{system_name_no_extra}/results/{args.model_version}/{args.system_name}-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}.csv'))
    
    correlations = {}

    for dms_column in args.dms_column:

        experimental_scores = df[dms_column].values

        predicted_scores = df[pred_column].values
        pdbids = df['wt_pdb'].values
        chainids = df['chain'].values
        variants = df['mutant'].values

        mask = np.logical_and(~np.isnan(experimental_scores), ~np.isnan(predicted_scores))
        experimental_scores = experimental_scores[mask]
        predicted_scores = predicted_scores[mask]
        pdbids = pdbids[mask]
        chainids = chainids[mask]
        variants = variants[mask]
        
        # add overall correlation (all these measurements are on a single protein at a time!)
        correlations[dms_column] = {
            'pearson': (pearsonr(experimental_scores, predicted_scores)[0], pearsonr(experimental_scores, predicted_scores)[1]),
            'spearman': (spearmanr(experimental_scores, predicted_scores)[0], spearmanr(experimental_scores, predicted_scores)[1]),
            'count': len(experimental_scores)
        }

    # save correlations
    with open(os.path.join(this_file_dir, f'{system_name_no_extra}/results/{args.model_version}/{args.system_name}-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}_correlations.json'), 'w') as f:
        json.dump(correlations, f, indent=4)
