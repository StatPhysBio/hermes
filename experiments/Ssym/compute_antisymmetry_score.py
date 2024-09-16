

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

import argparse



color_list = plt.get_cmap('tab20').colors

blue = color_list[0]
blue_light = color_list[1]
orange = color_list[2]
orange_light = color_list[3]
green = color_list[4]
green_light = color_list[5]
red = color_list[6]
red_light = color_list[7]
purple = color_list[8]
purple_light = color_list[9]
brown = color_list[10]
brown_light = color_list[11]
pink = color_list[12]
pink_light = color_list[13]
gray = color_list[14]
gray_light = color_list[15]
olive = color_list[16]
olive_light = color_list[17]
cyan = color_list[18]
cyan_light = color_list[19]

MODEL_TO_COLOR_dict = {
    'hermes_bp_000': blue,
    'hermes_bp_050': blue_light,
    'hermes_py_000': purple,
    'hermes_py_050': purple_light,

    'hermes_bp_000_ft_ros_ddg_st': cyan,
    'hermes_bp_050_ft_ros_ddg_st': cyan_light,
    'hermes_py_000_ft_ros_ddg_st': pink,
    'hermes_py_050_ft_ros_ddg_st': pink_light,

    'hermes_bp_000_ft_cdna117k_ddg_st': cyan,
    'hermes_bp_050_ft_cdna117k_ddg_st': cyan_light,
    'hermes_py_000_ft_cdna117k_ddg_st': pink,
    'hermes_py_050_ft_cdna117k_ddg_st': pink_light,

    'hermes_bp_000_ft_esmfold_cdna117k_ddg_st': brown,
    'hermes_bp_050_ft_esmfold_cdna117k_ddg_st': brown_light,
    'hermes_py_000_ft_esmfold_cdna117k_ddg_st': olive,
    'hermes_py_050_ft_esmfold_cdna117k_ddg_st': olive_light,

    'hermes_py_000_untrained_ft_cdna117k_ddg_st': orange,
    'hermes_py_050_untrained_ft_cdna117k_ddg_st': orange_light,

    'esm_1v_wt_marginals': gray,
    'proteinmpnn_v_48_002': green,
    'proteinmpnn_v_48_020': green_light,
    'proteinmpnn_v_48_030': green_light
}

MODEL_TO_COLOR = lambda x: MODEL_TO_COLOR_dict[x] if x in MODEL_TO_COLOR_dict else 'black'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True, help='HERMES model name')
    parser.add_argument('--use_mt_structure', type=int, default=0)
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
    
    ## assume the two dataframes are parallel, i.e. that each row represents the same site, antisymmetric in each respective dataframe
    dir_df = pd.read_csv(f'Ssym_dir/results/{args.model_version}/ssym_dir_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}.csv')
    dir_experimental_scores = dir_df['score'].values
    dir_predicted_scores = dir_df[pred_column].values
    dir_pdbids = dir_df['pdbid'].values

    inv_df = pd.read_csv(f'Ssym_inv/results/{args.model_version}/ssym_inv_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}.csv')
    inv_experimental_scores = inv_df['score'].values
    inv_predicted_scores = inv_df[pred_column].values
    inv_pdbids = inv_df['pdbid'].values

    mask = np.isfinite(dir_experimental_scores) & np.isfinite(dir_predicted_scores) & np.isfinite(inv_experimental_scores) & np.isfinite(inv_predicted_scores)
    dir_experimental_scores = dir_experimental_scores[mask]
    dir_predicted_scores = dir_predicted_scores[mask]
    dir_pdbids = dir_pdbids[mask]
    inv_experimental_scores = inv_experimental_scores[mask]
    inv_predicted_scores = inv_predicted_scores[mask]
    inv_pdbids = inv_pdbids[mask]

    neg_inv_predicted_scores = -inv_predicted_scores
    neg_inv_experimental_scores = -inv_experimental_scores


    ## make scatterplot of dir vs neg_inv predicted scores, color the points by dir_pdbid

    pdbid_to_color = {pdbid: plt.cm.tab20.colors[i] for i, pdbid in enumerate(list(set(dir_pdbids)))}

    # make the plot square

    fig, ax = plt.subplots()
    ax.scatter(dir_predicted_scores, inv_predicted_scores, color=MODEL_TO_COLOR(args.model_version), alpha=0.5)# , c=[pdbid_to_color[pdbid] for pdbid in dir_pdbids], alpha=0.5)
    
    # set xlim equal to ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lim_same = min(xlim[0], ylim[1]), max(xlim[1], ylim[0])
    lim_same = min(lim_same[0], -lim_same[1]), max(lim_same[1], -lim_same[0])
    ax.set_xlim(lim_same)
    ax.set_ylim(lim_same)

    ax.plot([0,1],[1,0], c='k', transform=ax.transAxes)

    ax.axvline(0, c='grey', ls='--')
    ax.axhline(0, c='grey', ls='--')
    ax.set_title(args.model_version + '\n' + r'$R^2$: %.3f' % r2_score(dir_predicted_scores, neg_inv_predicted_scores), fontsize=16)
    ax.set_xlabel('Ssym-direct predictions', fontsize=16)
    ax.set_ylabel('Ssym-reverse predictions', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plot_name = f'plots/ssym_antisymmetry_{args.model_version}-use_mt_structure={args.use_mt_structure}'
    plt.savefig(f'{plot_name}.png')
    plt.savefig(f'{plot_name}.pdf')
    plt.close()

    ## save r2_score to file
    os.makedirs('antisymmetry_scores', exist_ok=True)
    with open(f'antisymmetry_scores/ssym_antisymmetry_score_{args.model_version}-use_mt_structure={args.use_mt_structure}.txt', 'w') as f:
        f.write(str(r2_score(dir_predicted_scores, neg_inv_predicted_scores)))

