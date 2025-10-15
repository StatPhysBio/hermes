import os
import numpy as np
import pandas as pd
import json

def get_model_df_and_dir_and_pred_column(model):

    if 'hermes' in model:
        if 'relaxed' in model and not 'relaxed_pred' in model:
            directory = f'results_{model.split("__")[1]}/{model.split("__")[0]}/'
            df_path = os.path.join(directory, f'test_targets-{model.split("__")[0]}-use_mt_structure=1.csv')
            df = pd.read_csv(df_path)
            if 'ddg_st' in model:
                pred_column = 'logit_mt_in_mt__minus__logit_wt_in_wt'
            else:
                pred_column = 'log_proba_mt_in_mt__minus__log_proba_wt_in_wt'
        else:
            directory = f'results/{model}/'
            df_path = os.path.join(directory, f'test_targets-{model}-use_mt_structure=0.csv')
            df = pd.read_csv(df_path)
            if 'ddg_st' in model:
                pred_column = 'log_proba_mt__minus__log_proba_wt' # 'pe_mt__minus__pe_wt'
            else:
                pred_column = 'log_proba_mt__minus__log_proba_wt'
    elif 'proteinmpnn' in model:
        directory = f'results/{model}/'
        df_path = os.path.join(directory, f'test_targets-num_seq_per_target=10-use_mt_structure=0.csv')
        df = pd.read_csv(df_path)
        pred_column = 'log_p_mt__minus__log_p_wt'
    elif 'esm_1v' in model:
        directory = f'results/{model}/'
        df_path = os.path.join(directory, f'test_targets-esm_1v-use_mt_structure=0.csv')
        df = pd.read_csv(df_path)
        pred_column = 'avg_pred_ddg'
    elif 'esm_1v' in model:
        directory = f'results/{model}/'
        df_path = os.path.join(directory, f'test_targets-esm_1v-use_mt_structure=0.csv')
        df = pd.read_csv(df_path)
        pred_column = 'avg_pred_ddg'
    elif 'neg_abs_diff_vdw_radius' in model:
        directory = f'results/{model}/'
        df_path = os.path.join(directory, f'test_targets-neg_abs_diff_vdw_radius-use_mt_structure=0.csv')
        df = pd.read_csv(df_path)
        pred_column = 'substitution_matrix_score'
    elif 'stability_oracle' in model:
        directory = f'results/{model}/'
        df_path = os.path.join(directory, f'test_targets-stability_oracle-use_mt_structure=0.csv')
        df = pd.read_csv(df_path)
        pred_column = 'neg_ddg_pred'
    elif 'thermompnn' in model:
        directory = f'results/{model}/'
        df_path = os.path.join(directory, f'test_targets-thermompnn-use_mt_structure=0.csv')
        df = pd.read_csv(df_path)
        pred_column = 'neg_ddg_pred'

    df = df[df[pred_column].notnull()]

    return df, directory, df_path, pred_column


def get_model_json_results(model):
    _, _, df_path, _ = get_model_df_and_dir_and_pred_column(model)
    json_path = df_path.replace('.csv', '_correlations.json')
    with open(json_path) as f:
        results = json.load(f)
    return results


def get_avg_pred_matrix(model):

    base_hermes_path = '/gscratch/spe/gvisan01/hermes/'

    if 'hermes' in model:
        if 'relaxed' in model and not 'relaxed_pred' in model:
            directory = os.path.join(base_hermes_path, f'experiments/cdna117k/results_{model.split("__")[1]}/{model.split("__")[0]}/')
            matrix_df = pd.read_csv(os.path.join(directory, f'cdna117k_avg_pred_matrix_of_{model}.csv'), index_col=0)
        else:
            directory = os.path.join(base_hermes_path, f'experiments/cdna117k/results/{model}/')
            matrix_df = pd.read_csv(os.path.join(directory, f'cdna117k_avg_pred_matrix_of_{model}.csv'), index_col=0)
    elif 'proteinmpnn' in model:
        directory = os.path.join(base_hermes_path, f'experiments/cdna117k/results/{model}/')
        matrix_df = pd.read_csv(os.path.join(directory, f'cdna117k_avg_pred_matrix_of_{model}.csv'), index_col=0)
    
    return matrix_df



