
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.stats import spearmanr, pearsonr

HERMES_MODELS = 'hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'.split()

PROTEINMPNN_MODELS = ['proteinmpnn_v_48_002', 'proteinmpnn_v_48_020', 'proteinmpnn_v_48_030']


if __name__ == '__main__':

    # systems is all directories in the current folder that do not start with "__"
    SYSTEMS = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith("__")]

    ref_df = pd.read_csv('raw_df_from_esm.csv')

    dfs_trace = []

    for system in tqdm(SYSTEMS):

        # subset df with current system
        ref_df_curr = ref_df[ref_df['protein_name'] == system].copy()
        ref_mutations = ref_df_curr['mutant'].values

        # get mutations from original csv file (the mutations in the prediction csv files do not reflect the actual mutations as they might have gotten shifted)
        df_input = pd.read_csv(f'{system}/output_{system}.csv')
        prediction_mutations = df_input['mutant'].values

        columns_to_add = {}
        for model in HERMES_MODELS + PROTEINMPNN_MODELS:

            if model in HERMES_MODELS:
                prediction_column = 'log_proba_mt__minus__log_proba_wt'
            elif model in PROTEINMPNN_MODELS:
                prediction_column = 'log_p_mt__minus__log_p_wt'
            else:
                raise ValueError(f'Model {model} not recognized.')

            # get all csv files (one per structure used!)
            csv_files = glob(f'{system}/results/{model}/*.csv')

            # average predictions across files
            prediction_arrays = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                prediction_arrays.append(df[prediction_column].values)
            if len(prediction_arrays) == 0:
                columns_to_add[model] = np.nan * np.ones(len(ref_mutations))
                continue
                
            # if lengths don't match, just only use the longest one. TODO: need to find a way to realign them
            if len(set([len(arr) for arr in prediction_arrays])) > 1:
                print(f'Lengths of predictions for {model} do not match for system {system}.', file=sys.stderr)
                prediction_arrays = [arr for arr in prediction_arrays if len(arr) == max([len(arr) for arr in prediction_arrays])]
            
            try:
                predictions = np.nanmean(np.vstack(prediction_arrays), axis=0)
            except Exception as e:
                print()
                print(system)
                print()
                raise e
            # predictions = prediction_arrays[0]

            assert len(predictions) == len(prediction_mutations)

            # rearrange predictions such that prediction_mutations align with ref_mutations
            ref_predictions = predictions[np.array([np.where(prediction_mutations == ref_mut)[0][0] for ref_mut in ref_mutations])]
            columns_to_add[model] = ref_predictions
        
        # add columns to df
        for model, predictions in columns_to_add.items():
            ref_df_curr[model] = predictions
        
        dfs_trace.append(ref_df_curr)
    
    df = pd.concat(dfs_trace, axis=0)

    df.to_csv('raw_df_from_esm_with_predictions.csv', index=False)


    ## make dataframes with absolute value of pearson rho (one dataframne) and spearman rho (other dataframe) for each model

    target_column = 'gt'
    prediction_columns = [column for column in df.columns if column not in {'protein_name', 'mutant'}]

    pearsonr_dict = {column: [] for column in df.columns if column not in {'mutant'}}
    spearmanr_dict = {column: [] for column in df.columns if column not in {'mutant'}}

    for system in SYSTEMS:
        df_curr = df[df['protein_name'] == system].copy()
        pearsonr_dict['protein_name'].append(system)
        spearmanr_dict['protein_name'].append(system)

        for pred_col in prediction_columns:
            mask = np.logical_and(np.isfinite(df_curr[target_column]), np.isfinite(df_curr[pred_col]))
            if sum(mask) < 2:
                pearsonr_dict[pred_col].append(np.nan)
                spearmanr_dict[pred_col].append(np.nan)
            else:
                pearsonr_dict[pred_col].append(np.abs(pearsonr(df_curr[target_column][mask], df_curr[pred_col][mask])[0]))
                spearmanr_dict[pred_col].append(np.abs(spearmanr(df_curr[target_column][mask], df_curr[pred_col][mask])[0]))
    
    df_pearsonr = pd.DataFrame(pearsonr_dict)
    df_spearmanr = pd.DataFrame(spearmanr_dict)

    df_pearsonr.to_csv('dms_results_pearsonr.csv', index=False)
    df_spearmanr.to_csv('dms_results_spearmanr.csv', index=False)


    
            


