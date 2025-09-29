
import os
import numpy as np
import pandas as pd

AMINOACIDS = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

HERMES_MODELS = ['hermes_py_050', 'hermes_py_050_ft_cdna117k_relaxed_pred', 'hermes_py_050_ft_cdna117k_ddg_st', 'hermes_py_050_ft_mega_thermompnn_ddg_st']

PROTEINMPNN_MODELS = ['proteinmpnn_v_48_030']

THERMOMPNN_MODELS = ['thermompnn']

ANTIGENS = ['rsv_f', 'hmpv_f', 'ha', 'denv_e', 'sars_cov_2']

COLUMNS = ['antigen', 'model', 'label', 'mutation', 'chain', 'pdb', 'rank_wt', 'rank_mt'] + [f'{aa}_pred' for aa in AMINOACIDS]


def clean_hermes_df(df):
    df = df.rename(columns={f'logit_{aa}': f'{aa}_pred' for aa in AMINOACIDS})
    df = df[COLUMNS[2:]]
    return df

def clean_proteinmpnn_df(df):

    df['resnum'] = df['mutation'].apply(lambda x: int(x[1:-1]))
    df['mutant'] = df['mutation'].apply(lambda x: x[-1])

    data_to_add = {'rank_wt': [], 'rank_mt': []}
    for aa in AMINOACIDS:
        data_to_add[f'{aa}_pred'] = []
    
    desired_df = df.loc[df['desired_one'] == 1]

    for mutation in desired_df['mutation']:
        resnum = int(mutation[1:-1])
        wt = mutation[0]
        mt = mutation[-1]
        df_subset = df[df['resnum'] == resnum]

        aas = []
        ddgs = []
        for i, row in df_subset.iterrows():
            data_to_add[f'{row["mutant"]}_pred'].append(row['log_p_mt'])
            aas.append(row['mutant'])
            ddgs.append(row['log_p_mt'])
        rank_wt = (np.array(ddgs) > df_subset[df_subset['mutant'] == wt]['log_p_mt'].values[0]).sum() + 1
        rank_mt = (np.array(ddgs) > df_subset[df_subset['mutant'] == mt]['log_p_mt'].values[0]).sum() + 1
        data_to_add['rank_wt'].append(rank_wt)
        data_to_add['rank_mt'].append(rank_mt)

    for key in data_to_add:
        desired_df[key] = data_to_add[key]
    
    desired_df = desired_df[COLUMNS[2:]]

    return desired_df

def clean_thermompnn_df(df):
    df = df.rename(columns={f'ddG_pred_{aa}': f'{aa}_pred' for aa in AMINOACIDS})
    df = df[COLUMNS[2:]]
    return df


if __name__ == '__main__':

    full_df = None

    for antigen in ANTIGENS:

        antigen_df = None

        for model in HERMES_MODELS:
            df = pd.read_csv(f'./{antigen}/results/{model}.csv')
            df = clean_hermes_df(df)
            df['model'] = model
            if antigen_df is None:
                antigen_df = df
            else:
                antigen_df = pd.concat([antigen_df, df])
        
        for model in PROTEINMPNN_MODELS:
            df = pd.read_csv(f'./{antigen}/results/{model}/zero_shot_predictions/mutations_all-num_seq_per_target=10-use_mt_structure=0.csv')
            df = clean_proteinmpnn_df(df)
            df['model'] = model
            if antigen_df is None:
                antigen_df = df
            else:
                antigen_df = pd.concat([antigen_df, df])
        
        for model in THERMOMPNN_MODELS:
            df = pd.read_csv(f'./{antigen}/results/{model}.csv')
            df = clean_thermompnn_df(df)
            df['model'] = model
            if antigen_df is None:
                antigen_df = df
            else:
                antigen_df = pd.concat([antigen_df, df])

        antigen_df['antigen'] = antigen
        antigen_df = antigen_df[COLUMNS]


        if full_df is None:
            full_df = antigen_df
        else:
            full_df = pd.concat([full_df, antigen_df])

    full_df.to_csv('full_results.csv', index=None)


