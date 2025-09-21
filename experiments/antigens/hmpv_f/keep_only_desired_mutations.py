
import os
import numpy as np
import pandas as pd
import argparse

ind_to_aa_size = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'SER', 4: 'PRO',
       5: 'THR', 6: 'VAL', 7: 'ASP', 8: 'ILE', 9: 'LEU',
       10: 'ASN', 11: 'MET', 12: 'GLN', 13: 'LYS', 14: 'GLU',
       15: 'HIS', 16: 'PHE', 17: 'ARG', 18: 'TYR', 19: 'TRP'}
aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
ind_to_ol_size = {x: aa_to_one_letter[ind_to_aa_size[x]] for x in range(20)}

logit_columns = np.array([f'logit_{aa}' for _, aa in ind_to_ol_size.items()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_version', type=str, required=True)
    args = parser.parse_args()

    df_mut = pd.read_csv('mutations.csv')

    df_model = pd.read_csv(f'results_all_sites/{args.model_version}.csv')

    data_to_add = {}
    for index, row in df_mut.iterrows():
        
        # Get the mutation
        mutation = row['mutation']
        aa_wt = mutation[0]
        resnum = int(mutation[1:-1])
        aa_mt = mutation[-1]
        chain = row['chain']
        pdb = row['pdb']
        insertion_code = ' ' # assuming this because easier and I know it to be true for now

        # Get the mutation's index in the model dataframe
        index = df_model[(df_model['resname'] == aa_wt) & (df_model['resnum'] == resnum) & (df_model['chain'] == chain) & (df_model['pdb'] == pdb) & (df_model['insertion_code'] == insertion_code)].index
        assert len(index) == 1, f'Expected 1 index, got {len(index)}'

        model_row = df_model.loc[index]
        assert len(model_row) == 1, f'Expected 1 row, got {len(model_row)}'

        # compute delta logit
        delta_logit = model_row[f'logit_{aa_mt}'].values[0] - model_row[f'logit_{aa_wt}'].values[0]
        if 'delta_logit' not in data_to_add:
            data_to_add['delta_logit'] = []
        data_to_add['delta_logit'].append(delta_logit)

        # compute rank of the wildtype among the logit columns
        logit_values = model_row[logit_columns].values[0]
        sorted_indices = np.argsort(logit_values)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(logit_values)) + 1
        rank = ranks[np.where(logit_columns == f'logit_{aa_wt}')[0][0]]
        if 'rank_wt' not in data_to_add:
            data_to_add['rank_wt'] = []
        data_to_add['rank_wt'].append(rank)

        # compute rank of the mutation among the logit columns
        logit_values = model_row[logit_columns].values[0]
        sorted_indices = np.argsort(logit_values)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(logit_values)) + 1
        rank = ranks[np.where(logit_columns == f'logit_{aa_mt}')[0][0]]
        if 'rank_mt' not in data_to_add:
            data_to_add['rank_mt'] = []
        data_to_add['rank_mt'].append(rank)

        # add the logit columns to the data_to_add dictionary
        for col in logit_columns:
            if col not in data_to_add:
                data_to_add[col] = []
            data_to_add[col].append(model_row[col].values[0])
    
    # Add the new columns to the mutation dataframe
    for key, value in data_to_add.items():
        df_mut[key] = value
    
    # Save the new dataframe
    os.makedirs('results', exist_ok=True)
    df_mut.to_csv(f'results/{args.model_version}.csv', index=False)




        
        

