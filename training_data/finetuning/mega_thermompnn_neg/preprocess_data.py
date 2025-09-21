
import os
import numpy as np
import pandas as pd
import json

SEED = 42
np.random.seed(SEED)

from hermes.utils.protein_naming import aa_to_one_letter


if __name__ == '__main__':

    train_data = pd.read_csv('mega_train.csv')
    valid_data = pd.read_csv('mega_val.csv')
    test_data = pd.read_csv('mega_test.csv')

    # rename columns
    renaming_dict = {'mut_type': 'variant',
                     'WT_name': 'pdbid',
                     'ddG_ML': 'score'}
    train_data.rename(columns=renaming_dict, inplace=True)
    valid_data.rename(columns=renaming_dict, inplace=True)
    test_data.rename(columns=renaming_dict, inplace=True)

    # negative of score
    train_data['score'] = -train_data['score']
    valid_data['score'] = -valid_data['score']
    test_data['score'] = -test_data['score']

    # make chainid column with value 'A' everywhere
    train_data['chainid'] = 'A'
    valid_data['chainid'] = 'A'
    test_data['chainid'] = 'A'

    # remove rows where variant is equal to wt
    train_data = train_data[train_data['variant'] != 'wt']
    valid_data = valid_data[valid_data['variant'] != 'wt']
    test_data = test_data[test_data['variant'] != 'wt']

    # make 'position' column from the middle of the 'variant' column
    train_data['position'] = [int(variant[1:-1]) for variant in train_data['variant']]
    valid_data['position'] = [int(variant[1:-1]) for variant in valid_data['variant']]
    test_data['position'] = [int(variant[1:-1]) for variant in test_data['variant']]

    # remove '.pdb' suffix from pdbid column
    train_data['pdbid'] = [pdbid.split('.')[0] for pdbid in train_data['pdbid']]
    valid_data['pdbid'] = [pdbid.split('.')[0] for pdbid in valid_data['pdbid']]
    test_data['pdbid'] = [pdbid.split('.')[0] for pdbid in test_data['pdbid']]
    

    # make 'pdb_to_residues' json file
    pdb_to_residues = {}
    for data_df in [train_data, valid_data, test_data]:
        for i, row in data_df.iterrows():
            pdbid = row['pdbid']
            if pdbid not in pdb_to_residues:
                pdb_to_residues[pdbid] = set()
            pdb_to_residues[pdbid].add((row['chainid'], row['position'], ' ')) # assume no icode
        for pdbid in pdb_to_residues:
            pdb_to_residues[pdbid] = list(pdb_to_residues[pdbid])
    with open('pdb_to_residues.json', 'w') as f:
        json.dump(pdb_to_residues, f)


    # add 'score_squared' column
    train_data['score_squared'] = train_data['score']**2
    valid_data['score_squared'] = valid_data['score']**2
    test_data['score_squared'] = test_data['score']**2

    # save the data
    train_data.to_csv('train_targets.csv', index=False)
    valid_data.to_csv('valid_targets.csv', index=False)
    test_data.to_csv('test_targets.csv', index=False)



