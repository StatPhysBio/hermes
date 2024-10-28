
import os
import numpy as np
import pandas as pd
import json

NUM_VALID_PDBIDS = 8

SEED = 42
np.random.seed(SEED)

from hermes.utils.protein_naming import aa_to_one_letter


if __name__ == '__main__':

    train_and_valid_data = pd.read_csv('cdna117K_aligned_with_seq_pos.csv')
    test_data = pd.read_csv('T2837_aligned_with_seq_pos.csv')

    # rename columns
    renaming_dict = {'pdb_code': 'pdbid',
                     'chain_id': 'chainid',
                     'ddG': 'score'}
    train_and_valid_data.rename(columns=renaming_dict, inplace=True)
    test_data.rename(columns=renaming_dict, inplace=True)
    
    # make 'variant' column with 'from', 'position' and 'to' columns
    train_and_valid_data['variant'] = [f'{aa_to_one_letter[from_aa]}{position}{aa_to_one_letter[to_aa]}' for from_aa, position, to_aa in zip(train_and_valid_data['from'], train_and_valid_data['position'], train_and_valid_data['to'])]
    test_data['variant'] = [f'{aa_to_one_letter[from_aa]}{position}{aa_to_one_letter[to_aa]}' for from_aa, position, to_aa in zip(test_data['from'], test_data['position'], test_data['to'])]

    # make 'variant_seq' column with 'from', 'one_idxed_seq_pos' and 'to' columns
    train_and_valid_data['variant_seq'] = [f'{aa_to_one_letter[from_aa]}{position}{aa_to_one_letter[to_aa]}' for from_aa, position, to_aa in zip(train_and_valid_data['from'], train_and_valid_data['one_idxed_seq_pos'], train_and_valid_data['to'])]
    test_data['variant_seq'] = [f'{aa_to_one_letter[from_aa]}{position}{aa_to_one_letter[to_aa]}' for from_aa, position, to_aa in zip(test_data['from'], test_data['one_idxed_seq_pos'], test_data['to'])]

    # make 'pdb_to_residues' json file
    pdb_to_residues = {}
    for data_df in [train_and_valid_data, test_data]:
        for i, row in data_df.iterrows():
            pdbid = row['pdbid']
            if pdbid not in pdb_to_residues:
                pdb_to_residues[pdbid] = set()
            pdb_to_residues[pdbid].add((row['chainid'], row['position'], ' ')) # assume no icode
        for pdbid in pdb_to_residues:
            pdb_to_residues[pdbid] = list(pdb_to_residues[pdbid])
    with open('pdb_to_residues.json', 'w') as f:
        json.dump(pdb_to_residues, f)

    # split train_and_valid_data into train and valid
    pdbids = train_and_valid_data['pdbid'].unique()
    np.random.shuffle(pdbids)
    valid_pdbids = pdbids[:NUM_VALID_PDBIDS]
    train_pdbids = pdbids[NUM_VALID_PDBIDS:]
    train_data = train_and_valid_data[train_and_valid_data['pdbid'].isin(train_pdbids)]
    valid_data = train_and_valid_data[train_and_valid_data['pdbid'].isin(valid_pdbids)]

    # save the data
    train_data.to_csv('train_targets.csv', index=False)
    valid_data.to_csv('valid_targets.csv', index=False)
    test_data.to_csv('test_targets.csv', index=False)



