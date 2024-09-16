
'''

This just creates a training dataset with all of skempi, without a test split, so that we can fine-tune on the whole of skempi for downstream application on other non-skempi test data.
It creates a validation set, and just uses the same validation set as a test set as well.
I just don't want to lose skempi data from training.

'''

import os
import math
import numpy as np
import pandas as pd
import json
import pickle
import random

MULTIPLE_MUT_SEPARATOR = '|'

PERC_VALID_PDBIDS = 0.15


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()


    np.random.seed(args.seed)

    df = pd.read_csv('skempi_v2_cleaned.csv')

    # rename columns to match fine-tuning format
    renaming_dict = {'PDB_filename': 'pdbid',
                        'mutant_chain': 'chainid',
                        'mutant': 'variant',
                        'ddG': 'score'}
    df.rename(columns=renaming_dict, inplace=True)

    # remove not-finite measurements
    df = df[np.isfinite(df['score'])]

    # keep only single point mutations
    df = df[np.array([len(variant.split(MULTIPLE_MUT_SEPARATOR)) == 1 for variant in df['variant']])]

    # make 'pdb_to_residues' json file
    pdb_to_residues = {}
    for i, row in df.iterrows():
        pdbid = row['pdbid']
        if pdbid not in pdb_to_residues:
            pdb_to_residues[pdbid] = []
        pdb_to_residues[pdbid].append((row['chainid'], int(row['variant'][1:-1]), ' ')) # assume no icode
    with open('pdb_to_residues.json', 'w') as f:
        json.dump(pdb_to_residues, f)


    # split train_and_valid_data into train and valid
    pdbids = df['pdbid'].unique()
    np.random.shuffle(pdbids)
    num_valid_pdbs = int(PERC_VALID_PDBIDS * len(pdbids))
    valid_pdbids = pdbids[:num_valid_pdbs]
    train_pdbids = pdbids[num_valid_pdbs:]
    train_data = df[df['pdbid'].isin(train_pdbids)]
    valid_data = df[df['pdbid'].isin(valid_pdbids)]
    test_data = valid_data.copy()

    output_dir = 'sall'
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(f'{output_dir}/train_targets.csv', index=False)
    valid_data.to_csv(f'{output_dir}/valid_targets.csv', index=False)
    test_data.to_csv(f'{output_dir}/test_targets.csv', index=False)

    assert np.sum(~np.isfinite(train_data['score'])) == 0
    assert np.sum(~np.isfinite(valid_data['score'])) == 0
    assert np.sum(~np.isfinite(test_data['score'])) == 0

    