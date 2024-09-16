

import os
import math
import numpy as np
import pandas as pd
import json
import pickle
import random


def load_skempi_entries(csv_path, pdb_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=',')
    df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mut'] - df['dG_wt']

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    entries = []
    for i, row in df.iterrows():
        pdbcode, group1, group2 = row['#Pdb'].split('_')
        if pdbcode in block_list:
            continue
        mut_str = row['Mutation(s)_cleaned']
        muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))
        if not os.path.exists(pdb_path):
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {
            'id': i,
            'complex': row['#Pdb'],
            'mutstr': mut_str,
            'num_muts': len(muts),
            'pdbcode': pdbcode,
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'ddG': np.float32(row['ddG']),
            'pdb_path': pdb_path,
            'hold_out_type': row['Hold_out_type'],
            'hold_out_complexes': row['Hold_out_proteins'].split(',')
        }
        entries.append(entry)

    return entries


def get_pdb_splits(stringency_level_of_holdout, csv_path='skempi_v2_cleaned.csv', pdb_dir='/gscratch/spe/kborisia/learning/DSMbind_HCNN_comparison/PDBs', pdb_splits_cache='pdb_splits.pkl', split_seed=2022): # 2022 is the default seed in the RDE-PPi repository, so...

    num_cvfolds = 3 # do not change this

    # if os.path.exists(pdb_splits_cache):
    #     with open(pdb_splits_cache, 'rb') as f:
    #         return pickle.load(f)
    
    entries_full = load_skempi_entries(csv_path, pdb_dir)

    holdout_type_to_complexes = {}
    for e in entries_full:

        if str(e['hold_out_type']) == 'nan':
            continue

        holdout_types = e['hold_out_type'].split(',')
        for holdout_type in holdout_types:
            if holdout_type not in holdout_type_to_complexes:
                holdout_type_to_complexes[holdout_type] = set()
            holdout_type_to_complexes[holdout_type].add(e['complex'])

    complex_to_entries = {}
    complex_to_holdouts = {}
    for e in entries_full:

        if e['complex'] not in complex_to_entries:
            complex_to_entries[e['complex']] = []
        complex_to_entries[e['complex']].append(e)

        if e['complex'] not in complex_to_holdouts:
            complex_to_holdouts[e['complex']] = e['hold_out_complexes']
        else:
            assert complex_to_holdouts[e['complex']] == e['hold_out_complexes']
    
    # further augment the holdout type complexes by adding their holdouts
    for holdout_type in holdout_type_to_complexes:
        holdout_complexes = list(holdout_type_to_complexes[holdout_type])
        additions = []
        for complex in holdout_complexes:
            if complex not in holdout_type_to_complexes:
                additions.extend(complex_to_holdouts[complex])
        holdout_type_to_complexes[holdout_type] = set(holdout_complexes + additions)

    complex_list = sorted(complex_to_entries.keys())
    random.Random(split_seed).shuffle(complex_list)
    complex_set = set(complex_list)

    split_size = math.ceil(len(complex_list) / num_cvfolds)
    split_sizes = [split_size] * (num_cvfolds - 1) + [len(complex_list) - split_size * (num_cvfolds - 1)]


    # iterate over the complexes and assign them to the folds. for every complex, assign its holdouts to the same fold
    already_assigned = set()

    if stringency_level_of_holdout == 2: # in this case, remove all complexes that have a "type" holdout, as they will be assigned to the holdout fold
        for holdout_type in holdout_type_to_complexes:
            for complex in holdout_type_to_complexes[holdout_type]:
                if complex in complex_list:
                    complex_list.remove(complex)
        # complex_list = list(complex_set)

    complex_splits = []
    for i, split_size in enumerate(split_sizes):
        curr_split = []
        s = 0

        if stringency_level_of_holdout == 2: # in this case, just slit the three types each in its own fold, so as to keep the folds balanced in size and spliting the major types as well

            if i == 0:
                for complex in holdout_type_to_complexes['Pr/PI']:
                    if complex in already_assigned:
                        continue
                    curr_split.append(complex)
                    already_assigned.add(complex)
                    s += 1
            elif i == 1:
                for complex in holdout_type_to_complexes['AB/AG']:
                    if complex in already_assigned: # this may happen if a complex has more than one holdout tye! not much we can do in this case, it will leak, but it's not many of them
                        continue
                    curr_split.append(complex)
                    already_assigned.add(complex)
                    s += 1
            elif i == 2:
                for complex in holdout_type_to_complexes['TCR/pMHC']:
                    if complex in already_assigned: # this may happen if a complex has more than one holdout tye! not much we can do in this case, it will leak, but it's not many of them
                        continue
                    curr_split.append(complex)
                    already_assigned.add(complex)
                    s += 1

        for curr_complex in complex_list:
            if curr_complex in already_assigned:
                continue

            curr_split.append(curr_complex)
            already_assigned.add(curr_complex)
            s += 1

            if stringency_level_of_holdout > 0:

                curr_complex_holdouts = complex_to_holdouts[curr_complex]

                if stringency_level_of_holdout == 2:
                    holdouts_of_type = []
                    for complex_holdout in curr_complex_holdouts:
                        if complex_holdout in holdout_type_to_complexes:
                            holdouts_of_type.extend(holdout_type_to_complexes[complex_holdout])
                    curr_complex_holdouts += holdouts_of_type
                    curr_complex_holdouts = list(set(curr_complex_holdouts))

                for holdout_complex in curr_complex_holdouts:
                    if holdout_complex in already_assigned or holdout_complex not in complex_set:
                        continue
                    curr_split.append(holdout_complex)
                    already_assigned.add(holdout_complex)
                    s += 1

            if s >= split_size:
                break
    
        complex_splits.append(curr_split)
    
    for i, split in enumerate(complex_splits):
        print(f'Fold {i+1}: {len(split)} complexes')

    # split_size = math.ceil(len(complex_list) / num_cvfolds)
    # complex_splits = [
    #     complex_list[i*split_size : (i+1)*split_size] 
    #     for i in range(num_cvfolds)
    # ]

    # with open(pdb_splits_cache, 'wb') as f:
    #     pickle.dump(complex_splits, f)

    return complex_splits


MULTIPLE_MUT_SEPARATOR = '|'

PERC_VALID_PDBIDS = 0.15


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stringency_level_of_holdout', type=int, choices=[0, 1, 2], required=True,
                        help='Stringency level of holdout. Options: 0 (none), 1 (specific proteins are holdout), 2 (specific proteins and proteins of the same type are holdout)')
    parser.add_argument('--seed', type=int, default=43, help='Random seed')
    args = parser.parse_args()


    print("Exiting because the seed matters and I don't want to change the data.")
    exit(1)

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

    # split into three train_and_valid - test sets
    pdb_splits = get_pdb_splits(args.stringency_level_of_holdout)

    for fold_name, train_and_val_idxs, test_idx in [(f's{args.stringency_level_of_holdout}_fold1', [0, 1], 2),
                                                    (f's{args.stringency_level_of_holdout}_fold2', [0, 2], 1),
                                                    (f's{args.stringency_level_of_holdout}_fold3', [1, 2], 0)]:
        train_and_valid_pdbs = np.concatenate([pdb_splits[i] for i in train_and_val_idxs])
        test_pdbids = pdb_splits[test_idx]

        train_and_valid_data = df[df['#Pdb'].isin(train_and_valid_pdbs)]
        test_data = df[df['#Pdb'].isin(test_pdbids)]

        # split train_and_valid_data into train and valid
        pdbids = train_and_valid_data['pdbid'].unique()
        np.random.shuffle(pdbids)
        num_valid_pdbs = int(PERC_VALID_PDBIDS * len(pdbids))
        valid_pdbids = pdbids[:num_valid_pdbs]
        train_pdbids = pdbids[num_valid_pdbs:]
        train_data = train_and_valid_data[train_and_valid_data['pdbid'].isin(train_pdbids)]
        valid_data = train_and_valid_data[train_and_valid_data['pdbid'].isin(valid_pdbids)]

        os.makedirs(fold_name, exist_ok=True)
        train_data.to_csv(f'{fold_name}/train_targets.csv', index=False)
        valid_data.to_csv(f'{fold_name}/valid_targets.csv', index=False)
        test_data.to_csv(f'{fold_name}/test_targets.csv', index=False)

        assert np.sum(~np.isfinite(train_data['score'])) == 0
        assert np.sum(~np.isfinite(valid_data['score'])) == 0
        assert np.sum(~np.isfinite(test_data['score'])) == 0
        




