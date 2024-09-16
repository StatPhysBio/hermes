
import os
import numpy as np
import pandas as pd

from Bio import pairwise2
from Bio.Seq import Seq

from protein_holography_web.utils.protein_naming import aa_to_one_letter

def evaluate_alignment(df, verbose, pos_col='one_idxed_seq_pos'):


    if verbose: print()

    good_pdbs = []
    bad_pdbs = []

    for i, group_df in df.groupby(['pdb_code', 'chain_id']):
        pdbid = group_df.iloc[0]['pdb_code']
        chainid = group_df.iloc[0]['chain_id']

        num_mismatches = 0

        positions = []
        wt_aas = []
        for i, group in group_df.groupby(pos_col):
            positions.append(group.iloc[0][pos_col])
            wt_aas.append(aa_to_one_letter[group.iloc[0]['wtAA']])
        sorted_indices = np.argsort(positions)
        positions = np.array(positions)[sorted_indices]
        wt_aas = np.array(wt_aas)[sorted_indices]

        sequence = group_df.iloc[0]['sequence']

        for i, (wt_aa, pos) in enumerate(zip(wt_aas, positions)):

            if pos - 1 >= len(sequence) or pos - 1 < 0:
                num_mismatches += 1
                continue

            if sequence[pos - 1] != wt_aa:
                num_mismatches += 1
                print(f'{pdbid}, {chainid}: {pos} {wt_aa} {sequence[pos - 1]}')


        if num_mismatches > 0:
            if verbose: print(f'{pdbid}, {chainid}: {num_mismatches}/{len(wt_aas)} mismatches')
            if verbose: print()
            bad_pdbs.append((pdbid, chainid))
        else:
            good_pdbs.append((pdbid, chainid))
    
    print('----------------------------')
    print(f'Good PDBs: {len(good_pdbs)}')
    print(f'Bad PDBs: {len(bad_pdbs)}')
    print('----------------------------')

    return good_pdbs, bad_pdbs


if __name__ == '__main__':


    good_pdbs, bad_pdbs = evaluate_alignment(pd.read_csv('cdna117K_aligned_with_seq_pos.csv'), verbose=True, pos_col='one_idxed_seq_pos')

    good_pdbs, bad_pdbs = evaluate_alignment(pd.read_csv('T2837_aligned_with_seq_pos.csv'), verbose=True, pos_col='one_idxed_seq_pos')

    # for i, (pdbid, chainid) in enumerate(bad_pdbs):
    #     print(i, pdbid, chainid)



