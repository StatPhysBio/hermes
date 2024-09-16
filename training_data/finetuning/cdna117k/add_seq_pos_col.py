
import os
import numpy as np
import pandas as pd

from Bio import pairwise2
from Bio.Seq import Seq

from protein_holography_web.utils.protein_naming import aa_to_one_letter

if __name__ == '__main__':

    # for filename in ['cdna117K_aligned.csv', 'T2837_aligned.csv']:
    for filename in ['T2837_aligned.csv']:

        df = pd.read_csv(filename)

        dfs = []
        for i, group_df in df.groupby(['pdb_code', 'chain_id']):
            pdbid = group_df.iloc[0]['pdb_code']
            chainid = group_df.iloc[0]['chain_id']

            sequence = group_df.iloc[0]['sequence']

            positions = []
            wt_aas = []
            for i, group in group_df.groupby('position'):
                positions.append(group.iloc[0]['position'])
                wt_aas.append(aa_to_one_letter[group.iloc[0]['wtAA']])
            sorted_indices = np.argsort(positions)
            positions = np.array(positions)[sorted_indices]
            wt_aas = np.array(wt_aas)[sorted_indices]
            
            # make sequence, with gaps if necessary
            sequence_in_data = ''
            for i, (wt_aa, pos) in enumerate(zip(wt_aas, positions)):
                if i == 0: # beginning, no previous comparison
                    sequence_in_data += wt_aa
                    prev_pos = pos
                else:
                    if pos - prev_pos > 1:
                        sequence_in_data += '-' * (pos - prev_pos - 1)
                    sequence_in_data += wt_aa
                    prev_pos = pos

            alignments = pairwise2.align.globalxx(Seq(sequence), Seq(sequence_in_data))

            if len(alignments) == 0:
                print(f'{pdbid}, {chainid}: no alignment')
                print()
                continue
            
            # select alignment with no gaps in between aminoacids
            # second sequence (seqB) is ALMOST always contained in the first one (seqA)
            # when it's not, have to do something about it too
            sequence_with_gaps = None
            aligned_sequence = None
            for alignment in alignments:

                if sequence == alignment.seqA:
                    if sequence_in_data in alignment.seqB:
                        # print(f'{pdbid}, {chainid}: {alignment.seqB}')
                        aligned_sequence = alignment.seqB
                else:
                    if sequence_in_data in alignment.seqB:
                        # print(f'{pdbid}, {chainid}: {alignment.seqA}')
                        # print(f'{pdbid}, {chainid}: {alignment.seqB}')
                        aligned_sequence = alignment.seqB
                        sequence_with_gaps = alignment.seqA
            # print()

            if aligned_sequence is None:
                print(f'{pdbid}, {chainid}: no alignment.')
                print(sequence)
                print()
                one_idxed_seq_pos = group_df['position'].tolist()

            else:

                if sequence_with_gaps is not None:
                    # adjust aligned_sequence so that we eliminate gaps that also occur in sequence_with_gaps
                    new_aligned_sequence = ''
                    for i, (aa, aa_with_gap) in enumerate(zip(aligned_sequence, sequence_with_gaps)):
                        if aa_with_gap != '-':
                            new_aligned_sequence += aa
                    aligned_sequence = new_aligned_sequence            

                # now extract sequence of non-gap positions (1-indexed)
                non_gap_positions = [idx+1 for idx, aa in enumerate(aligned_sequence) if aa != '-']

                # now need to expand the positions to all the mutation rows that have the same position
                # I'm going to assume here that "non_gap_positions" is in parallel with "positions"
                # also going to assume that positions in the dataframe are in ascending order, which I know to be true
                assert len(non_gap_positions) == len(positions)
                one_idxed_seq_pos = []
                for non_gap_pos, pos in zip(non_gap_positions, positions):
                    one_idxed_seq_pos.extend([non_gap_pos] * len(group_df[group_df['position'] == pos]))
                
                assert len(one_idxed_seq_pos) == len(group_df)
            
            group_df['one_idxed_seq_pos'] = one_idxed_seq_pos
            dfs.append(group_df)
        
        df = pd.concat(dfs)
        df.to_csv(filename[:-4] + '_with_seq_pos.csv', index=False)


        



