

import os

from Bio.PDB import PDBParser, is_aa
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from hermes.utils.protein_naming import aa_to_one_letter


def extract_chain_residue_info(pdb_file):
    """
    Extract residue information from a PDB file using Biopython.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.

    Returns
    -------
    dict
        A nested dictionary with the following structure:
        
        {
            'chain_id': {
                'resnames': [list of 1-letter residue codes],
                'resnums': [list of residue numbers],
                'inscodes': [list of insertion codes]
            },
            ...
        }

        - Residue names are 1-letter codes (e.g., 'A', 'G').
        - Residue numbers are integers.
        - Insertion codes are single characters ('' if absent).

    Notes
    -----
    - Only standard amino acid residues are included.
    - Waters, ligands, and other non-protein residues are excluded.
    - Empty chains (chains with no standard residues) are omitted.
    """
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    chain_dict = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            resnames = []
            resnums = []
            inscodes = []
            
            for residue in chain:
                # Only keep standard amino acids
                if not is_aa(residue, standard=True):
                    continue
                
                resname3 = residue.get_resname()   # 3-letter residue name
                resnum = residue.id[1]            # Residue number (integer)
                inscode = residue.id[2].strip()   # Insertion code ('' if none)
                
                # Safely convert to 1-letter code, fallback to 'X'
                try:
                    resname1 = aa_to_one_letter[resname3]
                except KeyError:
                    resname1 = 'X'  # Unknown or modified residue
                
                resnames.append(resname1)
                resnums.append(resnum)
                inscodes.append(inscode)
            
            if resnames:
                chain_dict[chain_id] = {
                    'resnames': resnames,
                    'resnums': resnums,
                    'inscodes': inscodes
                }
                
    return chain_dict


def build_chain_sequences(chain_dict):
    """
    Build chain sequences with gaps from extracted residue information.

    Parameters
    ----------
    chain_dict : dict
        Output dictionary from `extract_chain_residue_info()`.

    Returns
    -------
    dict
        Dictionary mapping chain IDs to sequences (str),
        where gaps are represented by '-' for missing residues.
    """
    seq_dict = {}

    for chain_id, resdata in chain_dict.items():
        resnames = resdata['resnames']
        resnums = resdata['resnums']
        
        if not resnames:
            seq_dict[chain_id] = ''
            continue
        
        sequence = ''
        # Start from first residue number
        prev_num = resnums[0] - 1
        
        for resname, resnum in zip(resnames, resnums):
            gap = resnum - prev_num - 1  # How many residues missing?
            if gap > 0:
                sequence += '-' * gap  # Insert gaps
            sequence += resname
            prev_num = resnum
        
        seq_dict[chain_id] = sequence
    
    return seq_dict


if __name__ == '__main__':

    pdbids = ['7VDF', '7QA4']

    # ## download the pdbs
    # for pdbid in pdbids:
    #     os.system(f'wget https://files.rcsb.org/download/{pdbid}.pdb -O {pdbid}.pdb')

    ## extract the sequences
    pdbid_to_chain_info = {}
    pdbid_to_seq = {}
    for pdbid in pdbids:
        pdb_file = f'{pdbid}.pdb'
        chain_info = extract_chain_residue_info(pdb_file)
        sequences = build_chain_sequences(chain_info)

        for chain_id, info in chain_info.items():
            pdbid_to_chain_info[pdbid] = info
            break # only keep chain A
        
        for chain_id, seq in sequences.items():
            pdbid_to_seq[pdbid] = seq
            break # only keep chain A
        
    # Perform global alignment (Needleman-Wunsch)
    alignments = pairwise2.align.globalxx(pdbid_to_seq['7VDF'], pdbid_to_seq['7QA4'])

    # Print the best alignment(s)
    for aln in alignments:
        print(format_alignment(*aln))
        break  # just show the top one


    seq_7VDF = '-STATLCLGHHAVPNGTLVKTITDDQIEVTNATELVQSSSTGKICNNPHRILDGIDCTLIDALLGDPHCDVFQNETWDLFVERSKAFSNCYPYDVPDYASLRSLVASSGTLEFITEGFTWTGVTQNGGSNACKRGPGSGFFSRLNWLTKSGSTYPVLNVTMPNNDNFDKLYIWGVHHPSTNQEQTSLYVQASGRVTVSTRRSQQTIIPNIGSRPWVRGLSSRISIYWTIVKPGDVLVINSNGNLIAPRGYFKMRTGKSSIMRSDAPIDTCISECITPNGSIPNDKPFQNVNKITYGACPKYVKQNTLKLATGMRNVPE-------------AIAGFIENGWEGMIDGWYGFRH-QNSEGTGQAADLKSTQAAIDQINGK-LNRVIEKT---NEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALE-NQHTIDLTDSEMNKLFEKTRRQLRENAEDMGNGCFKIYHKCDNACIESIRNGTYDHDVYRDEALNNRFQ-'
    seq_7QA4 = 'NSTATLCLGHHAVPNGTLVKTITDDQIEVTNATELVQSSSTGKICNNPHRILDGIDCTLIDALLGDPHCDVFQNETWDLFVERSKAFSNCYPYDVPDYASLRSLVASSGTLEFITEGFTWTGVTQNGGSNACKRGPGSGFFSRLNWLTKSGSTYPVLNVTMPNNDNFDKLYIWGVHHPSTNQEQTSLYVQASGRVTVSTRRSQQTIIPNIGSRPWVRGLSSRISIYWTIVKPGDVLVINSNGNLIAPRGYFKMRTGKSSIMRSDAPIDTCISECITPNGSIPNDKPFQNVNKITYGACPKYVKQNTLKLATGMRNVP---------RGLFGAIAGFIENGWEGMIDGWYGFR-WQNSEGTGQAADLKSTQAAIDQING-ILNRVI------NEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVAL-INQHTIDLTDSEMNKLFEKTRRQLRENAEDMGNGCFKIYHKCDNACIESIRNGTYDHDVYRDEALNNRFQI'

    # i = 352
    # print()
    # print(seq_7VDF[i])
    # print(seq_7QA4[i+1])
    # print()
    # print(seq_7VDF[i+26])
    # print(seq_7QA4[i+26+1])
    # print()
    # print(seq_7VDF[i+26+56])
    # print(seq_7QA4[i+26+56+1])
    # print()
    # print()

    i = 346
    print(pdbid_to_seq['7VDF'][i])
    print(pdbid_to_seq['7VDF'][i+25])
    print(pdbid_to_seq['7VDF'][i+25+52])
    print()

    j = 338
    print(pdbid_to_chain_info['7VDF']['resnums'][j])
    print(pdbid_to_chain_info['7VDF']['resnames'][j])
    print(pdbid_to_chain_info['7VDF']['resnums'][j+25])
    print(pdbid_to_chain_info['7VDF']['resnames'][j+25])
    print(pdbid_to_chain_info['7VDF']['resnums'][j+25+52])
    print(pdbid_to_chain_info['7VDF']['resnames'][j+25+52])
    

