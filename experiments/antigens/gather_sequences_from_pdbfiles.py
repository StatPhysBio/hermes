

import os
from tqdm import tqdm

from Bio.PDB import PDBParser, is_aa

import argparse

aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

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
        prev_num = 0
        
        for i, (resname, resnum) in enumerate(zip(resnames, resnums)):
            gap = resnum - prev_num - 1  # How many residues missing?
            if gap > 0:
                sequence += 'X' * gap  # Insert gaps
            sequence += resname
            prev_num = resnum
        
        seq_dict[chain_id] = sequence
    
    return seq_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and build sequences from PDB files.")
    parser.add_argument("--pdb_dir", type=str, help="Directory containing PDB files.")
    parser.add_argument("--output_fasta", type=str, default=None, help="Output FASTA file to save sequences.")
    args = parser.parse_args()

    pdb_dir = args.pdb_dir

    with open(args.output_fasta, 'w+') as fasta_out:

        for pdb_file in tqdm(os.listdir(pdb_dir), total=len(os.listdir(pdb_dir))):
            if not pdb_file.endswith('.pdb'):
                continue
            
            pdb_path = os.path.join(pdb_dir, pdb_file)
            chain_info = extract_chain_residue_info(pdb_path)
            sequences = build_chain_sequences(chain_info)

            for chain_id, seq in sequences.items():
                header = f">{pdb_file[:-4]}_{chain_id}"
                fasta_out.write(f"{header}\n{seq}\n")
    

