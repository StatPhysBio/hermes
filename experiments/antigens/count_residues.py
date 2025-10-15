#!/usr/bin/env python3

from Bio.PDB import PDBParser
import sys

def count_protein_residues(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Standard protein residue names
    aa_resnames = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL'
    }

    residues = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in aa_resnames:
                    # Each residue is uniquely identified by (chain_id, resid)
                    residues.add((chain.id, residue.id[1]))

    return len(residues)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_residues.py <pdbfile>")
        sys.exit(1)

    pdb_file = sys.argv[1]
    n_residues = count_protein_residues(pdb_file)
    print(f"Number of protein residues: {n_residues}")
