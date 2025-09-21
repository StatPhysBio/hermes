

import os

# download pdbid from rcsb

pdbid = '5WB0'
os.system(f'wget https://files.rcsb.org/download/{pdbid}.pdb -O pdbs/{pdbid}.pdb')

