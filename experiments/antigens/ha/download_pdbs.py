

import os

# download pdbid from rcsb

pdbid = '7VDF'
os.system(f'wget https://files.rcsb.org/download/{pdbid}.pdb -O pdbs/{pdbid}.pdb')

