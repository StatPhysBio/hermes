

import os

# download pdbid from rcsb

pdbid = '6VSB'
os.makedirs('pdbs', exist_ok=True)
os.system(f'wget https://files.rcsb.org/download/{pdbid}.pdb -O pdbs/{pdbid}.pdb')

