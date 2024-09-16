
import json

with open('pdb_to_residues.json', 'r+') as f:
    pdb_to_residues = json.load(f)

pdbs = list(pdb_to_residues.keys())
with open('pdbs.txt', 'w') as f:
    for pdb in pdbs:
        f.write(pdb + '\n')
