
import os
import pandas as pd


## save pdb files
os.makedirs('./pdbs', exist_ok=True)

pdb_dirs = ['../Ssym_dir/pdbs', '../Ssym_inv/pdbs']

for pdbdir in pdb_dirs:
    for pdbfile in os.listdir(pdbdir):
        os.system(f"cp {os.path.join(pdbdir, pdbfile)} ./pdbs")

## make dataframe
dir_df = pd.read_csv('../Ssym_dir/ssym_dir_ddg_experimental.csv')
inv_df = pd.read_csv('../Ssym_inv/ssym_inv_ddg_experimental.csv')

dir_df['mt_pdbid'] = inv_df['pdbid']

dir_df.to_csv('ssym_dir_with_mut_structs_ddg_experimental.csv', index=False)

