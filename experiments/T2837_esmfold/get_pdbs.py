
import os
import numpy as np
import pandas as pd

df = pd.read_csv('T2837_esmfold_ddg_experimental.csv')

pdb_dir = '/gscratch/spe/gvisan01/protein_holography-web/training_data/finetuning/stability_oracle_cdna117K_esmfold/pdbs'

pdbids = np.unique(df['pdbid'].values)

os.makedirs('pdbs', exist_ok=True)

for pdbid in pdbids:
    os.system(f'cp {pdb_dir}/{pdbid}.pdb pdbs/')


