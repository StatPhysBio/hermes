
import os
import numpy as np
import pandas as pd

experiment_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

df = pd.read_csv(f'output_{experiment_name}.csv')

for pdbid, chain in zip(['AF-P28482-F1-model_v4'],
                        ['A']):
    df_copy = df.copy()
    df_copy['chain'] = [chain for _ in range(len(df_copy))]
    df_copy['wt_pdb'] = [pdbid for _ in range(len(df_copy))]

    df_copy.to_csv(f'output_{experiment_name}__{pdbid}.csv', index=False)

for pdbid, chain in zip(['4g6n'],
                        ['A']):
    df_copy = df.copy()

    # shift resnums by minus two, remove the ones with non-positive resnums

    new_resnums = np.array([int(mutant[1:-1])-2 for mutant in df_copy['mutant']])
    df_copy = df_copy[new_resnums >= 1]
    new_resnums = new_resnums[new_resnums >= 1]

    new_mutants = [f'{mutant[0]}{new_resnum}{mutant[-1]}' for mutant, new_resnum in zip(df_copy['mutant'], new_resnums)]
    df_copy['mutant'] = new_mutants

    df_copy['chain'] = [chain for _ in range(len(df_copy))]
    df_copy['wt_pdb'] = [pdbid for _ in range(len(df_copy))]

    df_copy.to_csv(f'output_{experiment_name}__{pdbid}.csv', index=False)
