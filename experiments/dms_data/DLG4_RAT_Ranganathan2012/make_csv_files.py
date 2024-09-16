
import os
import pandas as pd

experiment_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

df = pd.read_csv(f'output_{experiment_name}.csv')

for pdbid, chain in zip(['1qt3', '5hdy'],
                        ['A', 'A']):
    df_copy = df.copy()
    df_copy['chain'] = [chain for _ in range(len(df))]
    df_copy['wt_pdb'] = [pdbid for _ in range(len(df))]

    df_copy.to_csv(f'output_{experiment_name}__{pdbid}.csv', index=False)

