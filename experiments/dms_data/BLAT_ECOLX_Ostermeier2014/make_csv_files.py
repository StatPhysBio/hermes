
import os
import pandas as pd

experiment_name = 'BLAT_ECOLX_Ostermeier2014'

df = pd.read_csv(f'output_{experiment_name}.csv')

for pdbid, chain in zip(['1xpb', 'AF-P62593-F1-model_v4'],
                        ['A', 'A']):
    df_copy = df.copy()
    df_copy['chain'] = [chain for _ in range(len(df))]
    df_copy['wt_pdb'] = [pdbid for _ in range(len(df))]

    df_copy.to_csv(f'output_{experiment_name}__{pdbid}.csv', index=False)

