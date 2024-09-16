
import os
import pandas as pd

experiment_name = 'BG_STRSQ_hmmerbit'

df = pd.read_csv(f'output_{experiment_name}.csv')

for pdbid, chain in zip(['1gnx', '1gon', 'rank_1_model_4_ptm_seed_0_unrelaxed'],
                        ['A', 'A', 'A']):
    df_copy = df.copy()
    df_copy['chain'] = [chain for _ in range(len(df))]
    df_copy['wt_pdb'] = [pdbid for _ in range(len(df))]

    df_copy.to_csv(f'output_{experiment_name}__{pdbid}.csv', index=False)

