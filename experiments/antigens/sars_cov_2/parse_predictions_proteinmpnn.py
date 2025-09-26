
import os
import numpy as np
import pandas as pd


for model in ['proteinmpnn_v_48_002', 'proteinmpnn_v_48_030']:

    df = pd.read_csv(f'./results/{model}/zero_shot_predictions/mutations_all-num_seq_per_target=10-use_mt_structure=0.csv')

    df['position'] = df['mutation'].apply(lambda x: int(x[1:-1]))

    # label,mutation,chain,pdb
    final_df = {'label': [], 'mutation': [], 'chain': [], 'pdb': [], 'delta_logit': [], 'rank_wt': [], 'rank_mt': []}

    # group by position and chain
    for (position, chain), df_pos_chain in df.groupby(['position', 'chain']):

        aa_wt = df_pos_chain.loc[df_pos_chain['desired_one'] == 1]['mutation'].values[0][0]
        aa_mt = df_pos_chain.loc[df_pos_chain['desired_one'] == 1]['mutation'].values[0][-1]

        aas = df_pos_chain['mutation'].apply(lambda x: x[-1])
        assert len(aas) == 20, f'Expected 20 amino acids, got {len(aas)}'

        logit_values = df_pos_chain['log_p_mt'].values
        sorted_indices = np.argsort(logit_values)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(logit_values)) + 1
        rank_wt = ranks[np.where(aas == aa_wt)[0][0]]
        rank_mt = ranks[np.where(aas == aa_mt)[0][0]]

        delta_logit = logit_values[np.where(aas == aa_mt)[0][0]] - logit_values[np.where(aas == aa_wt)[0][0]]
        assert np.allclose(delta_logit, df_pos_chain.loc[df_pos_chain['desired_one'] == 1]['log_p_mt__minus__log_p_wt'].values[0])

        final_df['label'].append(df_pos_chain['label'].values[0])
        final_df['mutation'].append(f'{aa_wt}{position}{aa_mt}')
        final_df['chain'].append(chain)
        final_df['pdb'].append(df_pos_chain['pdb'].values[0])
        final_df['delta_logit'].append(delta_logit)
        final_df['rank_wt'].append(rank_wt)
        final_df['rank_mt'].append(rank_mt)
    
    final_df = pd.DataFrame(final_df)
    os.makedirs('./results', exist_ok=True)
    final_df.to_csv(f'./results/{model}.csv', index=False)




