
import pandas as pd

df = pd.read_csv('ssym_naive_inv_ddg_experimental.csv')

variants = df['variant'].values

new_variants = []
for variant in variants:
    wt = variant[0]
    mt = variant[-1]
    resnum = int(variant[1:-1])
    new_variants.append(f'{mt}{resnum}{wt}')

df['variant'] = new_variants

df['score'] = -df['score']

df.to_csv('ssym_naive_inv_ddg_experimental.csv', index=False)
