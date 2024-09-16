
import pandas as pd
import json

df = pd.concat([pd.read_csv('cdna117K_aligned_with_seq_pos.csv'), pd.read_csv('T2837_aligned_with_seq_pos.csv')])

adict = {}
for i in range(len(df)):
    adict[df['pdb_code'].values[i]] = df['sequence'].values[i]

with open('pdb_to_sequence_dict.json', 'w') as f:
    json.dump(adict, f, indent=4)
