
import os
import numpy as np
import pandas as pd
import json


df = pd.read_csv('T2837_ddg_experimental.csv')

translation_dict = {}

for i, group_df in df.groupby(['pdbid', 'chainid']):
    pdbid = group_df.iloc[0]['pdbid']
    chainid = group_df.iloc[0]['chainid']
    sequence = group_df.iloc[0]['sequence'] # should be unique for pdbid and chainid, if not then there's a bug in the data

    res_ids_resnums = [int(x) for x in group_df['position'].values]
    seq_resnums = [int(x) for x in group_df['one_idxed_seq_pos'].values]

    if pdbid not in translation_dict:
        translation_dict[pdbid] = {}
    
    if chainid not in translation_dict[pdbid]:
        translation_dict[pdbid][chainid] = {}
    
    translation_dict[pdbid][chainid]['seq'] = sequence
    translation_dict[pdbid][chainid]['pdb_resnum_to_seq_resnum'] = dict(zip(res_ids_resnums, seq_resnums))

with open('T2837_res_ids_resnums_to_seq_resnums.json', 'w') as f:
    json.dump(translation_dict, f, indent=4)

    


