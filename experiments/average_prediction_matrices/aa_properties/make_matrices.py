
import os
import numpy as np
import pandas as pd

AMINOACIDS = 'CMFILVWYAGTSNQDEHRKP'

df_aa_prop = pd.read_csv('aminoacid_properties.csv')
df_aa_prop = df_aa_prop.drop(columns={'AA'})
df_aa_prop = df_aa_prop.set_index('abbr')

# make a pseudo-substitution matrix for each property
for prop in df_aa_prop.columns:
    properties = dict(zip(df_aa_prop.index.values, df_aa_prop[prop].values))
    matrix = np.full((20, 20), np.nan)
    for i, aa1 in enumerate(AMINOACIDS):
        for j, aa2 in enumerate(AMINOACIDS):
            if aa1 == aa2:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = -np.abs(properties[aa1] - properties[aa2])
    prop_df = pd.DataFrame(matrix, index=list(AMINOACIDS), columns=list(AMINOACIDS))
    prop_df.to_csv(f'matrices/{prop}.csv')

