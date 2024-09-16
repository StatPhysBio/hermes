

import os
import pandas as pd

from protein_holography_web.utils.protein_naming import aa_to_one_letter

df = pd.read_csv('T2837_ddg_experimental.csv')

df['variant'] = [f'{aa_to_one_letter[from_aa]}{position}{aa_to_one_letter[to_aa]}' for from_aa, position, to_aa in zip(df['from'], df['position'], df['to'])]

df.to_csv('T2837_ddg_experimental.csv', index=False)
