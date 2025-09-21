

import numpy as np
import pandas as pd

df = pd.read_csv('aminoacid_properties.csv')

sorting_indices = np.argsort(df['V22'])

sorted_aa = df['abbr'][sorting_indices].values
sorted_vdw = df['V22'][sorting_indices].values

print(sorted_aa)
print(sorted_vdw)
