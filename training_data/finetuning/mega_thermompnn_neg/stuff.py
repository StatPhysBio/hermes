
import pandas as pd

df = pd.read_csv('valid_targets.csv')

print(len(df['pdbid'].unique()))
