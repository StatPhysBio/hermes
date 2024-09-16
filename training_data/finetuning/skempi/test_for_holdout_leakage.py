
import os
import numpy as np
import pandas as pd

for fold_name in ['fold1', 'fold2', 'fold3']:
    train_and_valid_data = pd.concat([pd.read_csv(f'{fold_name}/train_targets.csv'), pd.read_csv(f'{fold_name}/valid_targets.csv')])
    test_data = pd.read_csv(f'{fold_name}/test_targets.csv')

    train_and_valid_holdout_pdbids = train_and_valid_data['pdbid'].unique()

    

