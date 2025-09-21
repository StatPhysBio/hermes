
import numpy as np
import pandas as pd

df_preds = pd.read_csv('ThermoMPNN_Megascale-test_raw_preds.csv')

df_targets = pd.read_csv('../../test_targets.csv')


preds_scores = df_preds['ddG_true'].to_numpy()
target_scores = df_targets['score'].to_numpy()

# round the predcition to 1e-5
preds_scores = np.round(preds_scores, 3)
target_scores = np.round(target_scores, 3)

print(np.allclose(preds_scores, target_scores))

print(preds_scores[:5])
print(target_scores[:5])




df_targets['neg_ddg_pred'] = -df_preds['ddG_pred']
df_targets.to_csv('test_targets-thermompnn-use_mt_structure=0.csv', index=False)




