
import os
import numpy as np
import pandas as pd
import json

from hermes.utils.protein_naming import ol_to_ind_size, ind_to_ol_size


HERMES_MODELS = 'hermes_bp_000 hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_ros_ddg_st hermes_py_050 hermes_py_050_ft_ros_ddg_st'.split()


final_dict = {'Model': []}

for model_version in HERMES_MODELS:

    with open(f'results/{model_version}/test_ddg_rosetta-{model_version}-use_mt_structure=0_correlations.json', 'r') as f:
        data = json.load(f)
    
    final_dict['Model'].append(model_version)
    
    for pdb in data:
        if pdb == 'overall':
            continue
        
        if pdb not in final_dict:
            final_dict[pdb] = []
        
        final_dict[pdb].append(-data[pdb]['pearson'][0]) # negate it!
    
final_df = pd.DataFrame(final_dict)
final_df.to_csv('rosetta_ddg_results.csv', index=False)
    
    
        

    

