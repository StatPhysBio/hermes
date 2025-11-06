
import os
import numpy as np
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_version', type=str, required=True,
                        help='Name of HERMES model you want to use.')
    
    parser.add_argument('--csv_file', type=str, required=True,
                        help='CSV file with the mutations to score. Must have columns for the wildtype PDB file, the mutation, and the chain the mutation occurs on. If use_mt_structure=1, must also have a column for the mutant PDB file.')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the results.')

    args = parser.parse_args()

    predictions_dir = os.path.join(args.output_dir, f'{args.model_version}')
    split_predictions_dir = os.path.join(args.output_dir, f'{args.model_version}', 'split_files')

    output_file_identifier = f'-{args.model_version}-use_mt_structure=1.csv' # for legacy reasons
    csv_filename_out = os.path.basename(args.csv_file).split('/')[-1].replace('.csv', output_file_identifier)
    if not csv_filename_out.endswith('.csv'):
        csv_filename_out += '.csv'
    
    dfs = [pd.read_csv(os.path.join(split_predictions_dir, filename)) for filename in os.listdir(split_predictions_dir)]

    df_out = pd.concat(dfs)

    df_out['logit_mt_in_wt__minus__logit_wt_in_wt'] = df_out['logit_mt_in_wt'] - df_out['logit_wt_in_wt']
    df_out['logit_mt_in_mt__minus__logit_wt_in_mt'] = df_out['logit_mt_in_mt'] - df_out['logit_wt_in_mt']

    df_out.to_csv(os.path.join(predictions_dir, csv_filename_out))
