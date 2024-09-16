
import os
import numpy as np
import pandas as pd
import argparse

'''

Little utility script used to fix the .csv 'position' column so as to match the 'resnum' in the .pdb files

'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_code', type=str, required=True)
    parser.add_argument('--chain_id', type=str, required=True)
    parser.add_argument('--val', type=int, required=True)
    parser.add_argument('--starting_at', type=int, default=None)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--pos_col', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.filename)

    # change value of pos_col when pdbid and chainid are matched
    if args.starting_at is not None:
        df.loc[(df['pdb_code'] == args.pdb_code) & (df['chain_id'] == args.chain_id) & (df[args.pos_col] >= args.starting_at), args.pos_col] += args.val
    else:
        df.loc[(df['pdb_code'] == args.pdb_code) & (df['chain_id'] == args.chain_id), args.pos_col] += args.val

    df.to_csv(args.filename, index=False)

