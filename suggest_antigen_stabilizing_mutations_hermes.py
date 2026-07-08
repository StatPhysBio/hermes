

import os
import numpy as np
import pandas as pd
import argparse

from hermes.utils.protein import get_residues_info
from hermes.inference import run_hermes_on_pdbfile_or_pyrosetta_pose


AMINOACIDS_NO_C = 'GPAVILMFYWSTNQRHKDE'
BASE_INFO_COLUMNS = ['pdb', 'chain', 'resname', 'resnum', 'insertion_code']
EXTRA_INFO_COLUMNS = ['sec_struc', 'sasa']
ALL_INFO_COLUMNS = BASE_INFO_COLUMNS + EXTRA_INFO_COLUMNS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True)
    parser.add_argument('--pre_pdbpath', type=str, required=True)
    parser.add_argument('--pre_chains', type=str, required=True, nargs='+')
    parser.add_argument('--post_pdbpath', type=str, default=None)
    parser.add_argument('--post_chains', type=str, default=None, nargs='+')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--num_mutations', type=int, default=80, help='number of mutations to suggest')
    parser.add_argument('--maximum_muts_same_site', type=int, default=3, help='maximum number of mutations to suggest per site')
    args = parser.parse_args()

    ## 1) bookkeeping

    os.makedirs(args.outdir, exist_ok=True)

    # create pdbs directory with pdbfiles
    pdbs_dir = os.path.join(args.outdir, 'pdbs')
    os.makedirs(pdbs_dir, exist_ok=True)

    # move files to it and update paths
    pre_pdbfile = os.path.join(pdbs_dir, args.pre_pdbpath.split('/')[-1])
    os.rename(args.pre_pdbpath, pre_pdbfile)
    if args.post_pdbpath is not None:
        post_pdbfile = os.path.join(pdbs_dir, args.post_pdbpath.split('/')[-1])
        os.rename(args.post_pdbpath, post_pdbfile)
    
    # extract pdbids
    pre_pdbid = args.pre_pdbpath.split('/')[-1][:-4]
    if args.post_pdbpath is not None:
        post_pdbid = args.post_pdbpath.split('/')[-1][:-4]
    

    ## 2) get structural info for pre and post fusion
    df_info = get_residues_info(pdbs_dir)


    ## 3) run HERMES on pre and post fusion
    print(f'Running HERMES on pre-fusion, pdbid {pre_pdbfile}...', flush=True)
    df_results_pre, _ = run_hermes_on_pdbfile_or_pyrosetta_pose(
        args.model_version,
        pre_pdbfile,
        chain_and_sites_list=args.pre_chains,
        request='logits',
        batch_size = 256, # can tweak this to optimize memory usage
    )
    df_base_pre = pd.merge(df_results_pre, df_info, how='inner', on=BASE_INFO_COLUMNS)
    df_base_pre = df_base_pre.loc[df_base_pre['pdb'] == pre_pdbid]
    df_base_pre = df_base_pre.loc[df_base_pre['chain'].isin(args.pre_chains)]
    if args.post_pdbpath is not None:
        print(f'Running HERMES on post-fusion, pdbid {post_pdbfile}...', flush=True)
        df_results_post, _ = run_hermes_on_pdbfile_or_pyrosetta_pose(
            args.model_version,
            post_pdbfile,
            chain_and_sites_list=args.post_chains,
            request='logits',
            batch_size = 256, # can tweak this to optimize memory usage
        )
        df_base_post = pd.merge(df_results_post, df_info, how='inner', on=BASE_INFO_COLUMNS)
        df_base_post = df_base_post.loc[df_base_post['pdb'] == post_pdbid]
        df_base_post = df_base_post.loc[df_base_post['chain'].isin(args.post_chains)]


    ## 4) suggest mutations in the pre-fusion core
    df_base_pre_core = df_base_pre.loc[df_base_pre['sasa'] <= 1]
    print(f'number of core residues in pre: {len(df_base_pre_core)}')
    # construct all mutations df
    df = []
    for i, row in df_base_pre_core.iterrows():

        resnum = row['resnum']
        wt_aa = row['resname']

        if wt_aa == 'C':
            continue

        wt_logit = row[f'logit_{wt_aa}']

        temp_rows = []
        temp_rows_random = []
        for mt_aa in AMINOACIDS_NO_C:
            if mt_aa == wt_aa: continue
            newrow = row[ALL_INFO_COLUMNS]
            mt_logit = row[f'logit_{mt_aa}']
            delta_logit = mt_logit - wt_logit
            newrow['mutation'] = f'{wt_aa}{resnum}{mt_aa}'
            newrow['delta_logit'] = delta_logit
            temp_rows.append(newrow)

        # only consider at most k mutations per site
        rows_to_add = sorted(temp_rows, key=lambda row: row['delta_logit'])[-args.maximum_muts_same_site:]
        
        df.extend(rows_to_add)
    
    df = pd.DataFrame(df)

    # display most positive mutations
    df = df.sort_values('delta_logit', ascending=False)
    df_good = df.iloc[:args.num_mutations]

    df_good = df_good[['pdb', 'chain', 'mutation', 'delta_logit']]
    df_good.to_csv(os.path.join(args.outdir, f'suggested_mutations__{args.model_version}__stab_pre_core__max_per_site_{args.maximum_muts_same_site}.csv'), index=False)
    print(f"num_unique_resnums: {len(np.unique(list(map(lambda x: x[1:-1], df_good['mutation'].values))))}")


    ## 5) suggest mutations that stabilize pre-fusion and destabilize post-fusion
    if args.post_pdbpath is not None:
        # isolate pdb and symmetric units
        df_list = []
        for pre_or_post, df_loc in [('pre', df_base_pre),
                                    ('post', df_base_post)]:

            # construct all mutations df
            df = []
            for i, row in df_loc.iterrows():

                resnum = row['resnum']
                wt_aa = row['resname']
                if wt_aa == 'C':
                    continue
                wt_logit = row[f'logit_{wt_aa}']

                rows = []
                for mt_aa in AMINOACIDS_NO_C:
                    if mt_aa == wt_aa: continue
                    newrow = row[EXTRA_INFO_COLUMNS]
                    newrow = newrow.rename({col: col + '_' + pre_or_post for col in EXTRA_INFO_COLUMNS})
                    mt_logit = row[f'logit_{mt_aa}']
                    delta_logit = mt_logit - wt_logit
                    newrow['mutation'] = f'{wt_aa}{resnum}{mt_aa}'
                    newrow[f'pdb_{pre_or_post}'] = row['pdb']
                    newrow[f'delta_logit_{pre_or_post}'] = delta_logit
                    rows.append(newrow)
                
                df.extend(rows)

            df_list.append(pd.DataFrame(df))
        
        df = pd.merge(*df_list, on=['mutation'])

        # only consider mutations that are more beneficial than wt in prefusiuon, and less beneficial than wt in postfusion
        df['delta_logit_pre__positive'] = df['delta_logit_pre'] > 0
        df['delta_logit_post__negative'] = df['delta_logit_post'] < 0
        df = df.loc[df['delta_logit_pre__positive'] & df['delta_logit_post__negative']]

        # display mutations with the most divergence, only allow max
        df['delta_logit_pre__minus__delta_logit_post'] = df['delta_logit_pre'] - df['delta_logit_post']

        df = df.sort_values('delta_logit_pre__minus__delta_logit_post', ascending=False)

        curr_num_muts = 0
        resnums_counts = {}
        rows_good = []
        for _, row in df.iterrows(): # this will go in ascending order

            if curr_num_muts == args.num_mutations:
                break
            
            resnum = int(row['mutation'][1:-1])

            curr_count = resnums_counts.get(resnum, 0)
            curr_count += 1
            resnums_counts[resnum] = curr_count

            if curr_count <= args.maximum_muts_same_site:
                rows_good.append(row)
                curr_num_muts += 1
            
        df_good = pd.DataFrame(rows_good)
        df_good.to_csv(os.path.join(args.outdir, f'suggested_mutations__{args.model_version}__stab_pre_destab_post__max_per_site_{args.maximum_muts_same_site}.csv'), index=False)
        print(f"num_unique_resnums: {len(np.unique(list(map(lambda x: x[1:-1], df_good['mutation'].values))))}")

