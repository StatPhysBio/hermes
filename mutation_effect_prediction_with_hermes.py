

'''

Only will run HERMES on sites specified in the CSV file, as opposed to ALL sites in the protein.

Currently, HCNN gets called once per (PDB, chain) tuple. This is potentially still slow.
TODO to make it faster: make it so that HCNN can be called once per PDB, and then the results can be filtered by chain and resnum. Would have to be careful with sorting chaini and resnum but it shouldn't be a big problem.
TODO to make it even faster: collect zernikegrams from multiple PDBs in a single batch, feed them into HCNN in fewer calls.
    - This would require a bunch of restructuring and and writing of new code, but it's doable
    - Unclear that 

'''

import os, sys
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request

from scipy.special import log_softmax

from hermes.inference.inference_hermes import predict_from_pdbfile, load_hermes_models

from hermes.utils.dms_plots import dms_scatter_plot

from hermes.utils.protein_naming import ind_to_ol_size, ol_to_ind_size

import argparse
from hermes.utils.argparse import *


def check_arguments(args):
    if args.use_mt_structure: assert args.mt_pdb_column is not None, 'Must specify --mt_pdb_column if use_mt_structure=1.'


def make_filename(model_version, pdb, chain, resnums):
    return f"{model_version}__{pdb}__{chain}__{','.join([str(x) for x in resnums])}"

def parse_filename(name):
    model_version, pdb, chain, resnums = name.strip('.npz').split('__')
    resnums = [int(x) for x in resnums.split(',')]
    return model_version, pdb, chain, resnums

def get_file_that_matches_specs(inference_dir, model_version, pdb, chain, resnum):
    candidate_files = glob.glob(os.path.join(inference_dir, f"{model_version}__{pdb}__{chain}__*.npz"))
    for file in candidate_files:
        curr_resnums = parse_filename(os.path.basename(file))[3]
        if resnum in curr_resnums:
            return file

# @profile
import stopit

@stopit.threading_timeoutable(60*10) # stop it if it takes more than 10 minutes
def make_prediction(output_dir, pdbdir, chain, pdb, resnums, model_version, models, hparams, finetuning_hparams, sequence_pdb_alignment_json, embeddings_cache, batch_size, ensemble_at_logits_level, add_same_noise_level_as_training):

    # ## do not make predictions if they already exist (nice if some error or timehout happened on some PDB)
    # if os.path.exists(os.path.join(output_dir, f"{make_filename(model_version, pdb, chain, resnums)}.npz")):
    #     print(f'Predictions already exist for {pdb} {chain} {resnums}.', flush=True)
    #     return

    # filter out resnum duplicates
    # the "compute_zgrams_only_for_requested_regions" procedure implicitly sorts the residues by resnum, so keep them always sorted to make sure these resnums match the order of the predictions
    resnums = sorted(list(set(resnums)))

    ## assuming icode is ' ' for now!!
    region_ids = [(chain, resnum, ' ') for resnum in resnums]
    print('Region IDs:', region_ids)

    requested_regions = {'region': region_ids}
    try:
        ensemble_predictions_dict = predict_from_pdbfile(os.path.join(pdbdir, f'{pdb}.pdb'), models, hparams, batch_size, finetuning_hparams=finetuning_hparams, sequence_pdb_alignment_json=sequence_pdb_alignment_json, embeddings_cache_file=embeddings_cache, regions=requested_regions, add_same_noise_level_as_training=add_same_noise_level_as_training)
    except Exception as e:
        print(f'Error making predictions for {pdb} {chain} {resnums}: {e}')
        return
    
    ensemble_predictions_dict = ensemble_predictions_dict['region']
    pes = np.mean(ensemble_predictions_dict['logits'], axis=0)
    if ensemble_at_logits_level:
        logps = log_softmax(pes, axis=1)
    else:
        logps = np.log(np.mean(ensemble_predictions_dict['probabilities'], axis=0))

    resnums_in_res_ids = ensemble_predictions_dict['res_ids'][:, 3].astype(int)
    if not np.all(resnums_in_res_ids == np.array(resnums)):
        print(f"WARNING: Resnums in res_ids do not match the requested resnums. Some resnums were not computed.")
        if pes.shape[0] != resnums_in_res_ids.shape[0]:
            print('Shape mismatch in pes. Super weird. Skipping.')
            return
    
    wt_aas_in_res_ids = ensemble_predictions_dict['res_ids'][:, 0]

    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f"{make_filename(model_version, pdb, chain, resnums)}.npz"),
                pes=pes,
                logps=logps,
                resnums=resnums_in_res_ids, # using the correct resnums, which match the predictions for sure
                wt_aas=wt_aas_in_res_ids,
                chain=chain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_version', type=str, required=True,
                        help='Name of HERMES model you want to use.')
    
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    
    parser.add_argument('--csv_file', type=str, required=True,
                        help='CSV file with the mutations to score. Must have columns for the wildtype PDB file, the mutation, and the chain the mutation occurs on. If use_mt_structure=1, must also have a column for the mutant PDB file.')

    parser.add_argument('--folder_with_pdbs', type=str, required=True,
                        help='Folder with the PDB files. Must contain the PDB files specified in the CSV file.')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the results. The directory will contain subdirectory "inference" with .npz files containing batch prediction information (can be discarded), \
                              and subdirectory "zero_shot_predictions" with the CSV file containing the results.')

    parser.add_argument('--add_same_noise_level_as_training', type=int, default=0, choices=[0, 1],
                        help='1 for True, 0 for False. If True, will add the same noise level as was used during training. This is useful for debugging purposes. Default is False.')

    parser.add_argument('--wt_pdb_column', type=str, required=True,
                        help='Column name with the wildtype PDB file')
    
    parser.add_argument('--mt_pdb_column', type=optional_str, default=None,
                        help='Column name with the mutant PDB file. Must specify it if use_mt_structure=1.')
    
    parser.add_argument('--mutant_column', type=str, required=True,
                        help='Column name with the mutation')
    
    parser.add_argument('--mutant_chain_column', type=str, required=True,
                        help='Column name with the chain the mutation occurs on')
    
    parser.add_argument('--mutant_split_symbol', type=str, default='|',
                        help='Symbol used to split multiple mutations.')

    parser.add_argument('--use_mt_structure', type=int, default=0,
                        help='0 for false, 1 for true. If toggled, compute logits for mutations on the corresponding mutated structure.')

    parser.add_argument('-el', '--ensemble_at_logits_level', default=1, type=int, choices=[0, 1],
                        help="1 for True, 0 for False. When computing probabilities and log-probabilities, ensembles the logits before computing the softmax, as opposed to ansembling the individual models' probabilities. \
                              Should use 1 for fine-tuned models, since the logits are directly interpreted as deltaG values, but empirically there is little difference.")
    
    parser.add_argument('--dont_run_inference', type=int, default=0, choices=[0, 1],
                        help='1 for True, 0 for False. If True, will not run inference, only parse the .npz files. Mainly intended for debugging purposes.')
    
    parser.add_argument('--delete_inference_files', type=int, default=1, choices=[0, 1],
                        help='1 for True, 0 for False. If True, will delete the inference files after parsing them. Mainly intended for debugging purposes.')
    
    parser.add_argument('--num_splits', type=int, default=1, help='Number of splits to make in the CSV file. Useful for parallelizing the script.')

    parser.add_argument('--split_idx', type=int, default=0, help='Split index')
    
    # These are only used for making a scatter plot. Useful for a quick visualization. Do not need to use them
    parser.add_argument('--dms_column', type=optional_str, nargs='+', default=None,
                        help='Column with the values you want to correlater with (e.g. ddg)')

    # These are preliminary arguments that should not be used.
    parser.add_argument('--sequence_pdb_alignment_json', type=str, default=None,
                        help='(PRELIMINARY) JSON file with the PDB-to-sequence alignment of resnums. Not needed if not using a joint HERMES-ESM model.')
    
    parser.add_argument('--embeddings_cache', type=str, default=None,
                        help='(PRELIMINARY) Path to the embeddings cache. Not needed if not using a joint HERMES-ESM model.')

    args = parser.parse_args()

    check_arguments(args)


    '''
    1) Parse CSV file. Collect all mutations belonging to the same PDB file.
    2) Score the mutations using HERMES. Crucially, only compute zernikegrams and predictions for the sites that are present in the CSV file.
    3) Parse the .npz files and save the results in a new CSV file.
    '''

    trained_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', args.model_version)
    model_dir_list = [os.path.join(trained_models_path, model_rel_path) for model_rel_path in os.listdir(trained_models_path)]
    models, hparams, finetuning_hparams = load_hermes_models(model_dir_list)


    # prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    inference_dir = os.path.join(args.output_dir, f'{args.model_version}', 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    predictions_dir = os.path.join(args.output_dir, f'{args.model_version}')
    os.makedirs(predictions_dir, exist_ok=True)

    # read csv file
    df = pd.read_csv(args.csv_file)

    assert args.num_splits > 0
    assert args.split_idx >= 0 and args.split_idx < args.num_splits
    if args.num_splits == 1:
        output_file_identifier = f'-{args.model_version}-use_mt_structure={args.use_mt_structure}.csv'
    else:
        output_file_identifier = f'-{args.model_version}-use_mt_structure={args.use_mt_structure}-split={args.split_idx}_{args.num_splits}.csv'
        indices = np.array_split(df.index, args.num_splits)
        df = df.loc[indices[args.split_idx]]
    
    csv_filename_out = os.path.basename(args.csv_file).split('/')[-1].replace('.csv', output_file_identifier)
    if not csv_filename_out.endswith('.csv'):
        csv_filename_out += '.csv'
    
    # if os.path.exists(os.path.join(predictions_dir, csv_filename_out.replace('.csv', '_correlations.json'))):
    #     exit(0)


    # get pdbs
    pdbs = set()
    if args.use_mt_structure:
        for pdb in df[args.mt_pdb_column]:
            pdbs.add(pdb)
    for pdb in df[args.wt_pdb_column]:
        pdbs.add(pdb)
    pdbs = list(pdbs)

    # download necessary if they are not found in folder
    os.makedirs(args.folder_with_pdbs, exist_ok=True) # make folder if does not exist, so it doesn't have to be made beforehand if it's empty
    pdbs_in_folder = [file[:-4] for file in os.listdir(args.folder_with_pdbs)]
    print(pdbs_in_folder)
    # pdbs_to_download = set(pdbs) - set(pdbs_in_folder)
    # if len(pdbs_to_download) > 0:
    #     print(f'Downloading the following PDBs: {pdbs_to_download}')
    #     for pdb in pdbs_to_download:
    #         try:
    #             urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb}.pdb', os.path.join(args.folder_with_pdbs, f'{pdb}.pdb'))
    #         except Exception as e:
    #             print(f'Error downloading {pdb}: {e}')
    #             continue
    
    # get chains and resnums for each PDB (single call per individual chain)
    pdb_to_chain_to_resnums = {}
    for i, row in df.iterrows():
        if args.use_mt_structure:
            row_pdbs = [row[args.wt_pdb_column], row[args.mt_pdb_column]]
        else:
            row_pdbs = [row[args.wt_pdb_column]]
        
        if not isinstance(row[args.mutant_chain_column], str) or not isinstance(row[args.mutant_column], str):
            # sometimes the chain and mutant columns are NaN, in which case we skip the row
            continue
        
        chains = row[args.mutant_chain_column].split(args.mutant_split_symbol)
        mutants = row[args.mutant_column].split(args.mutant_split_symbol)
        assert len(chains) == len(mutants)

        for chain, mutant in zip(chains, mutants):
            resnum = int(mutant[1:-1])
            for pdb in row_pdbs:
                if pdb not in pdb_to_chain_to_resnums:
                    pdb_to_chain_to_resnums[pdb] = {}
                if chain not in pdb_to_chain_to_resnums[pdb]:
                    pdb_to_chain_to_resnums[pdb][chain] = set()
                
                pdb_to_chain_to_resnums[pdb][chain].add(resnum)
    
    print(pdb_to_chain_to_resnums)
        
    ## run inference!!
    from time import time
    start = time()
    if not args.dont_run_inference:
        for pdb in tqdm(pdb_to_chain_to_resnums):
            for chain, resnums in pdb_to_chain_to_resnums[pdb].items():
                ## split resnums into chunks of at most 20 otherwise I might get "File name too long" error
                resnums = list(sorted(list(resnums)))
                CHUNK_SIZE = 20
                resnums_chunks = [resnums[i:i+CHUNK_SIZE] for i in range(0, len(resnums), CHUNK_SIZE)]
                for res_chunk in resnums_chunks:
                    # print(f'Running inference for {pdb} {chain} {res_chunk}')
                    make_prediction(inference_dir, args.folder_with_pdbs, chain, pdb, res_chunk, args.model_version, models, hparams, finetuning_hparams, args.sequence_pdb_alignment_json, args.embeddings_cache, args.batch_size, args.ensemble_at_logits_level, args.add_same_noise_level_as_training)
    end = time()
    print(f'Inference took {end - start} seconds')

    ## parse the .npz files and save the results as a new column
    pe_wt_all = []
    pe_mt_all = []
    log_proba_wt_all = []
    log_proba_mt_all = []
    print('Parsing .npz files...')
    for i, row in tqdm(df.iterrows(), total=len(df)):

        wt_pdb = row[args.wt_pdb_column]

        if not isinstance(row[args.mutant_chain_column], str) or not isinstance(row[args.mutant_column], str):
            # sometimes the chain and mutant columns are NaN, in which case we skip the row
            print(f'WARNING: No file found for {wt_pdb} {chain} {resnum}.')
            pe_wt_all.append(np.nan)
            pe_mt_all.append(np.nan)
            log_proba_wt_all.append(np.nan)
            log_proba_mt_all.append(np.nan)
            continue

        chains = row[args.mutant_chain_column].split(args.mutant_split_symbol)
        mutants = row[args.mutant_column].split(args.mutant_split_symbol)
        assert len(chains) == len(mutants)

        ## average results across multiple mutations
        temp_pe_wt = []
        temp_pe_mt = []
        temp_log_proba_wt = []
        temp_log_proba_mt = []
        for chain, mutant in zip(chains, mutants):
            aa_wt = mutant[0]
            aa_mt = mutant[-1]
            resnum = int(mutant[1:-1])

            if aa_mt == 'X':
                print(f'WARNING: Mutant amino-acid is X at {wt_pdb} {chain} {resnum}.', file=sys.stderr)
                temp_pe_wt.append(np.nan)
                temp_pe_mt.append(np.nan)
                temp_log_proba_wt.append(np.nan)
                temp_log_proba_mt.append(np.nan)
                continue

            wt_file = get_file_that_matches_specs(inference_dir, args.model_version, wt_pdb, chain, resnum)
            if wt_file is None:
                print(f'WARNING: No file found for {wt_pdb} {chain} {resnum}.')
                temp_pe_wt.append(np.nan)
                temp_pe_mt.append(np.nan)
                temp_log_proba_wt.append(np.nan)
                temp_log_proba_mt.append(np.nan)
                continue

            wt_data = np.load(wt_file)

            if args.use_mt_structure:
                mt_pdb = row[args.mt_pdb_column]
                mt_file = get_file_that_matches_specs(inference_dir, args.model_version, mt_pdb, chain, resnum)
                if mt_file is None:
                    print(f'WARNING: No file found for {mt_pdb} {chain} {resnum}.')
                    temp_pe_wt.append(np.nan)
                    temp_pe_mt.append(np.nan)
                    temp_log_proba_wt.append(np.nan)
                    temp_log_proba_mt.append(np.nan)
                    continue
                mt_data = np.load(mt_file)
            else:
                mt_pdb = wt_pdb
                mt_data = wt_data
            
            if wt_data['pes'].shape[0] != wt_data['resnums'].shape[0]:
                print(f"WARNING: Shape mismatch in wt_data. pes is {wt_data['pes'].shape[0]}, resnums is {wt_data['resnums'].shape[0]}. Skipping {wt_pdb} {chain} {resnum}.")
                temp_pe_wt.append(np.nan)
                temp_pe_mt.append(np.nan)
                temp_log_proba_wt.append(np.nan)
                temp_log_proba_mt.append(np.nan)
                continue
            
            try:
                np.where(wt_data['resnums'] == resnum)[0][0]
            except IndexError:
                print(f"WARNING: Resnum {resnum} not found in wt_data. Skipping {wt_pdb} {chain} {resnum}.")
                temp_pe_wt.append(np.nan)
                temp_pe_mt.append(np.nan)
                temp_log_proba_wt.append(np.nan)
                temp_log_proba_mt.append(np.nan)
                continue
                
            # check that the wildtype amino-acids as they are in the csv file match the amino-acids in the structure
            aa_wt_in_structure = wt_data['wt_aas'][np.where(wt_data['resnums'] == resnum)[0][0]].decode('utf-8')
            # assert aa_wt_in_structure == aa_wt, f'Wildtype residue mismatch! {aa_wt_in_structure} != {aa_wt} at {wt_pdb} {chain} {resnum}'
            if aa_wt_in_structure != aa_wt:
                print(f'\nWARNING: Wildtype residue mismatch! {aa_wt_in_structure} != {aa_wt} at {wt_pdb} {chain} {resnum}\n', file=sys.stderr)
                # temp_pe_wt.append(np.nan)
                # temp_pe_mt.append(np.nan)
                # temp_log_proba_wt.append(np.nan)
                # temp_log_proba_mt.append(np.nan)
                # continue

            if args.use_mt_structure:
                aa_mt_in_structure = mt_data['wt_aas'][np.where(mt_data['resnums'] == resnum)[0][0]].decode('utf-8')
                # assert aa_mt_in_structure == aa_mt, f'Mutant residue mismatch! {aa_mt_in_structure} != {aa_mt} at {wt_pdb} {chain} {resnum}'
                if aa_mt_in_structure != aa_mt:
                    print(f'\nWARNING: Mutant residue mismatch! {aa_mt_in_structure} != {aa_mt} at {mt_pdb} {chain} {resnum}\n', file=sys.stderr)
                    # temp_pe_wt.append(np.nan)
                    # temp_pe_mt.append(np.nan)
                    # temp_log_proba_wt.append(np.nan)
                    # temp_log_proba_mt.append(np.nan)
                    # continue

            wt_pe = wt_data['pes'][np.where(wt_data['resnums'] == resnum)[0][0], ol_to_ind_size[aa_wt]]
            mt_pe = mt_data['pes'][np.where(mt_data['resnums'] == resnum)[0][0], ol_to_ind_size[aa_mt]]

            wt_logp = wt_data['logps'][np.where(wt_data['resnums'] == resnum)[0][0], ol_to_ind_size[aa_wt]]
            mt_logp = mt_data['logps'][np.where(mt_data['resnums'] == resnum)[0][0], ol_to_ind_size[aa_mt]]

            temp_pe_wt.append(wt_pe)
            temp_pe_mt.append(mt_pe)
            temp_log_proba_wt.append(wt_logp)
            temp_log_proba_mt.append(mt_logp)
        
        pe_wt_all.append(np.mean(temp_pe_wt))
        pe_mt_all.append(np.mean(temp_pe_mt))
        log_proba_wt_all.append(np.mean(temp_log_proba_wt))
        log_proba_mt_all.append(np.mean(temp_log_proba_mt))

    df['pe_wt'] = pe_wt_all
    df['pe_mt'] = pe_mt_all
    df['log_proba_wt'] = log_proba_wt_all
    df['log_proba_mt'] = log_proba_mt_all
    df['log_proba_mt__minus__log_proba_wt'] = df['log_proba_mt'] - df['log_proba_wt']


    df.to_csv(os.path.join(predictions_dir, csv_filename_out), index=False)

    if args.delete_inference_files:
        # remove the temporary inference folder
        os.system(f'rm -r {inference_dir}')


    if args.dms_column is not None:
        for dms_column in args.dms_column:
            dms_column = dms_column.strip('[ ').strip(' ]')
            dms_filepath = os.path.join(predictions_dir, f'correlation_with_{dms_column}-{csv_filename_out.replace(".csv", ".png")}')
            (pearson_r, pearson_pval), (spearman_r, spearman_pval) = dms_scatter_plot(df,
                                                                                  dms_column, 'log_proba_mt__minus__log_proba_wt',
                                                                                  dms_label=None, pred_label=r'H-CNN Prediction',
                                                                                  filename = dms_filepath)
        
        





