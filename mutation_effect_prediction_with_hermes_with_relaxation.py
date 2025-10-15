
import os, sys
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
import time

from scipy.special import log_softmax

from hermes.inference.inference_hermes import get_data_irreps, predict_from_zernikegrams, load_hermes_models, get_zernikegrams_from_pdbfile_and_regions

from hermes.utils.protein_naming import ind_to_ol_size, ol_to_ind_size
from hermes.pyrosetta_utils import PyrosettaPose

import argparse
from hermes.utils.argparse import *


def check_arguments(args):
    pass


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
                        help='Output directory for the results.')
    
    parser.add_argument('--num_relaxations_ensemble', type=int, default=2,
                        help='Number of relaxations to ensemble over. Runtime scales linearly with this number.')
    
    parser.add_argument('--nrepeats', type=int, default=1,
                        help='nrepeats argument for the pyrosetta fastrelax protocol.')
    
    parser.add_argument('--relax_wt', type=int, default=1, choices=[0, 1],
                        help='1 for True, 0 for False. If True, will relax the sites with wild-type amino-acid prior to scoring them. If False, a single score with the fixed wild-type structure will be used. Default is True.')

    parser.add_argument('--backbone_flexible_distance_threshold', type=float, default=None,
                        help='Distance threshold for backbone atoms to be considered flexible.')
    
    parser.add_argument('--sidechain_flexible_distance_threshold', type=float, default=12.0,
                        help='Distance threshold for sidechain atoms to be considered flexible.')


    # parser.add_argument('--add_same_noise_level_as_training', type=int, default=0, choices=[0, 1],
    #                     help='1 for True, 0 for False. If True, will add the same noise level as was used during training.  Default is False.')
    
    # parser.add_argument('--ensemble_with_noise', type=int, default=0, choices=[0, 1],
    #                     help='1 for True, 0 for False. If True, will run each model multiple times with some gaussian noise to the coordinates. Default is False.')

    parser.add_argument('--wt_pdb_column', type=str, required=True,
                        help='Column name with the wildtype PDB file')
    
    parser.add_argument('--mutant_column', type=str, required=True,
                        help='Column name with the mutation')
    
    parser.add_argument('--mutant_chain_column', type=str, required=True,
                        help='Column name with the chain the mutation occurs on')
    
    parser.add_argument('--mutant_split_symbol', type=str, default='|',
                        help='Symbol used to split multiple mutations.')

    parser.add_argument('-el', '--ensemble_at_logits_level', default=1, type=int, choices=[0, 1],
                        help="1 for True, 0 for False. When computing probabilities and log-probabilities, ensembles the logits before computing the softmax, as opposed to ansembling the individual models' probabilities. \
                              Should use 1 for fine-tuned models, since the logits are directly interpreted as deltaG values, but empirically there is little difference.")
    
    # parser.add_argument('--dont_run_inference', type=int, default=0, choices=[0, 1],
    #                     help='1 for True, 0 for False. If True, will not run inference, only parse the .npz files. Mainly intended for debugging purposes.')
    
    # parser.add_argument('--delete_inference_files', type=int, default=1, choices=[0, 1],
    #                     help='1 for True, 0 for False. If True, will delete the inference files after parsing them. Mainly intended for debugging purposes.')
    
    parser.add_argument('--num_splits', type=int, default=1, help='Number of splits to make in the CSV file. Useful for parallelizing the script.')

    parser.add_argument('--split_idx', type=int, default=0, help='Split index')

    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1],
                        help='1 for True, 0 for False. If True, will print more information to the console.')

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
    data_irreps, _ = get_data_irreps(hparams)


    # prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_dir = os.path.join(args.output_dir, f'{args.model_version}')
    os.makedirs(predictions_dir, exist_ok=True)
    split_predictions_dir = os.path.join(args.output_dir, f'{args.model_version}', 'split_files')
    os.makedirs(split_predictions_dir, exist_ok=True)

    # read csv file
    df = pd.read_csv(args.csv_file)

    # sort by pdbfile so that all pdbfiles are together. make it a stable deterministic sort
    df.sort_values(args.wt_pdb_column)

    assert args.num_splits > 0
    assert args.split_idx >= 0 and args.split_idx < args.num_splits
    if args.num_splits == 1:
        output_file_identifier = f'-{args.model_version}.csv'
    else:
        output_file_identifier = f'-{args.model_version}-split={args.split_idx}_{args.num_splits}.csv'
        indices = np.array_split(df.index, args.num_splits)
        df = df.loc[indices[args.split_idx]]
    
    csv_filename_out = os.path.basename(args.csv_file).split('/')[-1].replace('.csv', output_file_identifier)
    if not csv_filename_out.endswith('.csv'):
        csv_filename_out += '.csv'

    df_out_trace = []

    # group df by pdb, iterate over groups
    df_grouped = df.groupby(args.wt_pdb_column)
    for i_pdb, (pdb, df_pdb) in tqdm(enumerate(df_grouped), total=len(df_grouped)):

        # try:

        # load pdbfile
        pose = PyrosettaPose(verbose=args.verbose)
        pose.load_pdb(os.path.join(args.folder_with_pdbs, pdb + '.pdb'))

        zernikegrams_for_pdb = {'wt': {'res_ids': [], 'zernikegrams': []},
                                'mt': {'res_ids': [], 'zernikegrams': []}}
        
        has_been_processed_cache = {}

        # iterate over rows in the group
        for idx, row in df_pdb.iterrows():
            # get mutation
            chains = row[args.mutant_chain_column].split(args.mutant_split_symbol)
            mutations = row[args.mutant_column].split(args.mutant_split_symbol)
            assert len(mutations) == len(chains), 'Number of mutations and chains must be the same'
            assert len(mutations) == 1, 'Current code assumes one mutation only'
            
            wts = [m[0] for m in mutations]
            resnums = [int(m[1:-1]) for m in mutations]
            mts = [m[-1] for m in mutations]

            # set-up mutations dicts
            # assuming icode is empty for simplicity
            mutations_dict_wt = {(chain, resnum, ' ') : wt for chain, resnum, wt in zip(chains, resnums, wts)}
            mutations_dict_mt = {(chain, resnum, ' ') : mt for chain, resnum, mt in zip(chains, resnums, mts)}

            def stringify(iterable_of_iterables):
                string = []
                for iterable in iterable_of_iterables:
                    string.append('$'.join(iterable))
                return '&'.join(string)

            mutations_wt_str = stringify((chain, str(resnum), ' ', wt) for chain, resnum, wt in zip(chains, resnums, wts))
            mutations_mt_str = stringify((chain, str(resnum), ' ', mt) for chain, resnum, mt in zip(chains, resnums, mts))


            for key, mut_dict, mut_str in zip(['wt', 'mt'], 
                                                [mutations_dict_wt, mutations_dict_mt],
                                                [mutations_wt_str, mutations_mt_str]):
                
                
                if mut_str in has_been_processed_cache:
                    for k in range(args.num_relaxations_ensemble):
                        res_id, zgram = has_been_processed_cache[mut_str][k]
                        zernikegrams_for_pdb[key]['res_ids'].append(res_id)
                        zernikegrams_for_pdb[key]['zernikegrams'].append(zgram)
                    continue
                else:
                    has_been_processed_cache[mut_str] = []

                for k in range(args.num_relaxations_ensemble):
                
                    # mutate and relax wts (usually the wts correspond to what's already in the structure, but let's assume it doesn't and also just apply the relaxation)
                    if not(args.relax_wt and key == 'wt'):
                        pose.make_mutations_and_fastrelax_around_it(mut_dict,
                                                                    backbone_flexible_distance_threshold=args.backbone_flexible_distance_threshold,
                                                                    sidechain_flexible_distance_threshold=args.sidechain_flexible_distance_threshold,
                                                                    nrepeats=args.nrepeats)

                    # # save pdb temporarily
                    # temp_pdbfile = os.path.join(predictions_dir, f'{pdb}_{key}_{row[args.mutant_chain_column]}_{row[args.mutant_column]}.pdb')
                    # pose.save_pdb(temp_pdbfile)

                    # extract zernikegrams
                    # assuming icode is empty for simplicity
                    requested_regions = {'region': [(chain, resnum, ' ') for chain, resnum in zip(chains, resnums)]}
                    zgrams_dict_list_for_noise_levels = get_zernikegrams_from_pdbfile_and_regions(pose._pose, requested_regions, hparams) # can use pose._pose - and not save the temp_pdbfile -  if the model used is a pyrosetta model
                    
                    zgrams_dict = zgrams_dict_list_for_noise_levels[0] # just one noise level for now

                    # # delete pdbfile
                    # os.remove(temp_pdbfile)

                    pose.reset_pose() # very important!!!

                    zernikegrams_for_pdb[key]['res_ids'].append(zgrams_dict['res_id'])
                    zernikegrams_for_pdb[key]['zernikegrams'].append(zgrams_dict['zernikegram'])
                    has_been_processed_cache[mut_str].append((zgrams_dict['res_id'], zgrams_dict['zernikegram']))
            
        zernikegrams_for_pdb['wt']['res_ids'] = np.concatenate(zernikegrams_for_pdb['wt']['res_ids'], axis=0)
        zernikegrams_for_pdb['wt']['zernikegrams'] = np.concatenate(zernikegrams_for_pdb['wt']['zernikegrams'], axis=0)
        zernikegrams_for_pdb['mt']['res_ids'] = np.concatenate(zernikegrams_for_pdb['mt']['res_ids'], axis=0)
        zernikegrams_for_pdb['mt']['zernikegrams'] = np.concatenate(zernikegrams_for_pdb['mt']['zernikegrams'], axis=0)
        
        ensemble_predictions_dict_wt = predict_from_zernikegrams(zernikegrams_for_pdb['wt']['zernikegrams'],
                                                                zernikegrams_for_pdb['wt']['res_ids'],
                                                                models,
                                                                args.batch_size,
                                                                data_irreps)

        ensemble_predictions_dict_mt = predict_from_zernikegrams(zernikegrams_for_pdb['mt']['zernikegrams'],
                                                                zernikegrams_for_pdb['mt']['res_ids'],
                                                                models,
                                                                args.batch_size,
                                                                data_irreps)

        logits_wt_M_KS_20 = ensemble_predictions_dict_wt['logits']
        logits_mt_M_KS_20 = ensemble_predictions_dict_mt['logits']

        res_ids_wt_KS_6 = ensemble_predictions_dict_wt['res_ids']
        res_ids_mt_KS_6 = ensemble_predictions_dict_mt['res_ids']

        # rely on the sequentiality of the predictions, and group predictions sequentially
        KS = logits_wt_M_KS_20.shape[1]
        K = args.num_relaxations_ensemble
        S = KS // K
        logits_wt_M_S_20 = []
        logits_mt_M_S_20 = []
        res_ids_wt_S_6 = []
        res_ids_mt_S_6 = []
        for s in range(S):
            logits_wt_M_S_20.append(np.mean(logits_wt_M_KS_20[:, s*K:(s+1)*K, :], axis=1))
            logits_mt_M_S_20.append(np.mean(logits_mt_M_KS_20[:, s*K:(s+1)*K, :], axis=1))
            res_ids_wt_S_6.append(res_ids_wt_KS_6[s*K, :])
            res_ids_mt_S_6.append(res_ids_mt_KS_6[s*K, :])
        logits_wt_M_S_20 = np.stack(logits_wt_M_S_20, axis=1)
        logits_mt_M_S_20 = np.stack(logits_mt_M_S_20, axis=1)
        res_ids_wt_S_6 = np.stack(res_ids_wt_S_6, axis=0)
        res_ids_mt_S_6 = np.stack(res_ids_mt_S_6, axis=0)

        # ensemble logits across the 10 models
        logits_wt_S_20 = np.mean(logits_wt_M_S_20, axis=0)
        logits_mt_S_20 = np.mean(logits_mt_M_S_20, axis=0)

        # make res_id to logits dict
        chain_resnum_rescond__to__logits = {}
        for s in range(S):
            res_id_wt = res_ids_wt_S_6[s]
            chain_wt = res_id_wt[2].decode('utf-8')
            resnum_wt = int(res_id_wt[3].decode('utf-8'))
            rescond_wt = res_id_wt[0].decode('utf-8')
            chain_resnum_rescond__to__logits[(chain_wt, resnum_wt, rescond_wt)] = logits_wt_S_20[s]

            res_id_mt = res_ids_mt_S_6[s]
            chain_mt = res_id_mt[2].decode('utf-8')
            resnum_mt = int(res_id_wt[3].decode('utf-8'))
            rescond_mt = res_id_mt[0].decode('utf-8')
            chain_resnum_rescond__to__logits[(chain_mt, resnum_mt, rescond_mt)] = logits_mt_S_20[s]

        # now iterate over df_pdb again and populate it with predictions
        log_proba_wt_in_wt = []
        log_proba_mt_in_wt = []
        log_proba_wt_in_mt = []
        log_proba_mt_in_mt = []
        logit_wt_in_wt = []
        logit_mt_in_wt = []
        logit_wt_in_mt = []
        logit_mt_in_mt = []
        for idx, row in df_pdb.iterrows():
            
            chains = row[args.mutant_chain_column].split(args.mutant_split_symbol)
            mutations = row[args.mutant_column].split(args.mutant_split_symbol)
            assert len(mutations) == len(chains), 'Number of mutations and chains must be the same'
            assert len(mutations) == 1, 'Current code assumes one mutation only'

            chain = chains[0]
            mutation = mutations[0]
            wt = mutation[0]
            resnum = int(mutation[1:-1])
            mt = mutation[-1]

            wt_ind = ol_to_ind_size[wt]
            mt_ind = ol_to_ind_size[mt]

            logits_in_wt = chain_resnum_rescond__to__logits[(chain, resnum, wt)]
            logits_in_mt = chain_resnum_rescond__to__logits[(chain, resnum, mt)]
            log_probas_in_wt = log_softmax(logits_in_wt)
            log_probas_in_mt = log_softmax(logits_in_mt)

            log_proba_wt_in_wt.append(log_probas_in_wt[wt_ind])
            log_proba_mt_in_wt.append(log_probas_in_wt[mt_ind])
            log_proba_wt_in_mt.append(log_probas_in_mt[wt_ind])
            log_proba_mt_in_mt.append(log_probas_in_mt[mt_ind])
            logit_wt_in_wt.append(logits_in_wt[wt_ind])
            logit_mt_in_wt.append(logits_in_wt[mt_ind])
            logit_wt_in_mt.append(logits_in_mt[wt_ind])
            logit_mt_in_mt.append(logits_in_mt[mt_ind])
        
        df_pdb['log_proba_wt_in_wt'] = log_proba_wt_in_wt
        df_pdb['log_proba_mt_in_wt'] = log_proba_mt_in_wt
        df_pdb['log_proba_wt_in_mt'] = log_proba_wt_in_mt
        df_pdb['log_proba_mt_in_mt'] = log_proba_mt_in_mt
        df_pdb['log_proba_mt_in_mt__minus__log_proba_wt_in_wt'] = df_pdb['log_proba_mt_in_mt'] - df_pdb['log_proba_wt_in_wt']
        df_pdb['log_proba_mt_both__minus__log_proba_wt_both'] = (df_pdb['log_proba_mt_in_mt'] + df_pdb['log_proba_mt_in_wt']) - (df_pdb['log_proba_wt_in_wt'] + df_pdb['log_proba_wt_in_mt'])

        df_pdb['logit_wt_in_wt'] = logit_wt_in_wt
        df_pdb['logit_mt_in_wt'] = logit_mt_in_wt
        df_pdb['logit_wt_in_mt'] = logit_wt_in_mt
        df_pdb['logit_mt_in_mt'] = logit_mt_in_mt
        df_pdb['logit_mt_in_mt__minus__logit_wt_in_wt'] = df_pdb['logit_mt_in_mt'] - df_pdb['logit_wt_in_wt']
        df_pdb['logit_mt_both__minus__logit_wt_both'] = (df_pdb['logit_mt_in_mt'] + df_pdb['logit_mt_in_wt']) - (df_pdb['logit_wt_in_wt'] + df_pdb['logit_wt_in_mt'])
        
        # except Exception as e:
        #     print('Error processing PDB:', pdb, file=sys.stderr)
        #     print(e)

        #     df_pdb['log_proba_wt_in_wt'] = np.nan
        #     df_pdb['log_proba_mt_in_wt'] = np.nan
        #     df_pdb['log_proba_wt_in_mt'] = np.nan
        #     df_pdb['log_proba_mt_in_mt'] = np.nan
        #     df_pdb['log_proba_mt_in_mt__minus__log_proba_wt_in_wt'] = np.nan
        #     df_pdb['log_proba_mt_both__minus__log_proba_wt_both'] = np.nan

        #     df_pdb['logit_wt_in_wt'] = np.nan
        #     df_pdb['logit_mt_in_wt'] = np.nan
        #     df_pdb['logit_wt_in_mt'] = np.nan
        #     df_pdb['logit_mt_in_mt'] = np.nan
        #     df_pdb['logit_mt_in_mt__minus__logit_wt_in_wt'] = np.nan
        #     df_pdb['logit_mt_both__minus__logit_wt_both'] = np.nan


    
        df_out_trace.append(df_pdb)
    
    df_out = pd.concat(df_out_trace)

    if args.num_splits == 1:
        outdir = predictions_dir
    else:
        outdir = split_predictions_dir
    df_out.to_csv(os.path.join(outdir, csv_filename_out), index=False)



    
    

