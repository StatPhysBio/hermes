

'''

Change this a bunch:
# 1. Make user select what to return. Whether probabilities, logits, embeddings, or combinations of them. --> DONE
# 2. Make the output be a single .csv file with the res_ids and the requested data. Rows are sites --> DONE BUT NEED TO TEST
3. Make the .txt (multi-pdb) option *optionally* split the output by pdb file, and assign pdbid name to each output file.
# 4. Use different inference code. Namely, the inference code of TCRpMHC stuff
# 5. Change model_dir to be the model config name

'''

import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from scipy.special import softmax, log_softmax
from sklearn.metrics import accuracy_score

from hermes.inference.inference_hermes import predict_from_hdf5file, predict_from_pdbfile, load_hermes_models, get_zernikegrams_in_parallel
from hermes.utils.protein_naming import ind_to_ol_size, ol_to_ind_size

import argparse


def check_input_arguments(args):
    assert args.output_filepath.endswith('.csv'), '--output_filepath must be a ".csv" file.'
    assert args.request, 'At least one of --request must be specified.'
    assert args.hdf5_file or args.folder_with_pdbs, 'Either --hdf5_file or --folder_with_pdbs must be specified.'
    assert not (args.hdf5_file and args.folder_with_pdbs), 'Cannot specify both --hdf5_file and --folder_with_pdbs.'

def download_pdbfile(pdbid, folder_with_pdbs, verbose):
    # downloads from RCSB
    if verbose:
        silent_flag = ''
    else:
        silent_flag = '-s'
    os.system(f"curl {silent_flag} https://files.rcsb.org/download/{pdbid}.pdb -o {os.path.join(folder_with_pdbs, pdbid + '.pdb')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_version', type=str, required=True,
                        help='Name of HERMES model you want to use.')
    
    parser.add_argument('-hf', '--hdf5_file', type=str, default=None,
                        help='Path to an .hdf5 file containing zernikegrams and res_ids to run inference on.\n \
                              Cannot be specified together with --folder_with_pdbs.')
    
    parser.add_argument('-pd', '--folder_with_pdbs', type=str, default=None,
                        help='Directory containing PDB files to run inference on. Inference is run on all sites in the structure.\n \
                              Cannot be specified together with --hdf5_file.')
    
    parser.add_argument('-pn', '--file_with_pdbids_and_chains', type=str, default=None,
                        help='[Optional] Path to a .txt file containing pdbids and chains to run inference on.\n \
                              If not specified, and --folder_with_pdbs is specified, inference will be run on all sites in the structure.\n \
                              If specified, each line should be in the format "pdbid chain"; if chain is not specified for a given line, inference will be run on all chains in that structure.')
    
    parser.add_argument('-pp', '--parallelism', type=int, default=0,
                        help='If zero (default), pdb files are processed one by one. If greater than zero, pdb files are processed in parallel with specified parallelism (and number of cores available), by first generating zernikegrams in a temporary hdf5 file.')

    parser.add_argument('-o', '--output_filepath', type=str, required=True,
                        help='Must be a ".csv file". Embeddings will be saved separately, in a parallel array, with the same filename but with the extension "-embeddings.npy".')
    
    parser.add_argument('-r', '--request', nargs='+', type=str, default=['logits'], choices=['logprobas', 'probas', 'embeddings', 'logits'],
                        help='Which data to return. Can be a combination of "logprobas", "probas", "embeddings", and "logits".')
    
    parser.add_argument('-an', '--add_same_noise_level_as_training', type=int, default=0, choices=[0, 1],
                        help='1 for True, 0 for False. If True, will add the same noise level as was used during training. This is useful for debugging purposes. Default is False.')
    
    parser.add_argument('-el', '--ensemble_at_logits_level', default=1, type=int, choices=[0, 1],
                        help="1 for True, 0 for False. When computing probabilities and log-probabilities, ensembles the logits before computing the softmax, as opposed to ansembling the individual models' probabilities.\n \
                              There should not be a big difference, unless the ensembled models are trained very differently.")
    
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='Batch size for the model (number of sites). Higher batch sizes are faster, but may not fit in memory. Default is 512.')

    parser.add_argument('-v', '--verbose', type=int, default=0, choices=[0, 1],
                        help='0 for no, 1 for yes. Currently, "yes" will print out accuracy of the model on the data.')

    parser.add_argument('-lb', '--loading_bar', type=int, default=1, choices=[0, 1],
                        help='0 for no, 1 for yes.')
        
    args = parser.parse_args()


    check_input_arguments(args)

    trained_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', args.model_version)
    model_dir_list = [os.path.join(trained_models_path, model_rel_path) for model_rel_path in os.listdir(trained_models_path)]
    models, hparams, finetuning_params = load_hermes_models(model_dir_list)


    ## prepare header of csv file, initialize output dataframe and embeddings
    res_id_fields = np.array(['resname', 'pdb', 'chain', 'resnum', 'insertion_code', 'secondary_structure'])
    indices_of_res_ids = np.array([1, 2, 0, 3, 4]) # rearrange to put pdb in front, and remove secondary structure, here as single point of truth
    res_id_fields = res_id_fields[indices_of_res_ids]
    data_columns = []
    for request in args.request:
        if request == 'probas':
            data_columns.extend([f'proba_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))]) # len(ind_to_ol_size) == num aminoacids
        elif request == 'logprobas':
            data_columns.extend([f'logproba_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))])
        elif request == 'logits':
            data_columns.extend([f'logit_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))])
    columns = np.concatenate([res_id_fields, data_columns])
    df_out = pd.DataFrame(columns=columns)
    embeddings_out = [] if 'embeddings' in args.request else None


    def update_output(inference, df, embeddings=None):

        all_res_ids = inference['res_ids'].astype(str)
        all_res_ids = all_res_ids[:, indices_of_res_ids] # rearrange to put pdb in front, and remove secondary structure

        ## average the data of the ensemble
        inference['probabilities'] = np.mean(inference['probabilities'], axis=0)
        inference['logits'] = np.mean(inference['logits'], axis=0)
        inference['embeddings'] = np.mean(inference['embeddings'], axis=0)

        additional_data = []
        for request in args.request:
            if request == 'probas':
                if args.ensemble_at_logits_level:
                    additional_data.append(softmax(inference['logits'].astype(np.float64), axis=1))
                else:
                    additional_data.append(inference['probabilities'])
            elif request == 'logprobas':
                if args.ensemble_at_logits_level:
                    additional_data.append(log_softmax(inference['logits'].astype(np.float64), axis=1))
                else:
                    additional_data.append(np.log(inference['probabilities']))
            elif request == 'logits':
                additional_data.append(inference['logits'])
        
        if additional_data:
            additional_data = np.concatenate(additional_data, axis=1)
            data = np.concatenate([all_res_ids, additional_data], axis=1)
        else:
            data = all_res_ids

        df = pd.concat([df, pd.DataFrame(data, columns=columns)], axis=0)

        if embeddings is not None:
            if len(embeddings) == 0:
                embeddings = inference['embeddings']
            else:
                embeddings = np.concatenate([embeddings, inference['embeddings']], axis=0)
        
        return df, embeddings


    ## run inference
    if args.hdf5_file is not None:
        if args.verbose: print(f'Running inference on zernikegrams in the .hdf5 file: {args.hdf5_file}')
        inference = predict_from_hdf5file(args.hdf5_file, models, hparams, args.batch_size)

        if len(inference['best_indices'].shape) == 2:
            if args.verbose: print('Accuracy of first model in ensemble: %.3f' % accuracy_score(inference['targets'], inference['best_indices'][0, :]))
        else:
            if args.verbose: print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))

    elif args.folder_with_pdbs is not None:

        os.makedirs(args.folder_with_pdbs, exist_ok=True) # make it if it does not exist (i.e. if user wants to download all requested pdb files)

        if args.file_with_pdbids_and_chains is not None:
            pdb_files, chains = [], []
            with open(args.file_with_pdbids_and_chains, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    pdbid_and_chain = line.strip().split()
                    pdbid = pdbid_and_chain[0]
                    if len(pdbid_and_chain) == 1:
                        chain = None
                    elif len(pdbid_and_chain) == 2:
                        chain = pdbid_and_chain[1]
                    else:
                        raise ValueError('Each line in --file_with_pdbids_and_chains must be in the format "pdbid" or "pdbid chain"')
                    
                    pdbfile = os.path.join(args.folder_with_pdbs, pdbid + '.pdb')

                    if not os.path.exists(pdbfile):
                        download_pdbfile(pdbid, args.folder_with_pdbs, args.verbose)
                    
                    pdb_files.append(pdbfile)
                    chains.append(chain)
        else:
            pdb_files = [os.path.join(args.folder_with_pdbs, pdb) for pdb in os.listdir(args.folder_with_pdbs) if pdb.endswith('.pdb')]
            chains = [None for _ in pdb_files]

        if args.verbose: print(f'Running inference on {len(pdb_files)} pdb files found in: {args.folder_with_pdbs}')

        pdb_files_and_chains = zip(pdb_files, chains)
        
        if args.parallelism:

            if args.verbose: print(f'Running inference in parallel with parallelism: {args.parallelism}')

            zernikegrams_hdf5_file = get_zernikegrams_in_parallel(args.folder_with_pdbs, hparams, args.parallelism, pdb_files_and_chains=pdb_files_and_chains, add_same_noise_level_as_training=args.add_same_noise_level_as_training)

            inference = predict_from_hdf5file(zernikegrams_hdf5_file, models, hparams, args.batch_size)

            if len(inference['best_indices'].shape) == 2:
                if args.verbose: print('Accuracy of first model in ensemble: %.3f' % accuracy_score(inference['targets'], inference['best_indices'][0, :]))
            else:
                if args.verbose: print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))

            os.remove(zernikegrams_hdf5_file)
            
            df_out, embeddings_out = update_output(inference, df_out, embeddings_out)

        else:
            if args.loading_bar:
                pdb_files_and_chains = tqdm(zip(pdb_files, chains), total=len(pdb_files))
            
            for pdbfile, chain in pdb_files_and_chains:
                if args.verbose: print(f'Running inference on pdb file: {pdbfile}')

                try:
                    inference = predict_from_pdbfile(pdbfile, models, hparams, args.batch_size, chain=chain, add_same_noise_level_as_training=args.add_same_noise_level_as_training)
                except Exception as e:
                    print(f'Error running inference on pdb file: {pdbfile}')
                    print(f'Error message: {e}')
                    continue

                if len(inference['best_indices'].shape) == 2:
                    if args.verbose: print('Accuracy of first model in ensemble: %.3f' % accuracy_score(inference['targets'], inference['best_indices'][0, :]))
                else:
                    if args.verbose: print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))
                
                df_out, embeddings_out = update_output(inference, df_out, embeddings_out)

    else:
        raise ValueError('Either --hdf5_file or --folder_with_pdbs must be specified.')


    ## save output
    if args.verbose: print(f'Saving [residue ids, {", ".join([req for req in args.request if req != "embeddings"])}] output to: {args.output_filepath}')

    df_out.to_csv(args.output_filepath, index=False)
    if 'embeddings' in args.request:
        if args.verbose: print(f'Saving embeddings to: {args.output_filepath[:-4] + "-embeddings.npy"}')
        np.save(args.output_filepath[:-4] + '-embeddings.npy', embeddings_out)

    