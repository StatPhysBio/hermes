import os, sys
import numpy as np
import gzip, pickle
import json
from tqdm import tqdm
from hermes.inference.inference_hermes import get_channels
from zernikegrams import get_zernikegrams_from_pdbfile

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_version', type=str, required=True,
                        help='HCNN model name. The script will take the hparams from this model and use them to preprocess the protein structures accoridng to the way the model was pre-trained.')
    
    parser.add_argument('-n', '--add_noise', type=int, default=0, choices=[0, 1],
                        help='If model_version is a model with noise, and this is toggled, the same amount of noise as the model was trained with will be added to the structures. \
                              Note that this will create multiple zernikegram files with different noise seed. If model_version is a model without noise, this argument will be ignored.')
    
    parser.add_argument('-pd', '--pdb_dir', type=str, required=True,
                        help='The directory containing the pdb files to process. The script will process all pdbs in this directory.')
    
    parser.add_argument('-pl', '--pdb_list', type=str, default=None,
                        help='A .txt file containing a list of pdbids to process. Each line should contain a pdbid. If not specified, the script will process all pdbs in the pdb_dir.')
    
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='The directory to save the zernikegrams to. The zernikegrams will be saved in a file named after the model version. \
                              If adding noise to structures is requested, a suffix "_seed={noise_seed}" will be added to each filename.')
    
    parser.add_argument('-pr', '--pdb_to_residues_file', type=str, default=None,
                        help='A file containing a dictionary mapping pdbids to a list of residues to process. \
                              Called `pdb_to_residues.json` in the examples in this repository. \
                              The file should be a json file with the following format: {"pdbid1": [res1, res2, ...], "pdbid2": [res1, res2, ...], ...}. \
                              Each residue should be a tuple of strings (chainid, resnum, icode). \
                              If a pdbid is not in the file, all residues in the pdb will be processed. \
                              If not specified, all residues in all pdbs will be processed.')
    
    args = parser.parse_args()

    ## get list of pdbs
    if args.pdb_list is not None:
        with open(args.pdb_list, 'r') as f:
            pdbs = f.read().splitlines()
    else:
        print('No pdb list provided, processing all pdbs in the pdb_dir...')
        pdbs = [pdb[:-4] for pdb in os.listdir(args.pdb_dir) if pdb.endswith('.pdb') and not pdb.startswith('.')] # hidden files are ignored

        
        
    ## make get_residues function
    if args.pdb_to_residues_file is not None:

        with open(args.pdb_to_residues_file, 'r') as f:
            pdb_to_residues = json.load(f)

        def get_requested_residues(np_protein):
            pdb = np_protein['pdb'].decode()
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            all_res_ids_info_we_care_about = res_ids[:, 2:5]
            if pdb in pdb_to_residues:
                requested_residues_tuples = pdb_to_residues[pdb]
                # requested_residues_tuples = [(chain.encode(), str(resnum).encode(), icode.encode()) for chain, resnum, icode in requested_residues_tuples] # for loop is over a smaller list, so it should be faster
                requested_residues_tuples = np.unique(np.array(requested_residues_tuples).astype(all_res_ids_info_we_care_about.dtype), axis=0)
                indices = np.where(np.isin(all_res_ids_info_we_care_about, requested_residues_tuples).all(axis=1))[0]
                return res_ids[indices]
            else:
                return res_ids
    else:
        get_requested_residues = None
    
    ## get hparams, assume they are the same for all models in the model version (the relevant haprams should be!!!)
    trained_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', args.model_version)
    model_dir_list = [os.path.join(trained_models_path, model_rel_path) for model_rel_path in os.listdir(trained_models_path)]
    with open(os.path.join(model_dir_list[0], 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    
    if hparams['noise'] > 0 and args.add_noise: # add noise to zernikegrams!
        hparams_list = []
        for model_dir in model_dir_list:
            with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
                hparams_list.append(json.load(f))
    else:
        hparams_list = [hparams]


    for hparams in tqdm(hparams_list):

        ## prepare arguments of the pipeline
        channels = get_channels(hparams['channels'])

        get_structural_info_kwargs = {'padded_length': None,
                                        'parser': hparams['parser'],
                                        'SASA': 'SASA' in channels,
                                        'charge': 'charge' in channels,
                                        'DSSP': False,
                                        'angles': False,
                                        'fix': True,
                                        'hydrogens': 'H' in channels,
                                        'extra_molecules': hparams['extra_molecules'],
                                        'multi_struct': 'warn'}

        if args.add_noise:
            add_noise_kwargs = {'noise': hparams['noise'],
                                'noise_seed': hparams['noise_seed']}
        else:
            add_noise_kwargs = None

        get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                    'remove_central_residue': hparams['remove_central_residue'],
                                    'remove_central_sidechain': hparams['remove_central_sidechain'],
                                    'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                    'get_residues': get_requested_residues}

        get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                                    'radial_func_mode': hparams['radial_func_mode'],
                                    'radial_func_max': hparams['radial_func_max'],
                                    'Lmax': hparams['lmax'],
                                    'channels': channels,
                                    'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                    'request_frame': False,
                                    'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                                    'rst_normalization': hparams['rst_normalization']}
        
        ## run pipeline!

        all_amino_acids = []
        all_pdbids = []
        all_chainids = []
        all_resnums = []
        all_zernikegrams = []

        bad_pdbs_count = 0
        for pdb in tqdm(pdbs):
            pdb_path = os.path.join(args.pdb_dir, pdb + '.pdb')

            # print('Processing', pdb, '...', flush=True)
            
            try:
                data = get_zernikegrams_from_pdbfile(pdb_path, get_structural_info_kwargs, get_neighborhoods_kwargs, get_zernikegrams_kwargs, add_noise_kwargs=add_noise_kwargs)
            except Exception as e:
                print(f"Error processing {pdb}: {e}", file=sys.stderr)
                bad_pdbs_count += 1
                continue
            
            res_ids = data['res_id']
            zernikegrams = data['zernikegram']

            # unpack the res_id to make it more readable/user-friendly

            amino_acids = np.array([res_id[0].decode('utf-8') for res_id in res_ids])
            pdbids =  np.array([res_id[1].decode('utf-8') for res_id in res_ids])
            chainids = np.array([res_id[2].decode('utf-8') for res_id in res_ids])
            resnums = np.array([int(res_id[3].decode('utf-8')) for res_id in res_ids])

            all_amino_acids.append(amino_acids)
            all_pdbids.append(pdbids)
            all_chainids.append(chainids)
            all_resnums.append(resnums)
            all_zernikegrams.append(zernikegrams)

            # print(resnums.shape)
            # print(resnums)
        
        print(f"Failed to process {bad_pdbs_count}/{len(pdbs)} pdbs.", flush=True)
        
        all_amino_acids = np.concatenate(all_amino_acids)
        all_pdbids = np.concatenate(all_pdbids)
        all_chainids = np.concatenate(all_chainids)
        all_resnums = np.concatenate(all_resnums)
        all_zernikegrams = np.vstack(all_zernikegrams)

        num_zernikegrams = all_zernikegrams.shape[0]
        

        ## make output_dir if it does not exist
        os.makedirs(args.output_dir, exist_ok=True)

        if hparams['noise'] > 0 and args.add_noise:
            output_name = args.model_version + f'_seed={add_noise_kwargs["noise_seed"]}'
        else:
            output_name = args.model_version

        ## save the zernikegrams
        with gzip.open(os.path.join(args.output_dir, output_name + '.npz.gz'), 'wb') as f:
            pickle.dump({'pdbid': all_pdbids, 'chainid': all_chainids, 'resnum': all_resnums, 'amino_acid': all_amino_acids, 'zernikegram': all_zernikegrams}, f)
        
        print(f"Saved {num_zernikegrams} zernikegrams for {len(pdbs)} pdb files in {os.path.join(args.output_dir, output_name + '.npz.gz')}")