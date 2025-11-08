
import os, sys
import json
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from e3nn import o3
from scipy.special import softmax, log_softmax
import torch
import gzip, pickle
import glob

from typing import *

from zernikegrams.preprocessors.pdbs import PDBPreprocessor
import h5py
import hdf5plugin
import tempfile
from rich.progress import Progress

from hermes.cg_coefficients import get_w3j_coefficients
from hermes.models import SO3_ConvNet, CGNet, SO3_ConvNetPlusEmbeddings

# from hermes.protein_processing.pipeline import get_zernikegrams_from_pdbfile
from zernikegrams import get_zernikegrams_from_pdbfile
from hermes.utils.data import ZernikegramsDataset, ZernikegramsAndEmbeddingsDataset
from hermes.utils.protein_naming import ol_to_ind_size, ind_to_ol_size


def initialize_df(request):
    ## prepare header of csv file, initialize output dataframe and embeddings
    res_id_fields = np.array(['resname', 'pdb', 'chain', 'resnum', 'insertion_code', 'secondary_structure'])
    indices_of_res_ids = np.array([1, 2, 0, 3, 4]) # rearrange to put pdb in front, and remove secondary structure, here as single point of truth
    res_id_fields = res_id_fields[indices_of_res_ids]
    data_columns = []
    for request in request:
        if request == 'probas':
            data_columns.extend([f'proba_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))]) # len(ind_to_ol_size) == num aminoacids
        elif request == 'logprobas':
            data_columns.extend([f'logproba_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))])
        elif request == 'logits':
            data_columns.extend([f'logit_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))])
    columns = np.concatenate([res_id_fields, data_columns])
    df = pd.DataFrame(columns=columns)
    embeddings = [] if 'embeddings' in request else None
    return df, embeddings, indices_of_res_ids, columns

def update_output(inference, request, df, indices_of_res_ids, columns, ensemble_at_logits_level=True, embeddings=None):

    all_res_ids = inference['res_ids'].astype(str)
    all_res_ids = all_res_ids[:, indices_of_res_ids] # rearrange to put pdb in front, and remove secondary structure

    ## average the data of the ensemble
    inference['probabilities'] = np.mean(inference['probabilities'], axis=0)
    inference['logits'] = np.mean(inference['logits'], axis=0)
    inference['embeddings'] = np.mean(inference['embeddings'], axis=0)

    additional_data = []
    for request in request:
        if request == 'probas':
            if ensemble_at_logits_level:
                additional_data.append(softmax(inference['logits'].astype(np.float64), axis=1))
            else:
                additional_data.append(inference['probabilities'])
        elif request == 'logprobas':
            if ensemble_at_logits_level:
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

def fix_types_in_dataframe(df):
    if 'logit_A' in df.columns:
        for aa in ol_to_ind_size:
            df[f'logit_{aa}'] = df[f'logit_{aa}'].astype(float)
    if 'logproba_A' in df.columns:
        for aa in ol_to_ind_size:
            df[f'logproba_{aa}'] = df[f'logproba_{aa}'].astype(float)
    if 'proba_A' in df.columns:
        for aa in ol_to_ind_size:
            df[f'proba_{aa}'] = df[f'proba_{aa}'].astype(float)
    
    df['resnum'] = df['resnum'].astype(int)

    return df


def convert_predictions_results_to_standard_dataframe(results: Dict[str, Any], request: Union[str, List[str]], ensemble_at_logits_level=True):
    '''
    results is intended to be *for a specific region*
    '''
    df, embeddings, indices_of_res_ids, columns = initialize_df(request)
    df, embeddings = update_output(results, request, df, indices_of_res_ids, columns, ensemble_at_logits_level=ensemble_at_logits_level, embeddings=embeddings)
    df = fix_types_in_dataframe(df)
    return df, embeddings


def get_num_components(Lmax, ks, keep_zeros, mode, channels):
    num_components = 0
    if mode == "ns":
        for l in range(Lmax + 1):
            if keep_zeros:
                num_components += (
                    np.count_nonzero(np.array(ks) >= l) * len(channels) * (2 * l + 1)
                )
            else:
                num_components += (
                    np.count_nonzero(
                        np.logical_and(np.array(ks) >= l, (np.array(ks) - l) % 2 == 0)
                    )
                    * len(channels)
                    * (2 * l + 1)
                )

    if mode == "ks":
        for l in range(Lmax + 1):
            num_components += len(ks) * len(channels) * (2 * l + 1)
    
    return num_components


def get_channels(channels_str):

    if channels_str == 'dlpacker':
        channels = ['C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif channels_str == 'dlpacker_plus':
        channels = ['CAlpha', 'C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif channels_str == 'AAs':
        channels = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                    b'S', b'T', b'W', b'Y', b'V', b'G', "all other AAs"]
    else:
        channels = channels_str.split(',')
    
    return channels


def get_data_irreps(hparams):

    channels = get_channels(hparams['channels'])
    
    # construct data irreps from hparams
    mul_by_l = []
    if hparams['radial_func_mode'] == 'ks':
        for l in range(hparams['lmax'] + 1):
            mul_by_l.append((hparams['radial_func_max']+1) * len(channels))
    
    elif hparams['radial_func_mode'] == 'ns':
        ns = np.arange(hparams['radial_func_max'] + 1)
        for l in range(hparams['lmax'] + 1):
            # assuming not keeping zeros... because why would we?
            mul_by_l.append(np.count_nonzero(np.logical_and(np.array(ns) >= l, (np.array(ns) - l) % 2 == 0)) * len(channels))

    data_irreps = o3.Irreps('+'.join([f'{mul}x{l}e' for l, mul in enumerate(mul_by_l)]))
    ls_indices = np.concatenate([[l]*(2*l+1) for l in data_irreps.ls])

    return data_irreps, ls_indices


def load_hermes_models(model_dirs: List[str]):

    '''
    Assume that all models have the same hparams and same data_irreps
    '''
    
    models = []
    for i, model_dir in enumerate(model_dirs):

        # get hparams from json
        with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)
        
        if os.path.exists(os.path.join(model_dir, 'finetuning_params.json')):
            with open(os.path.join(model_dir, 'finetuning_params.json'), 'r') as f:
                finetuning_params = json.load(f)
        else:
            finetuning_params = None
        
        data_irreps, ls_indices = get_data_irreps(hparams)

        # setup device
        if i == 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('Running on %s.' % (device))

        # load w3j coefficients
        w3j_matrices = get_w3j_coefficients()
        for key in w3j_matrices:
            # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
            if device is not None:
                w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
            else:
                w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
            w3j_matrices[key].requires_grad = False
        
        # load model and pre-trained weights
        if finetuning_params is not None and 'embeddings_model_version' in finetuning_params:
            if hparams['model_type'] == 'cgnet':
                raise NotImplementedError()
            elif hparams['model_type'] == 'so3_convnet':
                model = SO3_ConvNetPlusEmbeddings(finetuning_params, data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
            else:
                raise NotImplementedError()
        else:
            if hparams['model_type'] == 'cgnet':
                model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
            elif hparams['model_type'] == 'so3_convnet':
                if finetuning_params is None or 'model_confidence_handling' not in finetuning_params or finetuning_params['model_confidence_handling'] == 'frequentist':
                    model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
                elif finetuning_params['model_confidence_handling'] == 'bayesian':
                    # model = SO3_ConvNet_WithExtraPredictor(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
                    raise NotImplementedError('model_confidence_handling "bayesian" not yet implemented')
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        
        model.load_state_dict(torch.load(os.path.join(model_dir, 'lowest_valid_loss_model.pt'), map_location=torch.device(device)))
        model.eval()
        models.append(model)
    
    return models, hparams, finetuning_params # assume that all models have the same hparams (except for random seed, which does not really matter), same data_irreps, and same finetuning_hparams


def get_zernikegrams_from_pdbfile_wrapper(*args, **kwargs):
    try:
        ret_value = get_zernikegrams_from_pdbfile(*args, **kwargs)
    except Exception as e:
        print(f"Error in parsing pdbfile: {e}")
        ret_value = None
    return ret_value

def get_zernikegrams_in_parallel(folder_with_pdbs: str,
                                 hparams: Dict,
                                 parallelism: int,
                                 pdb_files_and_chains: Optional[List[Tuple[str, str]]] = None,
                                 add_same_noise_level_as_training: bool = False,
                                 hdf5_name: Optional[str] = None):
    '''
    if hdf5_name is None, then a temporary file is created and returned
    '''
    
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

    if add_same_noise_level_as_training:
        add_noise_kwargs = {'noise': hparams['noise'],
                            'noise_seed': hparams['noise_seed']}
    else:
        add_noise_kwargs = None
    

    if pdb_files_and_chains is not None:
        pdb_list = []
        pdb_to_chain = {}
        for pdbpath, chain in list(pdb_files_and_chains):
            pdb = pdbpath.split('/')[-1][:-4]
            pdb_list.append(pdb)
            pdb_to_chain[pdb] = chain

        def get_residues_fn(np_protein):
            pdb = np_protein['pdb'].decode()
            chain = pdb_to_chain[pdb]
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            if chain is None: # no specified chain, return all residues
                return res_ids
            else:
                indices = np.where(np.isin(res_ids[:, 2], np.array([chain.encode()])))[0]
                return res_ids[indices]
    else:
        for pdbpath in glob.glob(folder_with_pdbs + '/*.pdb'):
            pdb = pdbpath.split('/')[-1][:-4]
            pdb_list.append(pdb)
        get_residues_fn = None


    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'remove_central_sidechain': hparams['remove_central_sidechain'],
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'get_residues': get_residues_fn}

    get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                                'radial_func_mode': hparams['radial_func_mode'],
                                'radial_func_max': hparams['radial_func_max'],
                                'Lmax': hparams['lmax'],
                                'channels': channels,
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'request_frame': False,
                                'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                                'rst_normalization': hparams['rst_normalization']}
    

    processor = PDBPreprocessor(pdb_list, folder_with_pdbs)

    L = np.max([5, processor.pdb_name_length])

    num_components = get_num_components(hparams['lmax'], np.arange(hparams['radial_func_max'] + 1), False, hparams['radial_func_mode'], channels)
    dt = np.dtype(
        [
            ("res_id", f"S{L}", (6,)),
            ("zernikegram", "f4", (num_components,)),
            ("label", "<i4"),
        ]
    )


    if hdf5_name is None:
        hdf5_file = tempfile.NamedTemporaryFile(delete=False)
        hdf5_name = hdf5_file.name
    else:
        pass # no need to do anything, this line here just for readibility


    with h5py.File(hdf5_name, "w") as f:
        f.create_dataset(
            'data',
            shape=(processor.size * 2000,), # assume an average of 2000 atoms per protein to start with, as that is the average single-chain length
            maxshape=(None,),
            dtype=dt,
            chunks=True,
            compression=hdf5plugin.LZ4(),
        )

    with Progress() as bar:
        task = bar.add_task("zernikegrams from multiple pdbs", total=processor.count())
        with h5py.File(hdf5_name, "r+") as f:
            n = 0
            for zgram_data_batch in processor.execute(
                callback=get_zernikegrams_from_pdbfile_wrapper,
                limit=None,
                params={
                    'get_structural_info_kwargs': get_structural_info_kwargs,
                    'get_neighborhoods_kwargs': get_neighborhoods_kwargs,
                    'get_zernikegrams_kwargs': get_zernikegrams_kwargs,
                    'add_noise_kwargs': add_noise_kwargs
                },
                parallelism=parallelism
            ):
                
                if zgram_data_batch is None: # failure happened
                    continue
                
                n_added_zgrams = zgram_data_batch['res_id'].shape[0]
                
                if n + n_added_zgrams > f["data"].shape[0]:
                    f["data"].resize((f["data"].shape[0] + n_added_zgrams*3,))
                
                for n_i in range(n_added_zgrams):
                    f["data"][n + n_i] = (zgram_data_batch['res_id'][n_i], zgram_data_batch['zernikegram'][n_i], zgram_data_batch['label'][n_i],)
                
                n += n_added_zgrams
                
                bar.update(task, advance=1)
            
            f["data"].resize((n,))

    return hdf5_name


def predict_from_hdf5file(hdf5_file: str,
                          models: List,
                          hparams: Dict,
                          batch_size: int,
                          regions: Optional[Dict[str, List[Tuple[str, int, str]]]] = None):
    '''
    NOTE: requested chains are already handled in the creation of the hdf5 file, so no need to worry about that here
    '''
    data_irreps, ls_indices = get_data_irreps(hparams)

    with h5py.File(hdf5_file, 'r') as f:
        zgrams_dict = {'res_id': f['data']['res_id'][:], 'zernikegram': f['data']['zernikegram'][:]}

    if regions is None: # return the predictions
        ensemble_predictions_dict = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps)
    else: # return the predictions for each region, in a dict indexed by region_name
        ensemble_predictions_dict = {}
        for region_name in regions:
            ensemble_predictions_dict[region_name] = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps, region=regions[region_name])
    
    return ensemble_predictions_dict


def get_zernikegrams_from_pdbfile_and_regions(pdb_file_or_pose: str, # or Pose
                                              regions: Dict[str, List[Tuple[str, int, str]]],
                                              hparams: Dict,
                                              add_same_noise_level_as_training: bool = False,
                                              ensemble_with_noise: bool = False) -> List[Dict]:

    # data_irreps, ls_indices = get_data_irreps(hparams)

    # this code template would be useful to limit the number of residues to compute zernikegrams - and do inference - for
    if regions is not None:
        def get_residues(np_protein):
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            all_res_ids_info_we_care_about = res_ids[:, 2:5]
            region_ids = []
            for region_name in regions:
                region_ids.extend(regions[region_name])
            region_ids = np.unique(np.array(region_ids).astype(all_res_ids_info_we_care_about.dtype), axis=0)
            indices = np.where(np.isin(all_res_ids_info_we_care_about, region_ids).all(axis=1))[0]
            return res_ids[indices]
    else:
        get_residues = None

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
    
    if add_same_noise_level_as_training:
        add_noise_kwargs_list = [{'noise': hparams['noise'],
                                  'noise_seed': hparams['noise_seed']}]
    elif ensemble_with_noise:
        add_noise_kwargs_list = [{'noise': 0.2, # 0.5 seems to be too much noise, it hinders performance
                                  'noise_seed': hparams['noise_seed'] + n} for n in range(5)] # 5 might be too little times
    else:
        add_noise_kwargs_list = [None]

    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'remove_central_sidechain': hparams['remove_central_sidechain'],
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'get_residues': get_residues}

    get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                               'radial_func_mode': hparams['radial_func_mode'],
                               'radial_func_max': hparams['radial_func_max'],
                               'Lmax': hparams['lmax'],
                               'channels': get_channels(hparams['channels']),
                               'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                               'request_frame': False,
                               'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                               'rst_normalization': hparams['rst_normalization']}
    
    zgrams_dict_list_for_noise_levels = []
    for add_noise_kwargs in add_noise_kwargs_list:
        zgrams_dict = get_zernikegrams_from_pdbfile(pdb_file_or_pose, get_structural_info_kwargs, get_neighborhoods_kwargs, get_zernikegrams_kwargs, add_noise_kwargs=add_noise_kwargs)
        zgrams_dict_list_for_noise_levels.append(zgrams_dict)
    
    return zgrams_dict_list_for_noise_levels

# @profile
def predict_from_pdbfile(pdb_file_or_pose: str, # or Pose
                          models: List,
                          hparams: Dict,
                          batch_size: int,
                          finetuning_hparams: Optional[Dict] = None,
                          sequence_pdb_alignment_json: Optional[str] = None,
                          embeddings_cache_file: Optional[str] = None,
                          add_same_noise_level_as_training: bool = False,
                          ensemble_with_noise: bool = False,
                          chain: Optional[str] = None,
                          regions: Optional[Dict[str, Union[str, List[Tuple[str, int, str]]]]] = None):

    if chain is not None and regions is not None:
        raise ValueError("Cannot specify both chain and regions")
    
    assert not (add_same_noise_level_as_training and ensemble_with_noise)

    data_irreps, ls_indices = get_data_irreps(hparams)

    # this code template would be useful to limit the number of residues to compute zernikegrams - and do inference - for
    if regions is not None:

        def get_residues(np_protein):
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            all_res_ids_info_we_care_about = res_ids[:, 2:5]
            region_ids = []
            for region_name in regions:
                if region_name != 'whole_chains':
                    region_ids.extend(regions[region_name])
            region_ids = np.unique(np.array(region_ids).astype(all_res_ids_info_we_care_about.dtype), axis=0)
            indices_single_sites = np.where(np.isin(all_res_ids_info_we_care_about, region_ids).all(axis=1))[0]

            indices = [indices_single_sites]

            # now consider the whole chains as well
            if 'whole_chains' in regions:
                for ch in regions['whole_chains']:
                    if not isinstance(ch, str):
                        raise ValueError(f"`whole_chains` as a region name must be reserved to chain identifiers, not lists of specific sites!")
                    indices.append(np.where(res_ids[:, 2] == ch.encode())[0])
            
            indices = np.concatenate(indices)

            return res_ids[indices]
        
    elif chain is not None:
        def get_residues(np_protein):
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            indices = np.where(res_ids[:, 2] == chain.encode())[0]
            return res_ids[indices]
        
    else:
        get_residues = None

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
    
    if add_same_noise_level_as_training:
        add_noise_kwargs_list = [{'noise': hparams['noise'],
                                  'noise_seed': hparams['noise_seed']}]
    elif ensemble_with_noise:
        add_noise_kwargs_list = [{'noise': 0.2, # 0.5 seems to be too much noise, it hinders performance
                                  'noise_seed': hparams['noise_seed'] + n} for n in range(5)] # 5 might be too little times
    else:
        add_noise_kwargs_list = [None]

    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'remove_central_sidechain': hparams['remove_central_sidechain'],
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'get_residues': get_residues}

    get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                               'radial_func_mode': hparams['radial_func_mode'],
                               'radial_func_max': hparams['radial_func_max'],
                               'Lmax': hparams['lmax'],
                               'channels': get_channels(hparams['channels']),
                               'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                               'request_frame': False,
                               'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                               'rst_normalization': hparams['rst_normalization']}
    
    ensemble_predictions_dict_list = []
    for add_noise_kwargs in add_noise_kwargs_list:
    
        zgrams_dict = get_zernikegrams_from_pdbfile(pdb_file_or_pose, get_structural_info_kwargs, get_neighborhoods_kwargs, get_zernikegrams_kwargs, add_noise_kwargs=add_noise_kwargs)

        np_embeddings = None
        if finetuning_hparams is not None:
            if 'embeddings_model_version' in finetuning_hparams:
                pdb_name = pdb_file.split('/')[-1][:-4]
                np_embeddings = get_esm_embeddings(pdb_name, zgrams_dict['res_id'], '_'.join(finetuning_hparams['embeddings_model_version'].split('_')[:4]), sequence_pdb_alignment_json, embeddings_cache_file=embeddings_cache_file)
        
        if regions is None: # return the predictions
            ensemble_predictions_dict = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps, np_embeddings=np_embeddings)
        else: # return the predictions for each region, in a dict indexed by region_name
            ensemble_predictions_dict = {}
            for region_name in regions:
                ensemble_predictions_dict[region_name] = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps, np_embeddings=np_embeddings, region=regions[region_name])
        
        ensemble_predictions_dict_list.append(ensemble_predictions_dict)
    
    # print()
    # print(ensemble_predictions_dict_list[0])
    # print()
    
    noise_ensemble_predictions_dict = {}
    if regions is None:
        for key in ensemble_predictions_dict_list[0].keys():
            if key in {'res_ids', 'targets'}:
                noise_ensemble_predictions_dict[key] = ensemble_predictions_dict_list[0][key]
            elif ensemble_predictions_dict_list[0][key] is None:
                noise_ensemble_predictions_dict[key] = None
            else:
                noise_ensemble_predictions_dict[key] = np.concatenate([ensemble_predictions_dict[key] for ensemble_predictions_dict in ensemble_predictions_dict_list], axis=0)
    else:
        for region_name in regions:
            noise_ensemble_predictions_dict[region_name] = {}
            for key in ensemble_predictions_dict_list[0][region_name]:
                if key in {'res_ids', 'targets'}:
                    noise_ensemble_predictions_dict[region_name][key] = ensemble_predictions_dict_list[0][region_name][key]
                elif ensemble_predictions_dict_list[0][region_name][key] is None:
                    noise_ensemble_predictions_dict[region_name][key] = None
                else:
                    noise_ensemble_predictions_dict[region_name][key] = np.concatenate([ensemble_predictions_dict[region_name][key] for ensemble_predictions_dict in ensemble_predictions_dict_list], axis=0)
    
    return noise_ensemble_predictions_dict




# @profile
def predict_from_zernikegrams(
    np_zgrams: np.ndarray,
    np_res_ids: np.ndarray,
    models: List,
    batch_size: int,
    data_irreps: o3.Irreps,
    region: Optional[List[Tuple[str, int, str]]] = None,
    np_embeddings: Optional[np.ndarray] = None
):
    if region is not None:
        region_idxs = get_res_locs_from_tups(np_res_ids, region)
        if len(region_idxs.shape) == 0:
            region_idxs = region_idxs.reshape([1]) # if only one residue, make it still a 1D array instead of a scalar so that stuff doesn't break later
        np_zgrams = np_zgrams[region_idxs]
        np_res_ids = np_res_ids[region_idxs]
        if np_embeddings is not None:
            np_embeddings = np_embeddings[region_idxs]

    N = np_zgrams.shape[0]
    aas = np_res_ids[:, 0]
    labels = np.array([ol_to_ind_size[x.decode('utf-8')] for x in aas])

    frames = np.zeros((N, 3, 3)) # dummy frames

    if np_embeddings is not None:
        dataset = ZernikegramsAndEmbeddingsDataset(np_zgrams, np_embeddings, data_irreps, labels, list(zip(list(frames), list(map(tuple, np_res_ids)))))
    else:
        dataset = ZernikegramsDataset(np_zgrams, data_irreps, labels, list(zip(list(frames), list(map(tuple, np_res_ids)))))

    ensemble_predictions_dict = {'embeddings': [], 'logits': [], 'probabilities': [], 'best_indices': [], 'targets': None, 'res_ids': np_res_ids, 'extra_predictions': []}
    for model in models:

        # not sure if I should re-instantiate the dataloader?
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        curr_model_predictions_dict = model.predict(dataloader, device='cuda' if torch.cuda.is_available() else 'cpu')

        assert (ensemble_predictions_dict['res_ids'][:5, :] == curr_model_predictions_dict['res_ids'].T[:5, :]).all() # sanity check that order of stuff is preserved, have to transpose it for some reason

        ensemble_predictions_dict['embeddings'].append(curr_model_predictions_dict['embeddings'])
        ensemble_predictions_dict['logits'].append(curr_model_predictions_dict['logits'])
        ensemble_predictions_dict['probabilities'].append(curr_model_predictions_dict['probabilities'])
        ensemble_predictions_dict['best_indices'].append(curr_model_predictions_dict['best_indices'])
        if 'extra_predictions' in curr_model_predictions_dict:
            ensemble_predictions_dict['extra_predictions'].append(curr_model_predictions_dict['extra_predictions'])

        if ensemble_predictions_dict['targets'] is None:
            ensemble_predictions_dict['targets'] = curr_model_predictions_dict['targets']
        else:
            assert (ensemble_predictions_dict['targets'][:10] == curr_model_predictions_dict['targets'][:10]).all()

    ensemble_predictions_dict['embeddings'] = np.stack(ensemble_predictions_dict['embeddings'], axis=0)
    ensemble_predictions_dict['logits'] = np.stack(ensemble_predictions_dict['logits'], axis=0)
    ensemble_predictions_dict['probabilities'] = np.stack(ensemble_predictions_dict['probabilities'], axis=0)
    ensemble_predictions_dict['best_indices'] = np.stack(ensemble_predictions_dict['best_indices'], axis=0)
    if len(ensemble_predictions_dict['extra_predictions']) > 0:
        ensemble_predictions_dict['extra_predictions'] = np.stack(ensemble_predictions_dict['extra_predictions'], axis=0)
    else:
        ensemble_predictions_dict['extra_predictions'] = None

    return ensemble_predictions_dict



def make_string_from_tup(x):
    return (x[0] + str(x[1]) + x[2]).encode()

def get_res_locs_from_tups(
    nh_ids: List,
    loc_tups: List
) -> np.ndarray:
    """Get indices of specific residues based on their residue ids"""

    # sometimes loc_tups contains a single string, indicating that all residues of that chain are to be considered
    # to handle this, we collect all the nh_string_ids belonging to every chain

    nh_string_ids = np.array([b''.join(x) for x in nh_ids[:,2:5]])

    loc_string_ids = []
    for x in loc_tups:
        if isinstance(x, str): # chain!!
            for y in nh_ids[:,2:5][np.where(nh_ids[:, 2] == x.encode())[0]]:
                loc_string_ids.append(b''.join(y))
        else:
            loc_string_ids.append(make_string_from_tup(x))
    
    loc_string_ids = np.array(loc_string_ids)

    # loc_string_ids = np.array([make_string_from_tup(x) for x in loc_tups])

    return np.squeeze(np.argwhere(
        np.logical_or.reduce(
            nh_string_ids[None,:] == loc_string_ids[:,None])))


def get_esm_embeddings(pdb_name: str, res_ids: np.ndarray, model_version: str, sequence_pdb_alignment_json: str, embeddings_cache_file: Optional[str] = None):

    import esm

    ESM_MODELS = {
        'esm2_t6_8M_UR50D': {'num_layers': 6, 'embedding_dim': 320},
        'esm2_t12_35M_UR50D': {'num_layers': 12, 'embedding_dim': 480},
        'esm2_t30_150M_UR50D': {'num_layers': 30, 'embedding_dim': 640},
        'esm2_t33_650M_UR50D': {'num_layers': 33, 'embedding_dim': 1280},
        'esm2_t36_3B_UR50D': {'num_layers': 36, 'embedding_dim': 2560},
        'esm2_t48_15B_UR50D': {'num_layers': 48, 'embedding_dim': 5120}
    }

    num_layers = ESM_MODELS[model_version]['num_layers']
    embedding_dim = ESM_MODELS[model_version]['embedding_dim']

    ## load requested ESM-2 model
    model, alphabet = eval(f'esm.pretrained.{model_version}()')
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # make each res_id into an object that can be used as a key in a dictionary

    assert pdb_name == res_ids[0][1].decode()

    # pdbids, aas, chains, resnums = [], [], [], []
    # for res_id in res_ids:
    #     pdbids.append(res_id[1].decode())
    #     aas.append(res_id[0].decode())
    #     chains.append(res_id[2].decode())
    #     resnums.append(int(res_id[3].decode()))
    # assert len(set(pdbids)) == 1
    # aas = np.array(aas)
    # chains = np.array(chains)
    # resnums = np.array(resnums)

    # unique_chains = np.unique(chains)
    # res_ids_resnums_dict = {pdb_name: {}}
    # for chain in unique_chains:
    #     aas_chain = aas[chains == chain]
    #     resnums_chain = resnums[chains == chain]
    #     res_ids_chain = res_ids[chains == chain]
    #     sorting_indices = np.argsort(resnums_chain)
    #     aas_chain = aas_chain[sorting_indices]
    #     resnums_chain = resnums_chain[sorting_indices]
    #     res_ids_resnums_dict[pdb_name][chain] = {'seq': ''.join(list(aas_chain)), 'resnums': list(resnums_chain), 'res_ids': res_ids_chain[sorting_indices]}
    
    if sequence_pdb_alignment_json is not None:
        with open(sequence_pdb_alignment_json, 'r') as f:
            sequence_pdb_alignment_dict = json.load(f)
    else:
        raise NotImplementedError()
    
    if embeddings_cache_file is not None and os.path.exists(embeddings_cache_file):
        with gzip.open(embeddings_cache_file, 'rb') as f:
            embeddings_cache = pickle.load(f)
    else:
        embeddings_cache = {}
    
    masked_sequences = []
    resnums = []
    for res_id in res_ids:
        pdbid = res_id[1].decode()
        assert pdbid == pdb_name # sanity check
        chain = res_id[2].decode()
        res_ids_resnum = str(res_id[3].decode()) # json files store keysas strings

        sequence = sequence_pdb_alignment_dict[pdbid][chain]['seq']
        resnum = sequence_pdb_alignment_dict[pdbid][chain]['pdb_resnum_to_seq_resnum'][res_ids_resnum]
        masked_sequence = sequence[:resnum-1] + '<mask>' + sequence[resnum:]

        masked_sequences.append(masked_sequence)
        resnums.append(resnum)
    
        
    batch_size = 256

    all_embeddings = []
    for i in range(0, len(masked_sequences), batch_size):

        curr_masked_sequences = masked_sequences[i:i+batch_size]
        curr_resnums = resnums[i:i+batch_size]

        cached_embeddings = []
        non_cached_embeddings = []

        cached_indices = []
        non_cached_indices = []
        non_cached_curr_masked_sequences = []
        non_cached_curr_resnums = []
        for i, seq in enumerate(curr_masked_sequences):
            if seq in embeddings_cache:
                cached_embeddings.append(embeddings_cache[seq])
                cached_indices.append(i)
            else:
                non_cached_curr_masked_sequences.append(seq)
                non_cached_curr_resnums.append(curr_resnums[i])
                non_cached_indices.append(i)
        
        curr_embeddings = np.zeros((len(curr_masked_sequences), embedding_dim))
        if len(cached_embeddings) > 0: curr_embeddings[cached_indices] = np.stack(cached_embeddings, axis=0)

        if len(non_cached_indices) > 0:

            data = []
            for i, seq in enumerate(non_cached_curr_masked_sequences):
                data.append((f'seq_{i}', seq))

            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != batch_converter.alphabet.padding_idx).sum(1)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[num_layers-1], return_contacts=False)
            token_representations = results["representations"][num_layers-1]

            ## NOTE: the first token is always a beginning-of-sequence token, so the first residue is token 1, conveniently following the protein resnums convention
            for i, (tokens_len, resnum, seq) in enumerate(zip(batch_lens, non_cached_curr_resnums, non_cached_curr_masked_sequences)):
                embedding = token_representations[i, resnum].cpu().detach().numpy()
                non_cached_embeddings.append(embedding)
                embeddings_cache[seq] = embedding
        
            curr_embeddings[non_cached_indices] = np.stack(non_cached_embeddings, axis=0)

        all_embeddings.append(curr_embeddings)
    
    all_embeddings = np.vstack(all_embeddings) # these should be in parallel with the res_ids!!!

    if embeddings_cache_file is not None:
        with gzip.open(embeddings_cache_file, 'wb') as f:
            pickle.dump(embeddings_cache, f)

    return all_embeddings


    

if __name__ == '__main__':


    trained_models_path = '/gscratch/spe/gvisan01/protein_holography-web/trained_models/HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_stability_oracle_cdna117K_all'
    model_dir_list = [os.path.join(trained_models_path, model_rel_path) for model_rel_path in os.listdir(trained_models_path)]
    models, hparams, finetuning_hparams = load_hermes_models(model_dir_list)

    pdb_file = '/gscratch/spe/gvisan01/protein_holography-web/training_data/finetuning/stability_oracle_cdna117K_tp/pdbs/2myx.pdb'

    sequence_pdb_alignment_json = 'test.json'



    predict_from_pdbfile(pdb_file,
                          models,
                          hparams,
                          256,
                          finetuning_hparams = finetuning_hparams,
                          sequence_pdb_alignment_json = sequence_pdb_alignment_json)
    






