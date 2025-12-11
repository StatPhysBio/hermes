
import os, sys
import numpy as np
import h5py
import hdf5plugin

import torch

from e3nn import o3
from torch.utils.data import Dataset

from typing import *



def get_norm_factor(projections: np.ndarray, data_irreps: o3.Irreps):
    ls_indices = np.concatenate([[l]*(2*l+1) for l in data_irreps.ls])
    batch_size = 2000
    norm_factors = []
    num_batches = projections.shape[0] // batch_size
    for i in range(num_batches):
        signals = projections[i*batch_size : (i+1)*batch_size]
        batch_norm_factors = np.sqrt(np.einsum('bf,bf,f->b', signals, signals, 1.0 / (2*ls_indices + 1)))
        norm_factors.append(batch_norm_factors)
    
    # final batch for the remaining signals
    if (projections.shape[0] % batch_size) > 0:
        signals = projections[(i+1)*batch_size:]
        batch_norm_factors = np.sqrt(np.einsum('bf,bf,f->b', signals, signals, 1.0 / (2*ls_indices + 1)))
        norm_factors.append(batch_norm_factors)

    norm_factor = np.mean(np.concatenate(norm_factors, axis=-1))

    return norm_factor


def get_data_irreps(hparams):

    # get list of channels. currently, we only really need the length of this list to compute the data irreps
    if hparams['channels'] == 'dlpacker':
        channels = ['C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif hparams['channels'] == 'dlpacker_plus':
        channels = ['CAlpha', 'C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif hparams['channels'] == 'AAs':
        channels = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                    b'S', b'T', b'W', b'Y', b'V', b'G', "all other AAs"]
    else:
        channels = hparams['channels'].split(',')
    
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
    # print('Data Irreps:', data_irreps)
    ls_indices = np.concatenate([[l]*(2*l+1) for l in data_irreps.ls])

    return data_irreps, ls_indices


def stringify(res_id):
    return '_'.join(list(map(lambda x: x.decode('utf-8'), list(res_id))))

def stringify_array(res_ids):
    return np.array([stringify(res_id) for res_id in res_ids])


class ZernikegramsDataset(Dataset):
    def __init__(self, x: np.ndarray, irreps: o3.Irreps, y: np.ndarray, c: List):
        self.x = x # [N, dim]
        self.y = y # [N,]
        self.c = c # [N, ANY]
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == len(c)

        self.ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps.ls])
        self.unique_ls = sorted(list(set(irreps.ls)))
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx: int):
        x_fiber = {}
        for l in self.unique_ls:
            x_fiber[l] = torch.tensor(self.x[idx][self.ls_indices == l]).view(-1, 2*l+1).float()
        
        return x_fiber, torch.tensor(self.x[idx]).float(), torch.tensor(self.y[idx]).to(torch.long), self.c[idx]


def load_single_split_data(hparams, split, get_norm_factor_if_training=True, test_data_filepath=None):

    assert split in {'validation_zgram', 'testing_zgram', 'validation', 'testing'} or 'training' in split # accomodate for divided training data, usually of the kind 'training__N'

    data_irreps, ls_indices = get_data_irreps(hparams)

    if test_data_filepath is not None:
        raise NotImplementedError('The use of "test_data_filepath" is not implemented yet for the new standardized data loading procedure.')

    pdb_list_filename = hparams['pdb_list_filename_template'].format(split=split)

    with h5py.File(hparams['data_filepath'].format(pdb_list_filename=pdb_list_filename, **hparams), 'r') as f:
        data = f['data'][:]

        zgrams = data['zernikegram']
        labels = data['label']
        res_ids = np.array(list(map(stringify, data['res_id'])))
        try:
            frames = f['data']['frame']
        except Exception as e:
            print('Warning: no frames.', file=sys.stderr)
            print(e)
            frames = np.zeros((labels.shape[0], 3, 3))
    
    if 'training' in split and get_norm_factor_if_training and hparams['normalize_input']:
        ##### debugging #####
        power = np.mean(np.sqrt(np.einsum('bf,bf,f->b', zgrams[:1000], zgrams[:1000], 1.0 / (2*ls_indices + 1))))
        print('Power before norm: %.4f' % power)
        sys.stdout.flush()
        ##### debugging #####

        size = zgrams.shape[0]
        num_to_use = 100_000
        idxs = np.random.default_rng(1234567890).integers(size, size=num_to_use)
        print(f'Getting norm factor using a random sample with fixed seed of {num_to_use} zgrams.', flush=True)
        norm_factor = get_norm_factor(zgrams[idxs], data_irreps)
        # zgrams = zgrams / norm_factor # ---> we don't divide it now anymore!
        print('Done, norm_factor:', norm_factor)
        
        ##### debugging #####
        power = np.mean(np.sqrt(np.einsum('bf,bf,f->b', zgrams[:1000] / norm_factor, zgrams[:1000] / norm_factor, 1.0 / (2*ls_indices + 1))))
        print('Power after norm: %.4f' % power)
        sys.stdout.flush()
        ##### debugging #####
    else:
        norm_factor = None

    print('Running on %s set with %d examples.' % (split, zgrams.shape[0]))

    dataset = ZernikegramsDataset(zgrams, data_irreps, labels, list(zip(list(frames), list(res_ids))))
                                    
    return dataset, data_irreps, norm_factor



def load_data(hparams, splits=['train', 'valid'], get_norm_factor_if_training=True, test_data_filepath=None):

    for split in splits:
        assert split in {'train', 'valid', 'test'}
        
    norm_factor = None
    datasets = {}
    if 'train' in splits:
        train_dataset, data_irreps, norm_factor = load_single_split_data(hparams, 'training', get_norm_factor_if_training=get_norm_factor_if_training)
        datasets['train'] = train_dataset
    
    if 'valid' in splits:
        valid_dataset, data_irreps, _ = load_single_split_data(hparams, 'validation')
        datasets['valid'] = valid_dataset
    
    if 'test' in splits:
        test_dataset, data_irreps, _ = load_single_split_data(hparams, 'testing') # 
        datasets['test'] = test_dataset

    return datasets, data_irreps, norm_factor






# from protein_holography_pytorch.preprocessing_faster import get_neighborhoods, get_structural_info, get_zernikegrams

# def get_zernikegrams_from_pdb(hparams: Dict,
#                                data_filepath: str,
#                                parser: str = 'pyrosetta',
#                                verbose: bool = False):
#     '''
#     Just an intuitive wrapper around get_zernikegrams_from_pdbs() for the case when a single pdb is provided.
#     '''
#     assert data_filepath[-4:] == '.pdb'
#     return get_zernikegrams_from_pdbs(hparams, data_filepath, pdb_dir=None, parser=parser, verbose=verbose)

# def get_zernikegrams_from_pdbs(hparams: Dict,
#                                data_filepath: str,
#                                pdb_dir: Optional[str] = None,
#                                parser: str = 'pyrosetta',
#                                verbose: bool = False):

#     if data_filepath[-4:] == '.pdb': # --> single pdb is provided
#         protein = get_structural_info(data_filepath, parser=parser)
#         nbs = get_neighborhoods(protein, hparams['rcut'], remove_central_residue = hparams['remove_central_residue'], backbone_only = False)

#     else: # (data_filepath[-4:] == '.txt' and pdb_dir is not None) --> list of pdbs is provided
#         with open(data_filepath, 'r') as f:
#             pdb_list = [pdb.strip() for pdb in f.readlines()]
        
#         if verbose: print('Collecting neighborhoods from %d PDB files...' % len(pdb_list))
#         sys.stdout.flush()
        
#         proteins = get_structural_info([os.path.join(pdb_dir, pdb+'.pdb') for pdb in pdb_list], parser=parser)
#         nbs = get_neighborhoods(proteins, hparams['rcut'], remove_central_residue = hparams['remove_central_residue'], backbone_only = False)

#     if verbose: print('Generating zernikegrams...')
#     sys.stdout.flush()
#     zgrams_data = get_zernikegrams(nbs, hparams['rcut'], hparams['radial_func_max'], hparams['lmax'], hparams['channels'].split(','), radial_func_mode=hparams['radial_func_mode'], backbone_only=False, request_frame=False, get_physicochemical_info_for_hydrogens=hparams['get_physicochemical_info_for_hydrogens'], rst_normalization=hparams['rst_normalization'])

#     if verbose: print(zgrams_data['zernikegram'].shape)

#     return prepare_zgrams(hparams, zgrams_data, verbose=verbose)


# def get_zernikegrams_from_hdf5(hparams: Dict,
#                                data_filepath: str,
#                                input_dataset_name: str = 'data',
#                                verbose:bool = False):
    
#     import h5py
#     import hdf5plugin # for reading compressed files
    
#     with h5py.File(data_filepath, 'r') as f:
#         mask = ~np.logical_and.reduce(f[input_dataset_name]['res_id'] == np.array([b'', b'', b'', b'', b'', b'']), axis=1) # select non-empty zernikegrams only
#         zgrams_data = f[input_dataset_name][mask]

#     return prepare_zgrams(hparams, zgrams_data, verbose=verbose)



# def prepare_zgrams(hparams, zgrams_data, verbose=False):

#     data_irreps, ls_indices = get_data_irreps(hparams)
    
#     if hparams['normalize_input']:
#         normalize_input_at_runtime = True
#     else:
#         normalize_input_at_runtime = False
    
#     def stringify(data_id):
#         return '_'.join(list(map(lambda x: x.decode('utf-8'), list(data_id))))

#     if zgrams_data['frame'] is None:
#         zgrams_data['frame'] = np.zeros((zgrams_data['label'].shape[0], 3, 3))

#     if verbose: print('Power: %.4f' % (np.mean(np.sqrt(np.einsum('bf,bf,f->b', zgrams_data['zernikegram'][:1000], zgrams_data['zernikegram'][:1000], 1.0 / (2*ls_indices + 1))))))

#     dataset = ZernikegramsDataset(zgrams_data['zernikegram'], data_irreps, zgrams_data['label'], list(zip(list(zgrams_data['frame']), list(map(stringify, zgrams_data['res_id'])))))

#     return dataset, data_irreps, normalize_input_at_runtime




