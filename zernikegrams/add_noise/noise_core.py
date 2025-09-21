import numpy as np

from typing import *


def add_noise(np_protein: np.ndarray, noise_level: float, rng: Optional[np.random.Generator] = None):
    '''
    Adds noise to protein coordinates, sampled from a gaussian distribution with mean 0 and std noise_level.

    np_protein: np.ndarray
        Protein data, as a numpy array, as outputted by get_structural_info() routine.
    noise_level: float
        Standard deviation - in angstroms - of the gaussian distribution from which the noise is sampled.
    
    '''

    real_idxs = np_protein['res_ids'][:, 0] != b''

    coords = np_protein['coords']
    if rng is not None:
        noise = rng.normal(0, noise_level, (np.sum(real_idxs), 3))
    else:
        noise = np.random.normal(0, noise_level, (np.sum(real_idxs), 3))

    coords[real_idxs] += noise

    np_protein['coords'] = coords

    return np_protein