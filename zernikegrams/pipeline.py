import numpy as np

from zernikegrams.structural_info import get_structural_info_fn
from zernikegrams.add_noise import add_noise
from zernikegrams.neighborhoods import get_neighborhoods_fn
from zernikegrams.holograms import get_holograms_fn

from typing import *

# @profile
def get_zernikegrams_from_pdbfile(pdb_file_or_pose: str, # or Pose
                                     get_structural_info_kwargs: Dict = None,
                                     get_neighborhoods_kwargs: Dict = None,
                                     get_zernikegrams_kwargs: Dict = None,
                                     add_noise_kwargs: Dict = None):

    proteins = get_structural_info_fn(pdb_file_or_pose, **get_structural_info_kwargs)

    if add_noise_kwargs is not None:
        if add_noise_kwargs['noise'] > 0:
            rng = np.random.default_rng(add_noise_kwargs['noise_seed'])
            for n in range(len(proteins)):
                proteins[n] = add_noise(proteins[n], add_noise_kwargs['noise'], rng)
    
    neighborhoods = get_neighborhoods_fn(proteins, **get_neighborhoods_kwargs)

    zernikegrams = get_holograms_fn(neighborhoods, **get_zernikegrams_kwargs)

    return zernikegrams

