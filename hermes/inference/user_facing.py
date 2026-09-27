
import os
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Union

from hermes.inference.inference_hermes import predict_from_pdbfile, load_hermes_models, convert_predictions_results_to_standard_dataframe

HERMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')


def get_trained_models_path(model_version: str) -> str:
    '''
    Locate the directory holding the weights of `model_version`.

    By default we look in the `trained_models` directory of the repository this
    file lives in. The trained models are too large to ship inside the python
    package, so when `hermes` is installed (`pip install .`) and used from
    outside the repository, the location must be provided explicitly via the
    HERMES_TRAINED_MODELS_DIR environment variable.
    '''
    trained_models_dir = os.environ.get(
        'HERMES_TRAINED_MODELS_DIR', os.path.join(HERMES_DIR, 'trained_models')
    )
    trained_models_path = os.path.join(trained_models_dir, model_version)

    if not os.path.isdir(trained_models_path):
        raise FileNotFoundError(
            f'Could not find the weights of model "{model_version}" in '
            f'"{trained_models_path}".\n'
            f'Trained models are not shipped with the installed python package '
            f'because of their size. Either run from a clone of the hermes '
            f'repository, or point the HERMES_TRAINED_MODELS_DIR environment '
            f'variable at the `trained_models` directory of your clone.'
        )

    return trained_models_path


def run_hermes_on_pdbfile_or_pyrosetta_pose(
    model_version: str,
    pdbfile_or_pose: str, # or Pose
    chain_and_sites_list: Optional[List[Union[str, Tuple[str, List[int]]]]] = None,
    request: Union[str, List[str]] = 'logits',
    batch_size: int = 256,
    ensemble_at_logits_level: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    '''
    Args:
    ----
    model_version: str
        Which HERMES model to use, as found in `trained_models`, e.g. hermes_py_050

    pdbfile_or_pose: str or Pose
        Either a string representing a path to a pdbfile, or a PyRosetta Pose
    
    chain_and_sites_list: Optional[List[Union[str, Tuple[str, List[int]]]]] (optional, default None)
        List where each element is either:
            A string, indicating a chain, in which case all sites in the chain are computed
            Tuple (or List, doesn't matter) of (chain, sites), where sites is a list of positions along the chain;
                a position can be either an integer - indicating a position with empty insertion code - or a string with format '{resnum}-{icode}'.
        If None (default) then all sites on all chains in the pdbfile_or_pose are considered
    
    request: Union[str, List[str]] (default 'logits')
        which output to provide, can be a single str or a list of str; options are ['logits', 'logprobas', 'probas', 'embeddings']
    
    batch_size: int (default 256)
        Batch size to run the HERMES forward pass
    
    ensemble_at_logits_level: bool (default True)
        As each HERMES model is an ensemble of 10 architectures, when computing probabilities and log-probabilities,
        this flag ensembles the logits before computing the softmax, as opposed to ansembling the individual models' probabilities.
        There should not be a big difference, unless the ensembled models are trained very differently.
    
    Returns:
    ----
    output_df: pd.DataFrame
        Contains residue identifiers, as well as logits, probabilities, or log-probabilities (as indicated in the `request` argument)

    output_embeddings: np.ndarray of shape [num_residues, embedding_dim] or None
        If 'embeddings' is in `request`, contains embeddings extracted from the last layer of HERMES before the classification head
        

    Example requests:
    ----
    whole of chain A:
        chain_and_sites_list = ['A']

    whole of chains A and C:
        chain_and_sites_list = ['A', 'C']
    
    whole of chain A and some residues of chain C:
        chain_and_sites_list = ['A', ('C', [10, 11, 13, 20])]
    
    whole of chain A and some residues of chain C, with some non-empty insertion codes:
        chain_and_sites_list = ['A', ('C', [10, 11, '13-A', 20])]
        chain_and_sites_list = ['A', ('C', ['10', '11', '13-A', '20'])]
    '''

    if isinstance(request, str):
        request = [request]

    # get model
    trained_models_path = get_trained_models_path(model_version)
    model_dir_list = [os.path.join(trained_models_path, model_rel_path) for model_rel_path in os.listdir(trained_models_path)]
    models, hparams, _ = load_hermes_models(model_dir_list)

    ## construct regions input
    if chain_and_sites_list is None:
        regions = None
    else:
        regions = {}
        for elem in chain_and_sites_list:
            if isinstance(elem, str): # chain identifier
                if 'whole_chains' not in regions:
                    regions['whole_chains'] = []
                regions['whole_chains'].append(elem)
            else:
                if 'specific_sites' not in regions:
                    regions['specific_sites'] = []
                chain, sites = elem
                for site in sites:
                    split_site = str(site).split('-')
                    assert len(split_site) <= 2
                    resnum = int(split_site[0])
                    if len(split_site) == 2:
                        icode = split_site[1]
                    else:
                        icode = ' '
                    regions['specific_sites'].append((chain, resnum, icode))
    
    region_name_to_results = predict_from_pdbfile(pdbfile_or_pose, models, hparams, batch_size, regions=regions, chain=None, add_same_noise_level_as_training=False, ensemble_with_noise=False)

    output_list = [convert_predictions_results_to_standard_dataframe(region_name_to_results[region_name], request, ensemble_at_logits_level=ensemble_at_logits_level) for region_name in region_name_to_results]

    output_df = pd.concat([out[0] for out in output_list])

    if 'embeddings' in request:
        # embeddings are numpy arrays, stacked in the same region order as output_df
        output_embeddings = np.concatenate([out[1] for out in output_list], axis=0)
    else:
        output_embeddings = None
    
    return output_df, output_embeddings


