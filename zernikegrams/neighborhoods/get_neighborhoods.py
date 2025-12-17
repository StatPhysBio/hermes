#
# This file computes the atomic spherical coordinates in a given set of
# neighborhoods and outputs a file with these coordinates.
#

"""Gather neighborhoods from structural infos"""
from argparse import ArgumentParser
import sys
from time import time

import h5py
from hdf5plugin import LZ4
import numpy as np
from rich.progress import Progress

from zernikegrams.neighborhoods.neighborhoods_core import (
    get_neighborhoods_from_protein,
    pad_neighborhoods,
)

from zernikegrams.preprocessors.proteins_hdf5 import HDF5Preprocessor
from zernikegrams.utils import log_config as logging

logger = logging.getLogger(__name__)


def get_neighborhoods_fn(
    proteins: np.ndarray,
    r_max: float,
    remove_central_residue: bool = False,
    central_residue_only: bool = False,
    remove_central_sidechain: bool = False,
    keep_central_CA: bool = False,
    backbone_only: bool = False,
    coordinate_system: str = "spherical",
    align_to_backbone_frame: bool = False,
    padded_length: int = 1000,
    unique_chains: bool = False,
    get_residues=None,
):

    L = len(proteins[0]["pdb"].decode("utf-8"))
    dt = np.dtype(
        [
            ("res_id", f"S{L}", (6)),
            ("atom_names", "S4", (padded_length)),
            ("elements", "S1", (padded_length)),
            ("res_ids", f"S{L}", (padded_length, 6)),
            ("coords", "f4", (padded_length, 3)),
            ("SASAs", "f4", (padded_length)),
            ("charges", "f4", (padded_length)),
        ]
    )

    neighborhoods = []
    num_nbs = 0
    for np_protein in proteins:
        pdb, nbs = get_padded_neighborhoods(
            np_protein,
            r_max,
            padded_length,
            unique_chains,
            remove_central_residue,
            remove_central_sidechain,
            central_residue_only,
            keep_central_CA,
            coordinate_system=coordinate_system,
            align_to_backbone_frame=align_to_backbone_frame,
            backbone_only=backbone_only,
            get_residues=get_residues,
        )
        if nbs is None:
            print(f"Error with PDB {pdb}. Skipping.")
            continue

        neighborhoods.append(nbs)
        num_nbs += len(nbs)

    np_neighborhoods = np.zeros(shape=(num_nbs,), dtype=dt)
    n = 0
    for nbs in neighborhoods:
        for nb in nbs:
            np_neighborhoods[n] = (*nb,)
            n += 1

    return np_neighborhoods


def get_proteinnet__pdb_chain_pairs(testing=False):
    if testing:
        f = open("/gscratch/stf/gvisan01/hermes/training_data/pretraining/casp12_chains/validation")
        print("Using ProteinNet validation chains.")
    else:
        f = open("/gscratch/stf/gvisan01/hermes/training_data/pretraining/casp12_chains/training_30")
        print("Using ProteinNet training_30 chains.")
    lines = f.readlines()
    f.close()

    pdbs = []
    chains = []
    d_pdbs = []
    id_line = False
    for line in lines[:]:
        if id_line:
            id_line = False
            split_line = line.split("_")
            pdb = split_line[0]
            if testing:
                pdb = pdb.split("#")[1]
            pdbs.append(pdb)
            if len(split_line) == 3:
                chains.append(split_line[2].split("\n")[0])
            else:
                chains.append(split_line[1][-3].upper())
                d_pdbs.append(split_line[0])
        if "[ID]" in line:
            id_line = True

    return set(map(lambda x: "_".join(x), zip(pdbs, chains)))


def get_padded_neighborhoods(
    np_protein,
    r_max,
    padded_length,
    unique_chains,
    remove_central_residue: bool,
    remove_central_sidechain: bool,
    central_residue_only: bool,
    keep_central_CA: bool,
    coordinate_system: str = "spherical",
    align_to_backbone_frame: bool = False,
    backbone_only: bool = False,
    get_residues=None,
):
    """
    Gets padded neighborhoods associated with one structural info unit

    Parameters:
    np_protein : np.ndarray
        Array representation of a protein
    r_max : float
        Radius of the neighborhood
    padded_length : int
        Total length including padding
    unique_chains : bool
        Flag indicating whether chains with identical sequences should
        contribute unique neoighborhoods
    """

    pdb = np_protein[0]
    sys.stdout.flush()

    logger.debug(f"Coordinate system is {coordinate_system}")

    try:
        if get_residues is None:
            res_ids = None
        else:
            res_ids = get_residues(np_protein)

        neighborhoods = get_neighborhoods_from_protein(
            np_protein,
            r_max=r_max,
            res_ids_selection=res_ids,
            uc=unique_chains,
            remove_central_residue=remove_central_residue,
            remove_central_sidechain=remove_central_sidechain,
            central_residue_only=central_residue_only,
            keep_central_CA=keep_central_CA,
            backbone_only=backbone_only,
            align_to_backbone_frame=align_to_backbone_frame,
            coordinate_system=coordinate_system,
        )
        padded_neighborhoods = pad_neighborhoods(
            neighborhoods, padded_length=padded_length
        )
    except Exception as e:
        print(e, flush=True)
        logging.error(e)
        logging.error(f"Error with{pdb}")
        # print(traceback.format_exc())
        return (
            pdb,
            None,
        )

    return (
        pdb,
        padded_neighborhoods,
    )


def get_neighborhoods_from_dataset(
    hdf5_in,
    input_dataset_name,
    r_max,
    hdf5_out,
    output_dataset_name,
    unique_chains,
    coordinate_system: str,
    align_to_backbone_frame: bool,
    remove_central_residue: bool,
    remove_central_sidechain: bool,
    central_residue_only: bool,
    keep_central_CA,
    backbone_only: bool = False,
    parallelism: int = 40,
    max_atoms=1000,
    get_residues_file=None,
    filter_out_chains_not_in_proteinnet=False,
    pdb_chain_pairs_to_consider_filepath=None
):
    """
    Parallel retrieval of neighborhoods from structural info file and writing
    to neighborhods hdf5_out file

    Parameters
    ----------
    hdf5_in : str
        Path to hdf5 file containing structural info
    protein_list : str
        Name of the dataset within the hdf5 file to process
    r_max : float
        Radius of the neighborhood
    hdf5_out : str
        Path to write the output file
    unique_chains : bool
        Flag indicating whether or not chains with identical sequences should each
        contribute neighborhoods
    parallelism : int
        Number of workers to use
    """
    # metadata = get_metadata()

    if filter_out_chains_not_in_proteinnet and pdb_chain_pairs_to_consider_filepath is not None:
        raise ValueError("Cannot use both filter_out_chains_not_in_proteinnet and pdb_chain_pairs_to_consider_filepath, as they conflict with each other.")

    ds = HDF5Preprocessor(hdf5_in, input_dataset_name)

    L = np.max([ds.pdb_name_length, 5])
    n = 0
    curr_size = 10000

    dt = np.dtype(
        [
            ("res_id", f"S{L}", (6)),
            ("atom_names", "S4", (max_atoms)),
            ("elements", "S2", (max_atoms)),
            ("res_ids", f"S{L}", (max_atoms, 6)),
            ("coords", "f4", (max_atoms, 3)),
            ("SASAs", "f4", (max_atoms)),
            ("charges", "f4", (max_atoms)),
        ]
    )

    logger.info("Writing hdf5 file")
    with h5py.File(hdf5_out, "w") as f:
        f.create_dataset(
            output_dataset_name,
            shape=(curr_size,),
            maxshape=(None,),
            dtype=dt,
            compression=LZ4(),
        )
        # record_metadata(metadata, f[protein_list])

    if filter_out_chains_not_in_proteinnet:
        print("Filtering out chains not in ProteinNet.")
        try:
            pdb_chain_pairs_to_consider = get_proteinnet__pdb_chain_pairs(
                testing=True if "testing" in hdf5_in else False
            )  # not an ideal if-confition... but it saves extra parameters
        except FileNotFoundError:
            print("Could not find ProteinNet file. Ignoring.")
    elif pdb_chain_pairs_to_consider_filepath is not None:
        print(f"Filtering out chains not in {pdb_chain_pairs_to_consider_filepath}.")
        with open(pdb_chain_pairs_to_consider_filepath, "r") as f:
            pdb_chain_pairs_to_consider = set(f.read().splitlines())

    # import user method
    if not get_residues_file is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "get_residues_module", get_residues_file
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["get_residues_module"] = module
        spec.loader.exec_module(module)

        from get_residues_module import get_residues
    else:
        get_residues = None

    logger.debug(f"Gathering unique chains {unique_chains}")
    nhs = np.empty(shape=(curr_size,), dtype=(f"S{L}", (6)))

    pdbs_pass = []
    pdbs_fail = []

    with Progress() as bar:
        task = bar.add_task("Neighborhoods", total=ds.count())
        with h5py.File(hdf5_out, "r+") as f:
            for i, (pdb, neighborhoods) in enumerate(
                ds.execute(
                    get_padded_neighborhoods,
                    limit=None,
                    params={
                        "r_max": r_max,
                        "padded_length": max_atoms,
                        "unique_chains": unique_chains,
                        "coordinate_system": coordinate_system,
                        "align_to_backbone_frame": align_to_backbone_frame,
                        "remove_central_residue": remove_central_residue,
                        "remove_central_sidechain": remove_central_sidechain,
                        "central_residue_only": central_residue_only,
                        "keep_central_CA": keep_central_CA,
                        "backbone_only": backbone_only,
                        "get_residues": get_residues
                    },
                    parallelism=parallelism,
                )
            ):
                try:

                    if neighborhoods is None:
                        pdbs_fail.append(pdb)
                        continue

                    if filter_out_chains_not_in_proteinnet or pdb_chain_pairs_to_consider_filepath is not None:
                        filtered_neighborhoods = []
                        for neighborhood in neighborhoods:
                            if (
                                "_".join(
                                    [
                                        neighborhood["res_id"][1].decode("utf-8"),
                                        neighborhood["res_id"][2].decode("utf-8"),
                                    ]
                                )
                                in pdb_chain_pairs_to_consider
                            ):
                                filtered_neighborhoods.append(neighborhood)
                        neighborhoods = np.array(filtered_neighborhoods)

                    neighborhoods_per_protein = neighborhoods.shape[0]

                    if neighborhoods_per_protein == 0:
                        logger.warning(f"No neighborhoods for {pdb}, possibly because no pdb_chain pair with this pdb is present in the file. Skipping.")
                        pdbs_fail.append(pdb)
                        continue

                    while n + neighborhoods_per_protein > curr_size:
                        curr_size += 10000
                        nhs.resize((curr_size, 6))
                        f[output_dataset_name].resize((curr_size,))


                    f[output_dataset_name][n : n + neighborhoods_per_protein] = neighborhoods
                    nhs[n : n + neighborhoods_per_protein] = neighborhoods["res_id"]

                    n += neighborhoods_per_protein
                except Exception as e:
                    logger.warning(
                        "Failed to process neighborhood with the following error:"
                    )
                    logger.exception(e)
                finally:
                    # attempt to address memory issues. currently unsuccessfully
                    pdbs_pass.append(pdb)
                    bar.update(
                        task,
                        advance=1,
                        description=f"Neighborhoods: {i + 1}/{ds.count()}",
                    )

            logger.info(f"Number of processed neighborhoods: {n}")
            f[output_dataset_name].resize((n,))
            nhs.resize((n, 6))

    with h5py.File(hdf5_out, "r+") as f:
        f.create_dataset("nh_list", data=nhs)
        # record_metadata(metadata, f["nh_list"])

        f.create_dataset("pdbs_pass", data=pdbs_pass)
        f.create_dataset("pdbs_fail", data=pdbs_fail)

    logger.info("Done with parallel computing")


def main():
    parser = ArgumentParser()

    parser.add_argument("--hdf5_in", type=str, help="hdf5 filename", required=True)
    parser.add_argument(
        "--hdf5_out", type=str, help="ouptut hdf5 filename", required=True
    )
    parser.add_argument(
        "--input_dataset_name",
        type=str,
        help='Name of the dataset within hdf5_in where the structural information is stored. We recommend keeping this set to simply "data".',
        default="data",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        help='Name of the dataset within hdf5_out where the neighborhoods will be stored. We recommend keeping this set to simply "data".',
        default="data",
    )
    parser.add_argument(
        "--r_max",
        type=float,  # TODO: change this to rcut
        help="Radius of neighborhood, with zero at central residue's CA",
        default=10.0,
    )
    parser.add_argument(
        "--coordinate_system",
        type=str,
        help="Coordinate system in which to store the neighborhoods.",
        default="spherical",
        choices=["spherical", "cartesian"],
    )
    parser.add_argument(
        "--align_to_backbone_frame",
        help="Whether to align the neighborhood to the central residue's backbone frame, thereby standardizing its orientation and providing a rudimental form of rotational invariance.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--remove_central_residue",
        help="Whether to remove the central residue from the neighborhood. Cannot be done in conjunction with --central_residue_only nor --remove_central_sidechain.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--remove_central_sidechain",
        help="Whether to remove the central residue's sidechain from the neighborhood, while keeping the backbone atoms. Cannot be done in conjunction with --central_residue_only nor --remove_central_residue.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--central_residue_only",
        help="Whether to only keep the central residue in the neighborhood. Cannot be done in conjunction with --remove_central_residue nor --remove_central_sidechain.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--keep_central_CA",
        help="Whether to keep the central residue's CA in the neighborhood.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--backbone_only",
        help="Whether to only include backbone atoms in the neighborhood, as opposed to all atoms.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--unique_chains",
        help="Only take one neighborhood per residue per unique chain",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--parallelism", type=int, help="Parallelism for multiprocessing.", default=4
    )
    parser.add_argument(
        "--get_residues_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--filter_out_chains_not_in_proteinnet",
        help="Whether to filter out chains not in proteinnet. Only relevant when training and testing on proteinnet casp12 PDBs. Has same effect as --pdb_chain_pairs_to_consider_filepath if the file were prepared in advanced, here for legacy purposes.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pdb_chain_pairs_to_consider_filepath",
        type=str,
        help="Path to file containing pdb_chain pairs to consider. Only relevant when filter_out_chains_not_in_proteinnet is True.",
        default=None
    )

    args = parser.parse_args()
    s = time()

    get_neighborhoods_from_dataset(
        args.hdf5_in,
        args.input_dataset_name,
        args.r_max,
        args.hdf5_out,
        args.output_dataset_name,
        args.unique_chains,
        args.coordinate_system,
        args.align_to_backbone_frame,
        args.remove_central_residue,
        args.remove_central_sidechain,
        args.central_residue_only,
        args.keep_central_CA,
        args.backbone_only,
        args.parallelism,
        get_residues_file=args.get_residues_file,
        filter_out_chains_not_in_proteinnet=args.filter_out_chains_not_in_proteinnet,
        pdb_chain_pairs_to_consider_filepath=args.pdb_chain_pairs_to_consider_filepath
    )

    logger.info(f"Total time = {time() - s:.2f} seconds")


if __name__ == "__main__":
    main()