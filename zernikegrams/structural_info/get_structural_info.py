import argparse
import h5py
import hdf5plugin
import os
import sys
import sqlitedict
import time

from typing import *
from rich.progress import Progress

import numpy as np

from zernikegrams.utils import log_config as logging
from zernikegrams.utils.pdb_lists import (
    pdb_list_from_dir,
    pdb_list_from_foldcomp,
)
from zernikegrams.preprocessors.pdbs import (
    PDBPreprocessor,
    FoldCompPreprocessor,
)
from zernikegrams.structural_info.structural_info_core import (
    get_structural_info_from_protein,
    pad_structural_info,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hdf5_out",
        "-o",
        type=str,
        help="Output hdf5 filename, where structural info will be stored.",
        required=True,
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        help='Name of the dataset within output_hdf5 where the structural information will be saved. We recommend keeping this set to simply "data".',
        default="data",
    )
    parser.add_argument(
        "--parser",
        type=str,
        default="biopython",
        choices=["biopython", "pyrosetta"],
        help="Parser to use for reading PDBs. Options are 'biopython' or 'pyrosetta'. The 'biopython' option uses biopython, pdbfixer, and reduce to process the structures.\
              The 'pyrosetta' option uses pyrosetta to process the structures. The 'pyrosetta' option requires a pyrosetta license. \
              The 'pyrosetta' option automatically adds hydrogens to the structures, it keeps all extra molecules except for water, and it does not substitute non-canonical residues.",
    )
    parser.add_argument(
        "--pdb_list_file",
        type=str,
        help="Path to file containing list of PDB files of interest, one per row. Required if --foldcomp is not set",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        help="directory of pdb files. Required if --foldcomp is not set",
    )
    parser.add_argument(
        "--foldcomp",
        type=str,
        help="Path to foldcomp file containing compressed PDBs. Required if --pdb_dir and --pdb_list_file not set",
    )
    parser.add_argument(
        "--parallelism",
        "-p",
        type=int,
        help="Maximum number of CPU cores to use.",
        default=4,
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        help="max number of atoms per protein, for padding purposes",
        default=200000,
    )
    parser.add_argument(
        "--angle_db", type=str, help="path to chi angle database", default=None
    )
    parser.add_argument(
        "--vec_db", type=str, help="path to normal vector database", default=None
    )
    parser.add_argument(
        "--SASA",
        "-S",
        action="store_true",
        default=False,
        help="If present, SASAs are calculated for each atom.",
    )
    parser.add_argument(
        "--charge",
        "-c",
        action="store_true",
        default=False,
        help="If present, charges are calculated for each atom using AMBER99 force fields.",
    )
    parser.add_argument(
        "--DSSP",
        action="store_true",
        default=False,
        help="If present, secondary structure annotations are calculated for each residue",
    )
    parser.add_argument(
        "--fix_pdbs",
        "-F",
        action="store_true",
        default=False,
        help="Use OpenMM to find and fix missing atoms in pdb",
    )
    parser.add_argument(
        "--add_hydrogens",
        "-H",
        action="store_true",
        default=False,
        help="Use Reduce to add hydrogen atoms to the incoming PDBs. This implies --fix_pdbs",
    )
    parser.add_argument(
        "--remove_extra_molecules",
        action="store_true",
        default=False,
        help="Filter out extra_molecules from PDB"
    )
    parser.add_argument(
        "--handle_multi_structures",
        "-m",
        default="warn",
        choices=["crash", "warn", "allow"],
        help="Behavior for handling PDBs with multiple structures",
    )
    parser.add_argument("--logging", type=str, help="logging level", default="INFO")
    parser.add_argument(
        "--fixed_pdb_dir",
        type=str,
        help="[Optional] Directory to save fixed pdb files, if -F or -H is selected"
    )
    return parser

def parse_args():
    args = get_parser().parse_args()
    if args.pdb_dir is None and args.foldcomp is None:
        msg = "pdb_dir or foldcomp must be set"
        logger.exception(msg)
        raise ValueError(msg)

    args.input_path = args.foldcomp if args.foldcomp is not None else args.pdb_dir

    if args.parser == "pyrosetta":
        print("Warning: Using pyrosetta for parsing. The 'pyrosetta' option automatically adds hydrogens to the structures, it keeps all extra molecules except for water, and it does not substitute non-canonical residues.")

    return args


def get_structural_info_fn(pdb_file: str,
                            parser: str = 'biopython',
                            padded_length: Optional[int] = None,
                            SASA: bool = True,
                            charge: bool = True,
                            DSSP: bool = True,
                            angles: bool = True,
                            fix: bool = False,
                            hydrogens: bool = False,
                            extra_molecules: bool = True,
                            multi_struct: str = "warn"):

    """
    Get structural info from a single pdb file.
    If padded_length is None, does not pad the protein.
    """

    if isinstance(pdb_file, str):
        L = len(pdb_file.split('/')[-1].split('.')[0])
    else:
        L = len(pdb_file[0].split('/')[-1].split('.')[0])
        for i in range(1, len(pdb_file)):
            L = max(L, len(pdb_file[i].split('/')[-1].split('.')[0]))

    if isinstance(pdb_file, str):
        pdb_file = [pdb_file]
    

    n = 0
    for i, pdb_file in enumerate(pdb_file):

        if padded_length is None:
            si = get_structural_info_from_protein(
                    pdb_file,
                    parser=parser,
                    calculate_SASA=SASA,
                    calculate_charge=charge,
                    calculate_DSSP=DSSP,
                    calculate_angles=angles,
                    fix=fix,
                    hydrogens=hydrogens,
                    extra_molecules=extra_molecules,
                    multi_struct=multi_struct)
        else:
            si = get_padded_structural_info(
                    pdb_file,
                    parser=parser,
                    padded_length=padded_length,
                    SASA=SASA,
                    charge=charge,
                    DSSP=DSSP,
                    angles=angles,
                    fix=fix,
                    hydrogens=hydrogens,
                    extra_molecules=extra_molecules,
                    multi_struct=multi_struct)

        if si[0] is None:
            print(f"Failed to process {pdb_file}", file=sys.stderr)
            continue

        try:
            pdb,atom_names,elements,res_ids,coords,sasas,charges,res_ids_per_residue,angles,vecs,multi_struc = si
        except ValueError:
            pdb,(atom_names,elements,res_ids,coords,sasas,charges,res_ids_per_residue,angles,vecs,multi_struc) = si

        if n == 0:
            if padded_length is None:
                length = len(atom_names)
                dt = np.dtype([
                    ('pdb',f'S{L}',()),
                    ('atom_names', 'S4', (length)),
                    ('elements', 'S2', (length)),
                    ('res_ids', f'S{L}', (length, 6)),
                    ('coords', 'f4', (length, 3)),
                    ('SASAs', 'f4', (length)),
                    ('charges', 'f4', (length)),
                ])
            else:
                dt = np.dtype([
                    ('pdb',f'S{L}',()),
                    ('atom_names', 'S4', (padded_length)),
                    ('elements', 'S2', (padded_length)),
                    ('res_ids', f'S{L}', (padded_length, 6)),
                    ('coords', 'f4', (padded_length, 3)),
                    ('SASAs', 'f4', (padded_length)),
                    ('charges', 'f4', (padded_length)),
                ])
            
            np_protein = np.zeros(shape=(len(pdb_file),), dtype=dt)

        np_protein[n] = (pdb,atom_names,elements,res_ids,coords,sasas,charges,)
        
        n += 1

    np_protein.resize((n,))

    return np_protein


def get_structural_info_from_dataset(
    input_path: str,
    pdb_list: List[str],
    max_atoms: int,
    hdf5_out: str,
    output_dataset_name: str,
    parallelism: int,
    parser: str = 'biopython',
    angle_db: str = None,
    vec_db: str = None,
    SASA: bool = True,
    charge: bool = True,
    DSSP: bool = True,
    fix: bool = False,
    hydrogens: bool = False,
    extra_molecules: bool = True,
    handle_multi_structures: str = "warn",
    fixed_pdb_dir: str = None,
) -> None:
    """
    Parallel processing of PDBs into structural info

    Parameters
    ---------
    inputs:
        either 1)
            pdb_dir : str
                Path where the pdb files are stored
        or 2)
            foldcomp: str
                path to foldcomp file with compressed PDBs
    pdb_list: List of PDBs to processes
    max_atoms : int
        Max number of atoms in a protein for padding purposes
    hdf5_out : str
        Path to hdf5 file to write
    parallelism : int
        Number of workers to use
    parser : str
        Parser to use for reading PDBs. Options are 'biopython' or 'pyrosetta'. The 'biopython' option uses biopython, pdbfixer, and reduce to process the structures.
    angle_db : str | None
        If set, path to sqlite db to store chi angles.
        Keys will be residue IDs and values will be up to four chi angles
    vec_db : str | None
        If set, path to sqlite db to store normal vectors.
        Keys will be residue IDs and values will be up to four normal vectors
    SASA: bool
        Whether or not to calculate SASAs
    charge: bool
        Whether or not to calculate charges
    DSSP: bool
        Whether or not to calculate DSSP
    Fix: bool
        Whether or not to fix missing atoms
    Hydrogens: bool
        Whether or not to add hydrogen atoms
    extra_molecules: bool
        Whether or not to keep extra_molecules
    handle_multi_structures
        Behavior for handling PDBs with multiple structures
    Fixed_pdb_dir
        Directory to save fixed pdbs
    """
    if os.path.isdir(input_path):
        pdb_dir = input_path

        pdb_list_from_dir = []
        for file in os.listdir(pdb_dir):
            if file.endswith(".pdb"):
                pdb = file.removesuffix(".pdb")
                pdb_list_from_dir.append(pdb)

        # filter out pdbs that are not in the directory
        pdb_list = list(set(pdb_list) & set(pdb_list_from_dir))

        processor = PDBPreprocessor(pdb_list, pdb_dir)

    else:
        processor = FoldCompPreprocessor(pdb_list, input_path)

    L = np.max([processor.pdb_name_length, 5])
    logger.info(f"Maximum pdb name L = {L}")

    # dt_arr = [
    #     ("pdb", f"S{L}", ()),
    #     ("atom_names", "S4", (max_atoms)),
    #     ("elements", "S2", (max_atoms)),
    #     ("res_ids", f"S{L}", (max_atoms, 6)),
    #     ("coords", "f4", (max_atoms, 3)),
    # ]
    # if SASA:
    #     dt_arr.append(("SASAs", "f4", (max_atoms)))
    # if charge:
    #     dt_arr.append(("charges", "f4", (max_atoms)))
    # dt = np.dtype(dt_arr)
    dt = np.dtype(
        [
            ("pdb", f"S{L}", ()),
            ("atom_names", "S4", (max_atoms)),
            ("elements", "S2", (max_atoms)),
            ("res_ids", f"S{L}", (max_atoms, 6)),
            ("coords", "f4", (max_atoms, 3)),
            ("SASAs", "f4", (max_atoms)),
            ("charges", "f4", (max_atoms)),
        ]
    )

    with h5py.File(hdf5_out, "w") as f:
        f.create_dataset(
            output_dataset_name,
            shape=(processor.size,),
            maxshape=(None,),
            dtype=dt,
            chunks=True,
            compression=hdf5plugin.LZ4(),
        )

    angle_dict = {}
    vec_dict = {}

    with Progress() as bar:
        task = bar.add_task("Structural Info", total=processor.count())
        with h5py.File(hdf5_out, "r+") as f:
            n = 0
            n_multimodel = 0
            for structural_info in processor.execute(
                callback=get_padded_structural_info,
                limit=None,
                params={
                    "parser": parser,
                    "padded_length": max_atoms,
                    "SASA": SASA,
                    "charge": charge,
                    "angles": vec_db is not None or angle_db is not None,
                    "DSSP": DSSP,
                    "fix": fix,
                    "hydrogens": hydrogens,
                    "extra_molecules": extra_molecules,
                    "multi_struct": handle_multi_structures,
                    "fixed_pdb_dir": fixed_pdb_dir,
                },
                parallelism=parallelism,
            ):
                try:
                    if structural_info[0] is None:
                        raise ValueError("Structural_info[0] is None")

                    (
                        pdb,
                        atom_names,
                        elements,
                        res_ids,
                        coords,
                        sasas,
                        charges,
                        res_ids_per_residue,
                        angles,
                        norm_vecs,
                        multi_model,
                    ) = (*structural_info,)

                    n_multimodel += multi_model[0]

                    if angle_db is not None or vec_db is not None:
                        for res_id, curr_angles, curr_norm_vecs in zip(
                            res_ids_per_residue, angles, norm_vecs
                        ):
                            res_id = "_".join([id_.decode("utf-8") for id_ in res_id])
                            angle_dict[res_id] = curr_angles.tolist()
                            vec_dict[res_id] = curr_norm_vecs.tolist()

                    f[output_dataset_name][n] = (
                        pdb,
                        atom_names,
                        elements,
                        res_ids,
                        coords,
                        sasas,
                        charges,
                    )  # [0:len(dt_arr)]

                    n += 1
                except Exception as e:
                    logger.warning("Failed to write PDB with the following error:")
                    logger.exception(e)
                finally:
                    bar.update(
                        task,
                        advance=1,
                        description=f"Structural Info: {n}/{processor.count()}",
                    )

            if handle_multi_structures in ("warn", "allow"):
                logger.info(f"PDBs with multiple models: {n_multimodel}")

            logger.info(f"PDBs successfully processed: {n}")
            f[output_dataset_name].resize((n,))

    if angle_db is not None:
        angle_db = sqlitedict.SqliteDict(angle_db, autocommit=False)
        for k, v in angle_dict.items():
            angle_db[k] = v
        angle_db.commit()
        angle_db.close()
        logger.info(f"Saved chi angles to {angle_db}")

    if vec_db is not None:
        vec_db = sqlitedict.SqliteDict(vec_db, autocommit=False)
        for k, v in vec_dict.items():
            vec_db[k] = v
        vec_db.commit()
        vec_db.close()
        logger.info(f"Saved normal vectors to {vec_db}")


def get_padded_structural_info(
    pdb_file: str,
    parser: str = "biopython",
    padded_length: int = 200000,
    SASA: bool = True,
    charge: bool = True,
    DSSP: bool = True,
    angles: bool = True,
    fix: bool = False,
    hydrogens: bool = False,
    extra_molecules: bool = True,
    multi_struct: str = "warn",
    fixed_pdb_dir: str = None,
) -> Tuple[
    bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Extract structural info used for holographic projection from Biopython pose.

    Parameters
    ----------
    pdb_file: path to file with pdb
    parser: parser to use for reading PDBs
    padded_length: size to pad to
    SASA: Whether or not to calculate SASA
    charge: Whether or not to calculate charge
    DSSP: Whether or not to calculate DSSP
    angles: Whether or not to calculate anglges
    Fix: Whether or not to fix missing atoms
    Hydrogens: Whether or not to add hydrogen atoms
    extra_molecules: Whether or not to keep extra_molecules
    multi_struct: Behavior for handling PDBs with multiple structures
    fixed_pdb_dir: Directory to save fixed pdbs

    Returns
    -------
    tuple of (bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray)
        The entries in the tuple are
            bytes encoding the pdb name string
            bytes array encoding the atom names of shape [max_atoms]
            bytes array encoding the elements of shape [max_atoms]
            bytes array encoding the residue ids of shape [max_atoms,6]
            float array of shape [max_atoms,3] representing the 3D Cartesian
              coordinates of each atom
            float array of shape [max_atoms] storing the SASA of each atom
            float array of shape [max_atoms] storing the partial charge of each atom
    """

    try:
        pdb, ragged_structural_info = get_structural_info_from_protein(
            pdb_file,
            parser=parser,
            calculate_SASA=SASA,
            calculate_charge=charge,
            calculate_DSSP=DSSP,
            calculate_angles=angles,
            fix=fix,
            hydrogens=hydrogens,
            extra_molecules=extra_molecules,
            multi_struct=multi_struct,
            fixed_pdb_dir=fixed_pdb_dir,
        )

        mat_structural_info = pad_structural_info(
            ragged_structural_info, padded_length=padded_length
        )
    except Exception as e:
        logger.error(f"Failed to process {pdb_file}")
        logger.exception(e)
        return (None,)

    return (pdb, *mat_structural_info)


def main():
    start_time = time.time()

    args = parse_args()
    logger.setLevel(args.logging)

    if args.pdb_dir is not None:
        if args.pdb_list_file is not None:
            with open(args.pdb_list_file, "r") as f:
                pdb_list = [pdb.strip() for pdb in f.readlines()]
        else:
            logger.info(
                "Generating a list of PDBs automatically from --pdb_dir. If this is not intended, use --pdb_list_file"
            )
            pdb_list = pdb_list_from_dir(args.pdb_dir)
    else:
        pdb_list = pdb_list_from_foldcomp(args.foldcomp)

    get_structural_info_from_dataset(
        args.input_path,
        pdb_list,
        args.max_atoms,
        args.hdf5_out,
        args.output_dataset_name,
        args.parallelism,
        args.parser,
        args.angle_db,
        args.vec_db,
        args.SASA,
        args.charge,
        args.DSSP,
        args.fix_pdbs,
        args.add_hydrogens,
        not args.remove_extra_molecules,
        args.handle_multi_structures,
        args.fixed_pdb_dir,
    )

    logger.info(f"Total time = {time.time() - start_time:.2f} seconds")