import os
import re
import tempfile
import io
import subprocess
import shutil

from contextlib import contextmanager

from collections import defaultdict
from functools import partial
from typing import (
    Dict,
    Tuple,
    List,
)

import numpy as np
import numpy.typing as npt

from zernikegrams.utils import log_config as logging

logger = logging.getLogger(__name__)

REDUCER = "reduce"


##################### Copied from https://github.com/nekitmm/DLPacker/blob/main/utils.py
# read in the charges from special file
CHARGES_AMBER99SB = defaultdict(lambda: 0)  # output 0 if the key is absent
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "charges.rtp"), "r"
) as f:
    for line in f:
        if line[0] == "[" or line[0] == " ":
            if re.match("\A\[ .{1,3} \]\Z", line[:-1]):
                key = re.match("\A\[ (.{1,3}) \]\Z", line[:-1])[1]
                CHARGES_AMBER99SB[key] = defaultdict(lambda: 0)
            else:
                l = re.split(r" +", line[:-1])
                CHARGES_AMBER99SB[key][l[1]] = float(l[3])
################################################################################


aa_to_one_letter = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# fmt: off
VEC_AA_ATOM_DICT = {
    'ARG' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','NE'], ['CD','NE','CZ']], #, ['NE','CZ','NH1']],
    'ASN' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','OD1']],
    'ASP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','OD1']],
    'CYS' : [['N','CA','CB'], ['CA','CB','SG']],
    'GLN' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','OE1']],
    'GLU' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','OE1']],
    'HIS' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','ND1']],
    'ILE' : [['N','CA','CB'], ['CA','CB','CG1'], ['CB','CG1','CD1']],
    'LEU' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'LYS' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','CE'], ['CD','CE','NZ']],
    'MET' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','SD'], ['CG','SD','CE']],
    'PHE' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'PRO' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD']],
    'SER' : [['N','CA','CB'], ['CA','CB','OG']],
    'THR' : [['N','CA','CB'], ['CA','CB','OG1']],
    'TRP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'TYR' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'VAL' : [['N','CA','CB'], ['CA','CB','CG1']]
}
# fmt: on


def get_chi_angle(
    plane_norm_1: npt.NDArray,
    plane_norm_2: npt.NDArray,
    a2: npt.NDArray,
    a3: npt.NDArray,
) -> float:
    """
    Calculates the dihedral angle given two plane norms and a2, a3 atom places
    """
    eps = 1e-6

    sign_vec = a3 - a2
    sign_with_magnitude = np.dot(sign_vec, np.cross(plane_norm_1, plane_norm_2))
    sign = sign_with_magnitude / (np.abs(sign_with_magnitude) + eps)

    dot = np.dot(plane_norm_1, plane_norm_2) / (
        np.linalg.norm(plane_norm_1) * np.linalg.norm(plane_norm_2)
    )
    chi_angle = sign * np.arccos(dot * (1 - eps))

    return np.degrees(chi_angle)


def get_chi_angles_and_norm_vecs(
    resname: str, residue: Dict[str, npt.NDArray], pdb: str
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Get chi angles and normal vectors (which are used to compute chi angles) from a residue.
    Uses the tables available at http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html

    Parameters
    ----------
    resname : str
        name of the residue
    residue : Dict[str, npt.NDArray]
        Dictionary mapping atom name to coords
    pdb: str
        Name of protein (logging use only)

    Returns
    -------
    np.ndarray
        The chi angles (4 of them)
        Will be nan if there are no vectors for the residue there

    np.ndarray
        The normal vectors (5 of them as the CB one is included)
        Will be nan if there are no vectors for the residue there
    """
    vecs = np.full((5, 3), np.nan, dtype=float)
    chis = np.full(4, np.nan, dtype=float)
    atom_names = VEC_AA_ATOM_DICT.get(resname)
    try:
        if atom_names is not None:
            for i in range(len(atom_names)):
                p1 = residue[atom_names[i][0]]
                p2 = residue[atom_names[i][1]]
                p3 = residue[atom_names[i][2]]
                v1 = p1 - p2
                v2 = p3 - p2
                # v1 = p1 - p2
                # v2 = p1 - p3
                x = np.cross(v1, v2)
                vecs[i] = x / np.linalg.norm(x)

            for i in range(len(atom_names) - 1):
                chis[i] = get_chi_angle(
                    vecs[i],
                    vecs[i + 1],
                    residue[atom_names[i][1]],
                    residue[atom_names[i][2]],
                )
    except Exception:
        logger.warning(
            (
                f"Failed to calculate chi angles/normal vectors for a {resname} in {pdb} with the error below. "
                "The remaining structural info for this protein, including other chi angles, is likely still valid."
            ),
            exc_info=True,
        )
        # returns chis and vecs empty
        vecs = np.full((5, 3), np.nan, dtype=float)
        chis = np.full(4, np.nan, dtype=float)

    return chis, vecs


def remove_waters_pdb(original: str, waterless: str, header: bool = False) -> None:
    """
    Removes water atoms from a pdb file

    Original: path to pdb file with original data
    waterless: path where new PDB file will be written
    header: add a dummy header
    """
    with io.StringIO() as buffer:
        if header:
            buffer.write("HEADER dummy header for DSSP\n")
        with open(original, "r") as f_in:
            for line in f_in.readlines():
                if "HOH" not in line:
                    buffer.write(line)

        with open(waterless, "w") as f_out:
            print(buffer.getvalue(), file=f_out)


@contextmanager
def remove_whiteout(tmp_dir):
    """
    In read-only apptainer, a "whiteout" file '.wh..wh..opq'
    is added to some directories. pdbfixer relies on enumerating and
    processing files in /opt/miniconda/envs/zernikegrams/lib/python3.10/site-packages/pdbfixer/templates.
    So we should remove this file manually and hope its not important.
    Or make a PR request to pdbfixer.

    By definition, cannot edit read-only FS. So we're doing this
    """
    import pdbfixer

    def copy_to_tmp():
        pdbfixer_true_path = pdbfixer.pdbfixer.__file__
        root = os.path.dirname(pdbfixer_true_path)

        if os.path.exists(os.path.join(root, "templates/.wh..wh..opq")):
            shutil.copytree(root, tmp_dir, dirs_exist_ok=True)

            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    if ".wh..wh..opq" in filename:
                        file_path = os.path.join(root, filename)
                        os.remove(file_path)

            open(f"{tmp_dir}/fake_file", "a").close()
            pdbfixer.pdbfixer.__file__ = f"{tmp_dir}/fake_file"

        return pdbfixer_true_path

    try:
        true_path = copy_to_tmp()
        yield
    finally:
        pdbfixer.pdbfixer.__file__ = true_path

def get_structural_info_from_protein(*args, parser='biopython', **kwargs):
    if parser == 'biopython':
        return get_structural_info_from_protein__biopython(*args, **kwargs)
    elif parser == 'pyrosetta':
        from zernikegrams.structural_info.pyrosetta_core import get_structural_info_from_protein__pyrosetta
        return get_structural_info_from_protein__pyrosetta(*args, **kwargs)
    else:
        raise ValueError(f"Unknown parser {parser}")

def get_structural_info_from_protein__biopython(
    pdb_file: str,
    calculate_SASA: bool = True,
    calculate_charge: bool = True,
    calculate_DSSP: bool = False,
    calculate_angles: bool = True,
    fix: bool = False,
    hydrogens: bool = False,
    extra_molecules: bool = True,
    multi_struct: str = "warn",
    fixed_pdb_dir: str = None,
) -> Tuple[str, Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    
    from Bio.PDB import (
        PDBParser,
        SASA,
        PDBIO
    )
    from zernikegrams.structural_info.RaSP import clean_pdb

    """
    Params:
        - pdb_file: path to pdb file
        - calculate_X: if set to false, go faster
        - if set, will fix missing atoms in pdb
        - hydrogens: if set, will add hydrogens to pdb
        - extra_molecules: if set, extra_molecules are left in
        - multi_struct: Behavior for handling PDBs with multiple structures
        - fixed_pdb_dir: directory to save fixed pdbs

    Returns:
        Tuple of (pdb, (atom_names, elements, res_ids, coords, sasas, charges, res_ids_per_residue, angles, norm_vecs, is_multi_model [1 or 0] ))

    By default, biopyton selects only atoms with the highest occupancy, thus behaving like pyrosetta does with the flag "-ignore_zero_occupancy false"
    """
    parser = PDBParser(QUIET=True)

    pdb_name = pdb_file[:-4]
    L = len(pdb_name)

    if fix or hydrogens:
        tmp = tempfile.NamedTemporaryFile()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with remove_whiteout(tmp_dir):
                clean_pdb(pdb_file, tmp.name, REDUCER, hydrogens, extra_molecules)

        remove_waters_pdb(original=tmp.name, waterless=tmp.name, header=True)

        pdb_file = tmp.name

    structure = parser.get_structure(pdb_name, pdb_file)


    # assume the pdb name was provided as id to create the structure
    pdb = structure.get_id()
    pdb = os.path.basename(pdb).replace(
        "_", "-"
    )  # "_" is reserved for the res_id delimiter later in pipeline

    if fixed_pdb_dir is not None:
        if not os.path.exists(fixed_pdb_dir):
            os.makedirs(fixed_pdb_dir)
        pdbio = PDBIO()
        pdbio.set_structure(structure)
        save_path = os.path.join(fixed_pdb_dir, f"{pdb}.pdb")
        pdbio.save(save_path)

    models = list(structure.get_models())
    if len(models) != 1:
        if multi_struct == "crash":
            raise ValueError(f"More than 1 model found for {pdb_file}")
        else:
            if multi_struct == "warn":
                logger.warn(
                    f"{len(models)} models found for {pdb_file}. Setting structure to the first model."
                )
            structure = models[0]

    if calculate_SASA:
        # Calculates SASAs with biopython; each atom will have a .sasa attribute
        SASA.ShrakeRupley().compute(structure, level="A")

    calculate_DSSP = False # not supporting DSSP for now to make installation lighter
    if calculate_DSSP:
        DSSP = "mkdssp"
        DSSP_VERSION = (
            subprocess.run(
                f"{DSSP} --version".split(), capture_output=True
            )
            .stdout.decode()
            .strip(f"mkdssp version ")
        )
        from Bio.PDB.DSSP import dssp_dict_from_pdb_file
        dssp_dict, _dssp_keys = dssp_dict_from_pdb_file(pdb_file, DSSP=DSSP, dssp_version=DSSP_VERSION)
    else:
        dssp_dict = {}

    if fix or hydrogens:
        tmp.close()

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges = []
    res_ids = []

    angles = []
    vecs = []
    res_ids_per_residue = []

    chi_atoms = {}

    def pad_for_consistency(string):
        return string.ljust(4, " ")

    # get structural info from each residue in the protein
    for atom in structure.get_atoms():
        atom_full_id = atom.get_full_id()

        chain = atom_full_id[2]
        resnum = atom_full_id[3][1]
        icode = atom_full_id[3][2]
        atom_name_unpadded = atom.get_name()
        atom_name = pad_for_consistency(atom_name_unpadded)
        element = atom.element
        coord = atom.get_coord()
        ss = dssp_dict.get((chain, (" ", resnum, " ")), ("_", "null"))[1]

        residue = atom.get_parent().resname
        if residue in aa_to_one_letter:
            aa = aa_to_one_letter[residue]
        else:
            aa = "Z"

        res_id = np.array([aa, pdb, chain, resnum, icode, ss], dtype=f"S{L}")

        res_key = tuple(res_id)
        if res_key not in chi_atoms:
            chi_atoms[res_key] = residue, {}
        chi_atoms[res_key][1][atom_name_unpadded] = coord

        atom_names.append(atom_name)
        elements.append(element)
        res_ids.append(res_id)
        coords.append(coord)
        if calculate_SASA:
            sasas.append(atom.sasa)

        if calculate_charge:
            res_charges = CHARGES_AMBER99SB[residue]
            if isinstance(res_charges, dict):
                charge = res_charges[atom_name_unpadded.upper()]
            elif isinstance(res_charges, float) or isinstance(res_charges, int):
                charge = res_charges
            else:
                raise ValueError(
                    f"Unknown charge type: {type(res_charges)}. Something must be wrong."
                )
            charges.append(charge)

    if calculate_angles:
        for res_key, (resname, atoms) in chi_atoms.items():
            res_ids_per_residue.append(np.array([*res_key], dtype=f"S{L}"))
            chis, norms = get_chi_angles_and_norm_vecs(resname, atoms, pdb)
            angles.append(chis)
            vecs.append(norms)

    atom_names = np.array(atom_names, dtype="|S4")
    elements = np.array(elements, dtype="S1")
    coords = np.array(coords)
    res_ids = np.array(res_ids)
    sasas = np.array(sasas)
    charges = np.array(charges)
    res_ids_per_residue = np.array(res_ids_per_residue)
    angles = np.array(angles)
    vecs = np.array(vecs)

    return pdb, (
        atom_names,
        elements,
        res_ids,
        coords,
        sasas,
        charges,
        res_ids_per_residue,
        angles,
        vecs,
        np.array(
            [0 if len(models) == 1 else 1]
        ),  # necessary to compute number of pdbs with multiple structures
    )


def pad(arr: npt.NDArray, padded_length: int = 100) -> npt.NDArray:
    """
    Pad an array long axis 0

    Parameters
    ----------
    arr : npt.NDArray
    padded_length : int

    Returns
    -------
    npt.NDArray
    """
    # get dtype of input array
    dt = arr.dtype

    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        logger.warn(
            f"Error: Padded length of {padded_length}",
            f"is smaller than original length of array {orig_length}",
        )

    # create padded array
    padded_shape = (padded_length, *shape)
    mat_arr = np.zeros(padded_shape, dtype=dt)

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)

    return mat_arr


def pad_structural_info(
    ragged_structure: Tuple[npt.NDArray, ...], padded_length: int = 100
) -> List[npt.NDArray]:
    """
    Pad structural into arrays
    """
    pad_custom = partial(pad, padded_length=padded_length)
    mat_structure = list(map(pad_custom, ragged_structure))

    return mat_structure