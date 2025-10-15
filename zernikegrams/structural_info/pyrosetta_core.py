
import numpy as np

import numpy.typing as npt
from typing import *

import pyrosetta
init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
pyrosetta.init(init_flags, silent=True)

from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import AtomID_Map_double_t, AtomID_Map_bool_t
from pyrosetta.rosetta.core.scoring import calc_per_atom_sasa
from pyrosetta.rosetta.utility import vector1_double

import logging
logger = logging.getLogger(__name__)

from zernikegrams.structural_info.structural_info_core import get_chi_angles_and_norm_vecs


def calculate_sasa(
    pose : Pose,
    probe_radius : float=1.4
) -> AtomID_Map_double_t:
    """Calculate SASA for a pose"""
    # pyrosetta structures for returning of sasa information
    all_atoms = AtomID_Map_bool_t()
    atom_sasa = AtomID_Map_double_t()
    rsd_sasa = vector1_double()
    
    # use pyrosetta to calculate SASA per atom
    calc_per_atom_sasa(
        pose,
        atom_sasa,
        rsd_sasa,
        probe_radius
    )
    
    return atom_sasa


def get_structural_info_from_protein__pyrosetta(
    pdb_file_or_pose: Union[str, Pose],
    calculate_SASA: bool = True,
    calculate_charge: bool = True,
    calculate_DSSP: bool = True,
    calculate_angles: bool = True,
    # fix: bool = False,
    # hydrogens: bool = False,
    # extra_molecules: bool = True,
    # multi_struct: str = "warn",
    **kwargs
 ) -> Tuple[str, Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    ## comment out these three lines for faster, bulk processing with pyrosetta, and uncomment the lines at the top of the script
    # import pyrosetta
    # init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
    # pyrosetta.init(init_flags, silent=True)

    from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.moves import DsspMover

    if isinstance(pdb_file_or_pose, str):
        pose = pyrosetta.pose_from_pdb(pdb_file_or_pose)
    else:
        pose = pdb_file_or_pose

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges_pyrosetta = []
    res_ids = []

    angles = []
    vecs = []
    res_ids_per_residue = []
    
    k = 0
    
    if calculate_DSSP:
        # extract secondary structure for use in res ids
        DSSP = DsspMover()
        DSSP.apply(pose)
    
    if calculate_SASA:
        # extract physico-chemical information
        atom_sasa = calculate_sasa(pose)
    
    coords_rows = pose_coords_as_rows(pose)
    
    pi = pose.pdb_info()
    pdb = pi.name().split('/')[-1][:-4] # remove .pdb extension
    L = len(pdb)

    logger.debug(f"pdb name in protein routine {pdb} - successfully loaded pdb into pyrosetta")

    # get structural info from each residue in the protein
    for i in range(1, pose.size()+1):
        
        # these data will form the residue id
        aa = pose.sequence()[i-1]
        chain = pi.chain(i)
        resnum = str(pi.number(i)).encode()
        icode = pi.icode(i).encode()
        if calculate_DSSP:
            ss = pose.secstruct(i)
        else:
            ss = None

        if calculate_angles:
            chis, norms = get_chi_angles_and_norm_vecs(pose.residue(i))
            angles.append(chis)
            vecs.append(norms)

        res_id = np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss
        ], dtype=f'S{L}')
        res_ids_per_residue.append(res_id)
        
        for j in range(1,len(pose.residue(i).atoms())+1):

            atom_name = pose.residue_type(i).atom_name(j)
            idx = pose.residue(i).atom_index(atom_name)
            atom_id = (AtomID(idx,i))
            element = pose.residue_type(i).element(j).name
            if calculate_SASA:
                sasa = atom_sasa.get(atom_id)
            else:
                sasa = 0
            curr_coords = coords_rows[k]
            if calculate_charge:
                charge_pyrosetta = pose.residue_type(i).atom_charge(j)
            else:
                charge_pyrosetta = 0

            res_id = np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss
            ], dtype=f'S{L}')
            
            atom_names.append(atom_name.strip().upper().ljust(4)) # adding this to make sure all atom names are 4 characters long, because for some atoms (the non-residue ones, and maybe others?) it is somehow not the case
            elements.append(element)
            res_ids.append(res_id)
            coords.append(curr_coords)
            sasas.append(sasa)
            charges_pyrosetta.append(charge_pyrosetta)
            
            k += 1
            
    atom_names = np.array(atom_names,dtype='|S4')
    elements = np.array(elements, dtype='S2')
    sasas = np.array(sasas)
    coords = np.array(coords)
    charges_pyrosetta = np.array(charges_pyrosetta)
    res_ids = np.array(res_ids)
    
    res_ids_per_residue = np.array(res_ids_per_residue)
    angles = np.array(angles)
    vecs = np.array(vecs)

    # return pdb,(atom_names,elements,res_ids,coords,sasas,charges_pyrosetta,charges_amber99sb,res_ids_per_residue,angles_pyrosetta,angles,vecs)

    return pdb, (
        atom_names,
        elements,
        res_ids,
        coords,
        sasas,
        charges_pyrosetta,
        res_ids_per_residue,
        angles,
        vecs,
        np.array(
            [0] # just there for compatibility, assuming only one model in the file
        ), 
    )