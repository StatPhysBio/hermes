

import pyrosetta
from .functional import DEFAULT_CART_SCOREFXN, init_flags, repack_residues, fastrelax_full_pose, fastrelax_positions, make_mutations, find_calpha_neighbors

from typing import *


class PyrosettaPose(object):
    '''
    A wrapper for a pyrosetta object that provides a handful of common functionalities.
    The methods for this class are the same as the functions in `functional.py`, but this class abstracts away more of the inner-workings of pyrosetta,
    thus making the use of the functions more user-friendly.

    For example, this class abstracts away the conversion of standard residue identifiers (chain, resnum, icode) into the 1-indexed pose residue numbers used by pyrosetta.
    '''
    def __init__(self,
                 cart_scorefxn: Union[str, pyrosetta.ScoreFunction] = DEFAULT_CART_SCOREFXN,
                 verbose: bool = False) -> None:
        '''
        Parameters 
        ----------
        pose_or_pdbfile : Union[str, Pose]
            A path to a pdb file, or a Pose object
        cart_scorefxn : pyrosetta.ScoreFunction, optional

        Returns
        -------
        None
        '''

        # init pyrosetta
        pyrosetta.init(init_flags)
        
        self.set_cartesian_scorefxn(cart_scorefxn)
        self._verbose = verbose
    
    def load_pdb(self, pdbfile: str) -> None:
        self._pose = pyrosetta.pose_from_pdb(pdbfile)
        self._original_pose = self._pose.clone() # TODO: ensure this makes a deep copy
    
    def reset_pose(self) -> None:
        self._pose = self._original_pose.clone() # TODO: ensure this makes a deep copy
    
    def save_pdb(self, filename: str) -> None:
        '''
        Save the pose to a pdb file.

        Parameters
        ----------
        filename : str
            The name of the file to save the pose to.
        
        Returns
        -------
        None
        '''
        self._pose.dump_pdb(filename)
    
    def set_cartesian_scorefxn(self, cart_scorefxn: Union[str, pyrosetta.ScoreFunction]) -> None:
        '''
        Set the score function used for cartesian relaxations.

        Parameters
        ----------
        scorefxn : pyrosetta.ScoreFunction

        Returns
        -------
        None
        '''
        if isinstance(cart_scorefxn, str):
            self._cart_scorefxn = pyrosetta.create_score_function(cart_scorefxn)
        else:
            self._cart_scorefxn = cart_scorefxn
    
    def _get_pdb_residue_id(self, posenum: int) -> Tuple[str, int, str]:
        '''
        This function turns a posenum into its PDB residue identifier.
        Inverse of _get_posenum().
        '''
        pi = self._pose.pdb_info()
        return (pi.chain(posenum), pi.number(posenum), pi.icode(posenum))
    
    def _get_posenum(self, chain: str, resnum: int, icode: str = ' ') -> int:
        '''
        This function turns a PDB residue identifier into its unique pose residue number.
        Inverse of _get_pdb_residue_id().
        '''
        return self._pose.pdb_info().pdb2pose(chain, resnum, icode)

    
    def repack_residues(self,
                        positions: List[Union[Tuple[str, int, str], int]],
                        expect_posenums: bool = False) -> None:
        '''
        Repack the sidechains at the residues in "positions".
        By default expects each residue identified by the tuple (chain, resnum, icode).
        If expect_pose_resnums is True, then expects each residue to be identified by its 1-indexed pose number.

        Parameters
        ----------
        positions : List[Union[Tuple[str, int, str], int]]
            A list of residue identifiers.
        
        expect_posenums : bool, optional
            If True, then the keys of mutations are expected to be 1-indexed pose residue numbers. Otherwise, they are expected to be PDB residue identifiers (chain, resnum, icode).
        
        Returns
        -------
        None
        '''
        if not expect_posenums:
            positions = [self._get_posenum(*pos) for pos in positions]
        
        repack_residues(self._pose, positions, scorefxn=self._cart_scorefxn, verbose=self._verbose)


    def fastrelax_positions(self,
                            backbone_flexible_positions: List[Union[Tuple[str, int, str], int]],
                            sidechain_flexible_positions: List[Union[Tuple[str, int, str], int]],
                            expect_posenums: bool = False,
                            nrepeats: int = 1) -> None:
        '''
        Fastrelax the backbones at the residues in "backbone_flexible_positions" and the sidechains at the residues in "sidechain_flexible_positions".
        By default expects each residue identified by the tuple (chain, resnum, icode).
        If expect_pose_resnums is True, then expects each residue to be identified by its 1-indexed pose number.

        Parameters
        ----------
        backbone_flexible_positions : List[Union[Tuple[str, int, str], int]]
            A list of residue identifiers for the residues whose backbones will be flexible.
        
        sidechain_flexible_positions : List[Union[Tuple[str, int, str], int]]
            A list of residue identifiers for the residues whose sidechains will be flexible.
        
        expect_posenums : bool, optional
            If True, then the keys of mutations are expected to be 1-indexed pose residue numbers. Otherwise, they are expected to be PDB residue identifiers (chain, resnum, icode).

        nrepeats : int, optional
            The number of fastrelax iterations to perform. Higher is slower but more accurate.
        
        Returns
        -------
        None
        '''
        if not expect_posenums:
            backbone_flexible_positions = [self._get_posenum(*pos) for pos in backbone_flexible_positions]
            sidechain_flexible_positions = [self._get_posenum(*pos) for pos in sidechain_flexible_positions]
        
        fastrelax_positions(self._pose, backbone_flexible_positions, sidechain_flexible_positions, self._cart_scorefxn, nrepeats=nrepeats)
    

    def fastrelax_full_pose(self,
                            relax_backbone: bool = False,
                            nrepeats: int = 1) -> None:
        '''
        Fastrelax the full pose.

        Parameters
        ----------
        relax_backbone : bool, optional
            If True, the backbone will be flexible during the fastrelax.
        
        nrepeats : int, optional
            The number of fastrelax iterations to perform. Higher is slower but more accurate.
        
        Returns
        -------
        None
        '''
        fastrelax_full_pose(self._pose, scorefxn=self._cart_scorefxn, relax_backbone=relax_backbone, nrepeats=nrepeats)
    

    def make_mutations(self,
                       mutations: Dict[Union[Tuple[str, int, str], int], str],
                       expect_posenums: bool = False) -> None:
        '''
        Make sequence changes and repack the mutated positions
        
        mutations is a dictionary mapping from pose residue number to new 1-letter aa

        Note: should use fastrelax around the mutated residues after running this.

        Parameters
        ----------
        mutations : Dict[Union[Tuple[str, int, str], int], str]
            A dictionary mapping from residue identifiers to the new 1-letter amino acid.
        
        expect_posenums : bool, optional
            If True, then the keys of mutations are expected to be 1-indexed pose residue numbers. Otherwise, they are expected to be PDB residue identifiers (chain, resnum, icode).
        
        Returns
        -------
        None
        '''
        if not expect_posenums:
            mutations = {self._get_posenum(*pos): aa for pos, aa in mutations.items()}
        
        make_mutations(self._pose, mutations, verbose=self._verbose)
    
    
    def find_calpha_neighbors(self,
                              core_positions: List[Union[Tuple[str, int, str], int]],
                              distance_threshold: float,
                              expect_posenums: bool = False) -> Set[Union[Tuple[str, int, str], int]]:
        '''
        This function finds neighbors of the residues in "core_positions" based on Calpha-Calpha distance.
        By default expects each residue identified by the tuple (chain, resnum, icode).
        If expect_pose_resnums is True, then expects each residue to be identified by its 1-indexed pose number. In this case,
        it will also return the neighbors as 1-indexed pose residue numbers.

        Parameters
        ----------
        core_positions : List[Union[Tuple[str, int, str], int]]
            A list of residue identifiers.
        
        distance_threshold : float
            The maximum distance between Calpha atoms for two residues to be considered neighbors, in Angstroms.
        
        expect_posenums : bool, optional
            If True, then the residues in core_positions are expected to be 1-indexed pose residue numbers.
        
        Returns
        -------
        Set[Union[Tuple[str, int, str], int]]
            A set of residue identifiers for the neighbors of the residues in core_positions.
        '''

        if not expect_posenums:
            core_positions = [self._get_posenum(*pos) for pos in core_positions]
        
        neighbor_positions = find_calpha_neighbors(core_positions, distance_threshold, self._pose)

        if not expect_posenums:
            return {self._get_pdb_residue_id(pos) for pos in neighbor_positions}
        else:
            return neighbor_positions
        
    
    def make_mutations_and_fastrelax_around_it(self,
                                               mutations: Dict[Union[Tuple[str, int, str], int], str],
                                               backbone_flexible_distance_threshold: Optional[Union[float, str]] = None,
                                               sidechain_flexible_distance_threshold: Optional[Union[float, str]] = None,
                                               backbone_flexible_positions: Optional[List[Union[Tuple[str, int, str], int]]] = None,
                                               sidechain_flexible_positions: Optional[List[Union[Tuple[str, int, str], int]]] = None,
                                               nrepeats: int = 1,
                                               expect_posenums: bool = False) -> None:
        '''
        `make_mutations` and `fastrelax_positions` combined into one function.
        This function makes the mutations and then fastrelaxes the residues around the mutations, according to the specified distance thresholds or positions.

        Parameters
        ----------
        mutations : Dict[Union[Tuple[str, int, str], int], str]
            A dictionary mapping from residue identifiers to the new 1-letter amino acid.
        
        backbone_flexible_distance_threshold : Union[float, str], optional
            The distance threshold - in Angstroms - to determine which residues will have flexible backbones during the fastrelax, based on Calpha-Calpha distance with the mutated residues.
            Either a float, or the string 'all' to indicate that all residues should have flexible backbones.
        
        sidechain_flexible_distance_threshold : Union[float, str], optional
            The distance threshold - in Angstroms - to determine which residues will have flexible sidechains during the fastrelax, based on Calpha-Calpha distance with the mutated residues.
            Either a float, or the string 'all' to indicate that all residues should have flexible sidechains.
        
        backbone_flexible_positions : Optional[List[Union[Tuple[str, int, str], int]]], optional
            A list of residue identifiers for the residues whose backbones will be flexible.
            Overrides backbone_flexible_distance_threshold if not None.
        
        sidechain_flexible_positions : Optional[List[Union[Tuple[str, int, str], int]]], optional
            A list of residue identifiers for the residues whose sidechains will be flexible.
            Overrides sidechain_flexible_distance_threshold if not None.
        
        nrepeats : int, optional
            The number of fastrelax iterations to perform. Higher is slower but more accurate.
        
        expect_posenums : bool, optional
            If True, then the keys of mutations are expected to be 1-indexed pose residue numbers. Otherwise, they are expected to be PDB residue identifiers (chain, resnum, icode).
        
        Returns
        -------
        None
        '''

        if not expect_posenums:
            mutations = {self._get_posenum(*pos): aa for pos, aa in mutations.items()}
        
        self.make_mutations(mutations, expect_posenums=True)

        if backbone_flexible_positions is None:
            # using the distance threshold
            if backbone_flexible_distance_threshold is None:
                # don't fastrelax any backbones
                backbone_flexible_positions = []
            elif isinstance(backbone_flexible_distance_threshold, str):
                if backbone_flexible_distance_threshold == 'all':
                    backbone_flexible_positions = list(range(1, self._pose.total_residue() + 1))
                else:
                    raise ValueError('backbone_flexible_distance_threshold must be a float, int, or "all". It is currently "{}" of type {}.'.format(backbone_flexible_distance_threshold, type(backbone_flexible_distance_threshold)))
            elif isinstance(backbone_flexible_distance_threshold, (int, float)):
                backbone_flexible_positions = list(self.find_calpha_neighbors(list(mutations.keys()), backbone_flexible_distance_threshold, expect_posenums=True))
            else:
                raise ValueError('backbone_flexible_distance_threshold must be a float, int, or "all". It is currently "{}" of type {}.'.format(backbone_flexible_distance_threshold, type(backbone_flexible_distance_threshold)))
        else:
            if not expect_posenums:
                backbone_flexible_positions = [self._get_posenum(*pos) for pos in backbone_flexible_positions]

        if sidechain_flexible_positions is None:
            # using the distance threshold
            if sidechain_flexible_distance_threshold is None:
                # don't fastrelax any sidechains
                sidechain_flexible_positions = []
            elif isinstance(sidechain_flexible_distance_threshold, str):
                if sidechain_flexible_distance_threshold == 'all':
                    sidechain_flexible_positions = list(range(1, self._pose.total_residue() + 1))
                else:
                    raise ValueError('sidechain_flexible_distance_threshold must be a float, int, or "all". It is currently "{}" of type {}.'.format(sidechain_flexible_distance_threshold, type(sidechain_flexible_distance_threshold)))
            elif isinstance(sidechain_flexible_distance_threshold, (int, float)):
                if sidechain_flexible_distance_threshold == backbone_flexible_positions:
                    # save the time of finding the neighbors again since the threshold is the same
                    sidechain_flexible_positions = backbone_flexible_positions
                else:
                    sidechain_flexible_positions = list(self.find_calpha_neighbors(list(mutations.keys()), sidechain_flexible_distance_threshold, expect_posenums=True))
            else:
                raise ValueError('sidechain_flexible_distance_threshold must be a float, int, or "all". It is currently "{}" of type {}.'.format(sidechain_flexible_distance_threshold, type(sidechain_flexible_distance_threshold)))
        else:
            if not expect_posenums:
                sidechain_flexible_positions = [self._get_posenum(*pos) for pos in sidechain_flexible_positions]

        assert isinstance(backbone_flexible_positions, list)
        assert isinstance(sidechain_flexible_positions, list)

        if len(backbone_flexible_positions) == 0 and len(sidechain_flexible_positions) == 0:
            if self._verbose:
                print('No residues to fastrelax around. Skipping fastrelax.')
        else:
            self.fastrelax_positions(backbone_flexible_positions, sidechain_flexible_positions, expect_posenums=True, nrepeats=nrepeats)






    

        


    




