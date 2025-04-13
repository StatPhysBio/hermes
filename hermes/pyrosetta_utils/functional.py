

import pyrosetta
from pyrosetta.rosetta import core, protocols
import numpy as np

from typing import *

# same flags as zernikegrams repo. TODO: import them from zernikegrams repo instead of storing them here
init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'

DEFAULT_CART_SCOREFXN = 'ref2015_cart.wts'


def repack_residues(
    pose,
    positions, # 1-indexed pose numbering
    scorefxn,
    verbose=False
):
    ''' Repack the sidechains at the residues in "positions" 
    '''

    tf = core.pack.task.TaskFactory()
    tf.push_back(core.pack.task.operation.InitializeFromCommandline()) # use -ex1 and -ex2 rotamers if requested

    # dont allow any design
    op = core.pack.task.operation.RestrictToRepacking()
    tf.push_back(op)

    # freeze residues not in the positions list
    op = core.pack.task.operation.PreventRepacking()
    for i in range(1,pose.size()+1):
        if i not in positions:
            op.include_residue(i)
        else:
            print('repacking at residue', i)
    tf.push_back(op)
    packer = protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)
    packer.score_function(scorefxn)

    if verbose:
        # show the packer task
        print(tf.create_task_and_apply_taskoperations(pose))
    
    packer.apply(pose)


def fastrelax_full_pose(pose,
                        scorefxn,
                        relax_backbone = False,
                        nrepeats = 1):
    
    resnums = list(range(1, pose.size()+1))

    fastrelax_positions(pose,
                        resnums if relax_backbone else [],
                        resnums,
                        scorefxn = scorefxn,
                        nrepeats = nrepeats)


def fastrelax_positions(
        pose,
        backbone_flexible_positions,
        sidechain_flexible_positions,
        scorefxn,
        nrepeats = 1,
):
    ''' "Relax" iterates between repacking and gradient-based minimization
    here we are doing "cartesian" relax, which allows bond lengths and angles to change slightly
    (the positions of the atoms are the degrees of freedom, rather than the internal coordinates)
    So the scorefxn should have terms to constrain these near ideal values, eg ref2015_cart.wts
    '''
    # movemap:
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)

    for i in backbone_flexible_positions:
        mm.set_bb(i, True)

    for i in sidechain_flexible_positions:
        mm.set_chi(i, True)

    fr = protocols.relax.FastRelax(scorefxn_in=scorefxn,
                                   standard_repeats=nrepeats)
    fr.cartesian(True)
    fr.set_movemap(mm)
    fr.set_movemap_disables_packing_of_fixed_chi_positions(True)

    # For non-Cartesian scorefunctions, use "dfpmin_armijo_nonmonotone"
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)


def find_calpha_neighbors(
    core_positions,
    distance_threshold,
    pose
):
    ''' This function finds neighbors of the residues in "core_positions" based on Calpha-Calpha distance
    '''
    # include all the 'core' positions as neighbors (of themselves, e.g.)
    nbr_positions = set(core_positions)

    distance_threshold_squared = distance_threshold**2
    
    for i in range(1, pose.size()+1): # stupid Rosetta 1-indexing
        rsd1 = pose.residue(i)
        try:
            rsd1_CA = rsd1.xyz("CA") # access by string is a *little* slow; could use integer indices
            for j in core_positions:
                rsd2 = pose.residue(j)
                if rsd1_CA.distance_squared(rsd2.xyz("CA")) <= distance_threshold_squared:
                    nbr_positions.add(i)
                    break
        except:
            continue
    return nbr_positions


def get_pdb_residue_info(
    pose,
    posenum,
):
    '''
    This function turns a pose residue number into its PDB residue identifier.
    Inverse of get_pose_residue_number().
    '''
    pi = pose.pdb_info()
    return (pi.chain(posenum), pi.number(posenum), pi.icode(posenum))


def get_pose_residue_number(
    pose,
    chain,
    resnum,
    icode=' ',
):
    '''
    This function turns a PDB residue identifier into its unique pose residue number.
    Inverse of get_pdb_residue_info().
    '''
    return pose.pdb_info().pdb2pose(chain, resnum, icode)


def make_mutations(
    pose,
    mutations,
    verbose=False
):
    ''' Make sequence changes and repack the mutated positions
    
    mutations is a dictionary mapping from pose residue number to new 1-letter aa
    mutations is 1-indexed
    
    Note that we don't specify the score function! I guess the packer here is
    using a default fullatom scorefunction... Huh

    Note: should use fastrelax aorund the mutated residues after running this.
    '''
    oldseq = pose.sequence()

    tf = core.pack.task.TaskFactory()
    #tf.push_back(core.pack.task.operation.InitializeFromCommandline()) # potentially include extra rotamers

    # freeze non-mutated
    op = core.pack.task.operation.PreventRepacking()
    for i in range(1,pose.size()+1):
        if i not in mutations:
            op.include_residue(i)
    tf.push_back(op)

    # force desired sequence at mutations positions
    for i, aa in mutations.items():
        op = core.pack.task.operation.RestrictAbsentCanonicalAAS()
        op.include_residue(i)
        op.keep_aas(aa)
        tf.push_back(op)
        if verbose:
            print('make mutation:', i, oldseq[i-1], '-->', aa)

    packer = protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    # show the packer task
    if verbose:
        print(tf.create_task_and_apply_taskoperations(pose))

    packer.apply(pose)


def get_dG_of_binding(pdbfile_or_pose: Union[str, pyrosetta.rosetta.core.pose.Pose],
                      chains_one_list: List[str],
                      chains_two_list: List[str],
                      pyrosetta_was_already_initialized: bool = False,
                      ignore_waters: str = True,
                      add_hydrogens: str = True,
                      relax_sidechains_before_scoring: str = True):
    

    if not pyrosetta_was_already_initialized:
        # init pyrosetta with desired flags
        if add_hydrogens and ignore_waters:
            flags = default_flags
        elif add_hydrogens and not ignore_waters:
            flags = wet_flags
        else:
            raise NotImplementedError('add_hydrogens=False is not yet supported')
        pyrosetta.init(flags)


    if isinstance(pdbfile_or_pose, str):
        complex_pose = pyrosetta.pose_from_pdb(pdbfile_or_pose)
    elif isinstance(pdbfile_or_pose, pyrosetta.rosetta.core.pose.Pose):
        complex_pose = pdbfile_or_pose
    else:
        raise ValueError('pdbfile_or_pose must be a str or a Pose object')


    # split the pose into separate chains
    chains = complex_pose.split_by_chain()
    chain_letter_to_idx = {chain.pdb_info().chain(1): idx+1 for idx, chain in enumerate(chains)}

    # create poses for each binding partner
    pose_one = pyrosetta.Pose()
    for chain_one in chains_one_list:
        chain_one_idx = chain_letter_to_idx[chain_one]
        pose_one.assign(chains[chain_one_idx])
        pose_one.append_pose_by_jump(chains[chain_one_idx], 1)

    pose_two = pyrosetta.Pose()
    for chain_two in chains_two_list:
        chain_two_idx = chain_letter_to_idx[chain_two]
        pose_two.assign(chains[chain_two_idx])
        pose_two.append_pose_by_jump(chains[chain_two_idx], 1)
    
    
    scorefxn = pyrosetta.create_score_function(DEFAULT_CART_SCOREFXN)
    
    if relax_sidechains_before_scoring:
        fastrelax_full_pose(complex_pose, scorefxn, relax_backbone = True, nrepeats = 2)
        fastrelax_full_pose(pose_one, scorefxn, relax_backbone = True, nrepeats = 2)
        fastrelax_full_pose(pose_two, scorefxn, relax_backbone = True, nrepeats = 2)
    
    # score the poses - currently scoring the whole complex, not just the interface!
    G_complex = scorefxn(complex_pose)
    G_one = scorefxn(pose_one)
    G_two = scorefxn(pose_two)

    # get the dG of binding
    dG_binding = G_complex - (G_one + G_two)

    return dG_binding



def compute_ddG(pose, mutations):
    '''

    From the RDE-PPI paper:
        Rosetta (Alford et al., 2017; Leman et al., 2020) The version we used is 2021.16, and the scoring function is ref2015_cart.
        Every protein structures in the SKEMPI2 dataset are first pre-processed using the relax application.
        The mutant structure is built by cartesian_ddg.
        The binding free energies of both wild-type and mutant structures are predicted by interface_energy (dG_separated/dSASAx100).
        Finally, the binding ∆∆G is calculated by substracting the binding energy of the wild-type structure from the binding energy of the mutant.

    For cartesian_ddg, see this (https://www.rosettacommons.org/docs/latest/cartesian-ddG) which says we need to relax with unrestrained backbone and side-chains first.
    It also seems to indicate that cartesian_ddg only exists within Rosetta as a command-line function, and not pyrosetta... we could ask Phil, because he developed cartesian_ddg.

    From the paper introducing cartesian_ddg:
        Mutational ∆∆G calculation. In this study, we introduce a new Rosetta protocol for calculating the effect of single-site mutations on protein folding free energy (mutational ∆∆G). This test is only used for validation. We have found this protocol is more accurate (see RESULTS), more robust, and >10 fold faster than the best performing protocol in the previously published study (19).  One of the key challenges in predicting the effects of mutation is accounting for backbone changes in mutants.  Letting all backbones move is challenging as the large conformational changes can dominate the relatively small energetic signal from a single point mutant.  Consequently, previous efforts required many replicates with strong constraints to maintain reasonable prediction accuracy.  Instead, we use Cartesian-space refinement to allow small local backbone movement in the course of refinement.  Initially, the wild-type protein is relaxed in Cartesian space using the FastRelax protocol. Then, the best rotameric side-chain conformation for the wild-type and each mutation is determined, and is followed by FastRelax in Cartesian space, only allowing movement in a locally restricted region around the residue of interest: all side-chains within 6 Å of the mutated residue, and the backbone of a threeresidue window around the mutated residue are allowed to move. The energy gap between the refined mutant structure and the refined wild-type structure, multiplied by an energy-function-specific scaling factor, becomes the predicted mutational ∆∆G. Scaling factors are introduced to fit the overall scale of estimated values to actual experimental free energies measured in kcal/mole. A least-squares fit determined a scaling factor for talaris2014 of 1.0/1.84 and for opt-nov15 of 1.0/2.94. Evaluation of mutational ∆∆G accuracy uses two metrics: a) Pearson correlation coefficient between predicted versus experimental values, and b) fraction of correct classification of stabilizing / destabilizing mutations. Correct assignment of stabilizing mutation implies predicting ∆∆G < -1.0 for a mutation with experimental ∆∆G < -1.0, and correct assignment of destabilizing ones is the opposite (∆∆G > 1.0) 
   
    Maybe I can just replicate the protocol with pyrosetta commands. However, I would need to change it to be about the binding ddG instead of the stability ddG, following what RDE-PPI did.
    They separate the proteins and compute their energy (dG), and then use also SASA to rescale the dG (?) maybe to normalize the energy by the interface size in some way?
    Maybe we can also do something similar with HCNN? If we want to treat them like real binding energies.
    Discuss this during protein meeting.
    
    '''

    raise NotImplementedError('This function is not yet implemented')

    ## fastrelax the whole thing, backbone included
    fastrelax_full_pose(pose,
                        DEFAULT_CART_SCOREFXN,
                        relax_backbone = True,
                        nrepeats = 1)

    ## use cartesian_ddG
