

def rename_resnums(pdbfile):
    '''
    Renames the resnums in a pdb file to be consecutive integers starting from 1.
    Saves the new pdb file with the resnums renamed and a json file with the renaming information.
    '''
    from Bio.PDB import PDBParser
    from Bio.PDB import PDBIO
    import json
    from copy import deepcopy
    parser = PDBParser()
    structure = parser.get_structure('X', pdbfile)
    assert len(structure) == 1, 'More than one model in the pdb file. This is not supported'
    renaming_dict = {}
    for chain in structure[0]:
        renaming_dict[chain.id] = {}
        old_ids = []
        max_resnum = len(chain) * 10 + 1000 # just a large number to ensure that the resnums are unique, it's not ~guaranteed~ to always work
        for i, residue in enumerate(chain):
            old_ids.append(deepcopy(residue.id))
            residue.id = (' ', max_resnum+i+1, ' ') # placeholder value

        for i, (residue_id, residue) in enumerate(zip(old_ids, chain)):
            hetero_flag = residue_id[0]
            resnum = residue_id[1]
            icode = residue_id[2]
            residue.id = (hetero_flag, i+1, ' ')
            renaming_dict[chain.id][i+1] = (resnum, icode)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile.replace('.pdb', '_renamed_resnums.pdb'))
    with open(pdbfile.replace('.pdb', '_renaming.json'), 'w') as f:
        json.dump(renaming_dict, f, indent=4)
