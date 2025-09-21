# This code is taken from the preprocessing routine of the RaSP model
# https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg/blob/main/src/pdb_parser_scripts/clean_pdb.py
# under the apache 2 license https://www.apache.org/licenses/LICENSE-2.0

import os
import subprocess
import tempfile

import Bio.PDB
import Bio.PDB.Polypeptide
import Bio.SeqIO
import pdbfixer
import simtk
import openmm
import openmm.app

PDBIO = Bio.PDB.PDBIO()
PDB_PARSER = Bio.PDB.PDBParser(PERMISSIVE=0, QUIET=True)


class NonHetSelector(Bio.PDB.Select):
    """Remove HET atoms and choose first conformation of disordered atoms"""

    def accept_residue(self, residue):
        norm_res_bool = residue.get_resname() in [
            pdbfixer.pdbfixer.substitutions[key]
            for key in pdbfixer.pdbfixer.substitutions
        ]
        abnorm_res_bool = residue.get_resname() in [
            key for key in pdbfixer.pdbfixer.substitutions
        ]
        return (
            norm_res_bool or abnorm_res_bool
        )  # Accept abnorm since they are converted later

    def accept_atom(self, atom):
        return (
            not atom.is_disordered()
            or atom.get_altloc() == "A"
            or atom.get_altloc() == "1"
        ) and atom.id[0] in ["C", "H", "N", "O", "S", "P"]

class FirstDisorderedSelector(Bio.PDB.Select):
    """Choose first conformation of disordered atoms"""

    def accept_atom(self, atom):
        return (
            not atom.is_disordered()
            or atom.get_altloc() == "A"
            or atom.get_altloc() == "1"
        )


class PDBFixerResIdentifiabilityIssue(Exception):
    pass


def _step_0_remove_hydrogens(pdb_input_filename, temp0):
    with open(pdb_input_filename, "r") as f:
        lines = f.readlines()

    atom_number_counter = 0
    with open(temp0.name, "w") as f:

        for line in lines:

            if line[0:6] in ["ANISOU"]:
                # change atom number in line with atom_number_counter
                line = line[:6] + str(atom_number_counter).rjust(5) + line[11:]
                f.write(line)

            elif line[0:6] in ["ATOM  ", "HETATM"]:

                if line[77] != "H": # 77 is the element column; used to be 13 for the middle of the atom name, but that's not reliable
                    atom_number_counter += 1
                    # change atom number in line with atom_number_counter
                    line = line[:6] + str(atom_number_counter).rjust(5) + line[11:]
                    f.write(line)
                
            else:
                f.write(line)

    temp0.flush()
    return temp0


def _step_1_reduce(
    reduce_executable,
    pdb_input_filename,
    pdbid,
    temp1,
):

    # Add hydrogens using reduce program
    command = [
        reduce_executable,
        "-BUILD",
        "-DB",
        os.path.join(
            os.path.dirname(os.path.dirname(reduce_executable)),
            "reduce_wwPDB_het_dict.txt",
        ),
        "-Quiet",
        pdb_input_filename,
    ]
    with open(os.devnull, "w") as devnull:
        error_code = subprocess.Popen(command, stdout=temp1, stderr=devnull).wait()
    temp1.flush()

    first_model = PDB_PARSER.get_structure(pdbid, temp1.name)[0]

    return first_model


def _step_3_pdbfixer(first_model, temp3, hydrogens):
    for chain in first_model:
        for res in chain:
            for atom in res:
                atom.set_altloc(" ")
    PDBIO.set_structure(first_model)
    PDBIO.save(temp3)
    temp3.flush()

    # Use PDBFixer to fix common PDB errors
    fixer = pdbfixer.PDBFixer(temp3.name)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if hydrogens:
        fixer.addMissingHydrogens(7.0)
    return temp3, fixer


def _step_4_fix_numbering(fixer, temp3, temp4):
    simtk.openmm.app.PDBFile.writeFile(
        fixer.topology, fixer.positions, temp4, keepIds=False
    )
    temp4.flush()
    # Fix IDs manually since pdbfixer does not preserve insertion codes
    structure_before = PDB_PARSER.get_structure(temp3.name, temp3.name)
    structure_after = PDB_PARSER.get_structure(temp4.name, temp4.name)
    residues_before = []
    atoms_before = [  ]
    for chain in structure_before[0]:
        residues_before.append(chain.get_list())
        for res in chain:
            for atom in res:
                atoms_before.append(atom)
    residues_after = []
    atoms_after = []
    for chain in structure_after[0]:
        residues_after.append(chain.get_list())
        for res in chain:
            for atom in res:
                atoms_after.append(atom)
    chain_counter = ""
    for i, chain in enumerate(structure_before[0]):
        try:
            if (
                structure_after[0].get_list()[i].id
                != structure_before[0].get_list()[i].id
            ):
                try:
                    # HACK BECAUSE OF https://github.com/biopython/biopython/issues/1551
                    # Essentially, a new change in biopython prevents you from changing the
                    # id to an already existing id which broke this initial script.
                    # Therefore, we now change the ids to "change_counter" which will never look
                    # like a canonical chainid.
                    structure_after[0][
                        structure_before[0].get_list()[i].id
                    ].id = chain_counter
                    chain_counter += "KK"
                except KeyError:
                    pass
                structure_after[0].get_list()[i].id = (
                    structure_before[0].get_list()[i].id
                )
            if len(residues_before[i]) != len(residues_after[i]):
                raise PDBFixerResIdentifiabilityIssue()

        # When exceeding chainid Z, pdbfixer has discarded it, whereas biopython has not.
        # For simplicity, we just discard it as well and pretend it does not exist.
        # This is a very rare instance and will likely never be a problem unless you
        # are extremely unlucky to work with huge proteins where you care about the
        # truncation.
        except IndexError:
            continue

        counter = 99999  # A large residue number that will never exist in a pdb.
        for res1, res2 in zip(residues_before[i], residues_after[i]):
            assert (
                res1.get_resname().strip() == res2.get_resname().strip()
                or pdbfixer.pdbfixer.substitutions[res1.get_resname()].strip()
                == res2.get_resname().strip()
            )
            if res2.id != res1.id:
                try:
                    # Similar issue as previous hack https://github.com/biopython/biopython/issues/1551
                    structure_after[0][chain.get_id()][res1.id].id = (
                        " ",
                        counter,
                        " ",
                    )
                except KeyError:
                    pass
                res2.id = res1.id
                counter += 1
        for atom1, atom2 in zip(atoms_before, atoms_after):
            atom2.bfactor = atom1.bfactor

    return structure_after


def clean_pdb(pdb_input_filename: str, out_path: str, reduce_executable: str, hydrogens: bool, extra_molecules: bool):
    """
    Function to clean pdbs using reduce and pdbfixer.

    Parameters
    ----------
    pdb_input_filename: str
        PDB filename
    out_dir: str
        Output directory.
    reduce_executable: str
        Path to the reduce executable
    Hydrogens: bool
        include hydrogens
    extra_molecules: bool
        include extra_molecules (whatever is flagged as hetero)
    """

    pdbid = pdb_input_filename.split("/")[-1].split(".pdb")[0]

    # Step 0: remove hydrogen atoms if they already exist, since reduce will add duplicates and that can cause issues
    with tempfile.NamedTemporaryFile(mode="wt", delete=True) as temp0:
        temp0 = _step_0_remove_hydrogens(pdb_input_filename, temp0)

        # Step 1: Add hydrogens using reduce program
        with tempfile.NamedTemporaryFile(mode="wt", delete=True) as temp1:

            if hydrogens:
                first_model = _step_1_reduce(
                    reduce_executable, temp0.name, pdbid, temp1
                )
            else:
                first_model = PDB_PARSER.get_structure(pdbid, temp0.name)[0]

            # Step 2: NonHetSelector filter
            with tempfile.NamedTemporaryFile(mode="wt", delete=True) as temp2:

                if not extra_molecules:
                    PDBIO.set_structure(first_model)
                    PDBIO.save(temp2, select=NonHetSelector())
                    temp2.flush()
                    first_model = PDB_PARSER.get_structure(temp2.name, temp2.name)[0]
                else:
                    PDBIO.set_structure(first_model)
                    PDBIO.save(temp2, select=FirstDisorderedSelector())
                    temp2.flush()
                    first_model = PDB_PARSER.get_structure(temp2.name, temp2.name)[0]

                # Step 3: Replace altloc chars to " " and use pdbfixer
                with tempfile.NamedTemporaryFile(mode="wt", delete=True) as temp3:
                    temp_3, fixer = _step_3_pdbfixer(first_model, temp3, hydrogens)

                    # Step 4: Correct for pdbfixer not preserving insertion codes
                    with tempfile.NamedTemporaryFile(mode="wt", delete=True) as temp4:
                        structure_after = _step_4_fix_numbering(fixer, temp3, temp4)
                        with open(out_path, "w") as outpdb:
                            PDBIO.set_structure(structure_after[0])
                            PDBIO.save(outpdb)