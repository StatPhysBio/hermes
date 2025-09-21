
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch, Selection


from hermes.utils.protein_naming import aa_to_one_letter

# Define cutoff (in Ångstrom)
cutoff = 4.0

df_skempi = pd.read_csv("skempi_filtered_ddg.csv")
pdbs_dir = "/gscratch/stf/gvisan01/skempi/pyrosetta_mutated_pdbs/"
pdbs_and_chains = df_skempi['pdb'].unique()

pdbs_to_save = []
chains_to_save = []
aas_to_save = []
resnums_to_save = []

for pdb_and_chain in tqdm(pdbs_and_chains):
    pdb = pdb_and_chain.split("_")[0]
    chains_1_ids = list(pdb_and_chain.split("_")[1])
    chains_2_ids = list(pdb_and_chain.split("_")[2])

    # Load structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb, os.path.join(pdbs_dir, f'{pdb}.pdb'))

    # # remove all waters from structure
    for model in structure:
        for chain in model:
            residues_to_delete = [res for res in chain if res.id[0] == 'W']
            for res in residues_to_delete:
                chain.detach_child(res.id)

    # Extract chains
    chains_1 = [chain for model in structure for chain in model if chain.id in chains_1_ids]
    chains_2 = [chain for model in structure for chain in model if chain.id in chains_2_ids]

    # Flatten all atoms for fast search
    atoms_1 = Selection.unfold_entities(chains_1, 'A')  # 'A' for atoms
    atoms_2 = Selection.unfold_entities(chains_2, 'A')

    # Build Neighbor Search object
    ns = NeighborSearch(atoms_2)

    # Find interface residues
    interface_residues_1 = set()
    interface_residues_2 = set()

    for atom in atoms_1:
        neighbors = ns.search(atom.coord, cutoff)
        if neighbors:
            interface_residues_1.add(atom.get_parent())  # Parent is residue
            for neighbor_atom in neighbors:
                interface_residues_2.add(neighbor_atom.get_parent())

    interface_residues = list(interface_residues_1.union(interface_residues_2))

    # iterate over residues, and save them
    for interface_residue in interface_residues:
        resname = interface_residue.resname
        hetero_flag, resnum, icode = interface_residue.id
        chain = interface_residue.get_parent().id

        if hetero_flag == 'W':
            print('water!')

        # Check if the residue is a standard amino acid
        if resname in aa_to_one_letter:
            # skip non-empy icodes because they are not accounted for by some of our scripts. shouldn't be many anyway
            if icode == ' ':
                one_letter_resname = aa_to_one_letter[resname]
                pdbs_to_save.append(pdb)
                chains_to_save.append(chain)
                aas_to_save.append(one_letter_resname)
                resnums_to_save.append(resnum)

# Save the results to a CSV file
df = pd.DataFrame({
    'pdb': pdbs_to_save,
    'chain': chains_to_save,
    'resname': aas_to_save,
    'resnum': resnums_to_save
})
df.to_csv("interface_residues.csv", index=False)
print("Interface residues saved to interface_residues.csv")

# now expand the file to include all possible mutations, to be used by our proteinmpnn script

all_aas = list(aa_to_one_letter.values())

df = pd.read_csv("interface_residues.csv")
df_expanded = []
for i, row in df.iterrows():
    aa_wt = row['resname']
    for aa_mt in all_aas:
        if aa_mt != aa_wt:
            new_row = row.copy()
            new_row['mutant'] = f'{aa_wt}{row["resnum"]}{aa_mt}'
            df_expanded.append(new_row)

df_expanded = pd.DataFrame(df_expanded)
df_expanded.to_csv("interface_residues_expanded.csv", index=False)
print("Expanded interface residues saved to interface_residues_expanded.csv")
