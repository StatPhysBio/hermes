
import os

files = [file for file in os.listdir('./') if file.endswith('.npz.gz')]

NAME_MAP = {
    'HCNN_biopython_proteinnet_extra_mols_0p00': 'hermes_bp_000',
    'HCNN_biopython_proteinnet_extra_mols_0p50': 'hermes_bp_050',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': 'hermes_py_000',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': 'hermes_py_050',
    'HCNN_pyrosetta_untrained_extra_mols_0p00': 'hermes_py_000_untrained',
    'HCNN_pyrosetta_untrained_extra_mols_0p50': 'hermes_py_050_untrained',
}

for file in files:
    for name, new_name in NAME_MAP.items():
        if name in file:
            newfile = file.replace(name, new_name)
            os.rename(file, newfile)
            break
