
import os

MAP_NAME = {
    'HCNN_biopython_proteinnet_extra_mols_0p00__all': 'hermes_bp_000',
    'HCNN_biopython_proteinnet_extra_mols_0p50__all': 'hermes_bp_050',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00__all': 'hermes_py_000',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50__all': 'hermes_py_050'
}

for fold_dir in ['s0_fold1', 's0_fold2', 's0_fold3', 's1_fold1', 's1_fold2', 's1_fold3', 's2_fold1', 's2_fold2', 's2_fold3']:
    for file in os.listdir(os.path.join(fold_dir, 'configs')):
        for name, new_name in MAP_NAME.items():
            if name in file:
                newfile = file.replace(name, new_name)
                os.rename(os.path.join(fold_dir, 'configs', file), os.path.join(fold_dir, 'configs', newfile))
                break

