
from hermes.inference import run_hermes_on_pdbfile_or_pyrosetta_pose

## from pdbfile
df, embeddings = run_hermes_on_pdbfile_or_pyrosetta_pose('hermes_bp_050', 'pdbs/5jzy.pdb', chain_and_sites_list=[('L', ['14', '14-A', '14-B', '14-D'])], request=['probas', 'embeddings'])
df, embeddings = run_hermes_on_pdbfile_or_pyrosetta_pose('hermes_bp_050', 'pdbs/5jzy.pdb', chain_and_sites_list=[('L', ['14', '14-A', '14-B', '14-D'])], request=['embeddings', 'probas'])