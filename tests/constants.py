TEMPDIR = 'temp'
OUTDIR = 'output'
MODEL_VERSION = 'hermes_bp_050' # biopython model, used by default in tests
PYROSETTA_MODEL_VERSION = 'hermes_py_050' # needed for tests that take a pyrosetta Pose as input, which biopython models cannot parse
SEED = 42 # biopython models place missing atoms and hydrogens randomly, so seed them to make outputs comparable across runs
