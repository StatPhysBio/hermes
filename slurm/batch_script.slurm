#!/bin/bash
## Job Name
#SBATCH --job-name=zgrams_casp12_testing_10_5_ks_10_square_C,N,O,S,H,SASA,charge_0.5_10009_pyrosetta_None_0
## Allocation Definition
## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).
#SBATCH --account=stf
#SBATCH --partition=cpu-g2-mem2x
## Resources
## Total number of Nodes
#SBATCH --nodes=1
## Number of cores per node
#SBATCH --ntasks-per-node=1
## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.
#SBATCH --time=23:00:00
## Memory per node. It is important to specify the memory since the default memory is very small.
## For mox, --mem may be more than 100G depending on the memory of your nodes.
## For ikt, --mem may be 58G or more depending on the memory of your nodes.
## See above section on "Specifying memory" for choices for --mem.
#SBATCH --mem=96G
## Specify the working directory for this job
#SBATCH --chdir=/mmfs1/gscratch/spe/gvisan01/protein_holography-pytorch/slurm
##turn on e-mail notification
## export all your environment variables to the batch job session
#SBATCH --export=all
#SBATCH -e logs/zgrams_casp12_testing_10_5_ks_10_square_C,N,O,S,H,SASA,charge_0.5_10009_pyrosetta_None_0.err
#SBATCH -o logs/zgrams_casp12_testing_10_5_ks_10_square_C,N,O,S,H,SASA,charge_0.5_10009_pyrosetta_None_0.log

cd /gscratch/spe/gvisan01/protein_holography-pytorch/protein_holography_pytorch/preprocessing_faster

python -u get_zernikegrams.py \
                      --hdf5_in /gscratch/stf/gvisan01/casp12/neighborhoods/neighborhoods-no_central_res-casp12_testing-r_max=10-charges=pyrosetta-noise=0.5_10009-relaxation=None_0.hdf5 \
                      --hdf5_out /gscratch/stf/gvisan01/casp12/zernikegrams/zernikegrams-no_central_res-casp12_testing-r_max=10-l_max=5-radial_func_mode=ks-radial_func_max=10-channels=C,N,O,S,H,SASA,charge-rst_normalization=square-charges=pyrosetta-noise=0.5_10009-relaxation=None_0.hdf5 \
                      --r_max 10 \
                      --l_max 5 \
                      --radial_func_mode ks \
                      --radial_func_max 10 \
                      --channels C,N,O,S,H,SASA,charge \
                      --rst_normalization square \
                      --parallelism 1
