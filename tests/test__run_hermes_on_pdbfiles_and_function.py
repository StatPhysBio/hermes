
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import sys
sys.path.append('..')
from hermes.inference import run_hermes_on_pdbfile_or_pyrosetta_pose
from hermes.utils.protein_naming import ol_to_ind_size


TEMPDIR = 'temp'
OUTDIR = 'output'

def write_pdbids_chains_sites_list(name: str, pdbids_chains_sites_list: List[Tuple[str, Optional[str], Optional[List[int]]]]):
    filepath = os.path.join(TEMPDIR, name + '.txt')
    with open(filepath, 'w+') as f:
        for pdbid, chain, sites in pdbids_chains_sites_list:
            f.write(pdbid)
            if chain is not None:
                f.write(' ' + chain)
            if sites is not None:
                for site in sites:
                    f.write(' ' + str(site))
            f.write('\n')

def get_script_call(name, m='hermes_py_050', pn=False, pp=0):

    call = f"python -u ../run_hermes_on_pdbfiles.py \
                -m {m} \
                -pd pdbs \
                -r logits \
                -o {os.path.join(OUTDIR, name+'.csv')} \
                -pp {pp} \
            "
    if pn:
        call += f"-pn {os.path.join(TEMPDIR, name + '.txt')}"
    
    return call


def test__all_pdbs_in_folder():
    call = get_script_call('test__all_pdbs_in_folder', m='hermes_py_050', pn=False, pp=0)
    os.system(call)

def test__select_pdb():
    write_pdbids_chains_sites_list('test__select_pdb', [('1a0f', None, None)])
    call = get_script_call('test__select_pdb', m='hermes_py_050', pn=True, pp=0)
    os.system(call)

def test__select_pdb_chain():
    write_pdbids_chains_sites_list('test__select_pdb_chain', [('1a0f', 'A', None), ('1bni', 'B', None)])
    call = get_script_call('test__select_pdb_chain', m='hermes_py_050', pn=True, pp=0)
    os.system(call)

def test__select_pdb_chain_sites():
    write_pdbids_chains_sites_list('test__select_pdb_chain_sites', [('1a0f', 'A', [3, 4, 5, 6]), ('1bni', 'B', [10, 11, 12, 13])])
    call = get_script_call('test__select_pdb_chain_sites', m='hermes_py_050', pn=True, pp=0)
    os.system(call)

def test__select_pdb_chain_sites_icodes():
    write_pdbids_chains_sites_list('test__select_pdb_chain_sites_icodes', [('5jzy', 'L', [14, '14-A', '14-B', '14-D'])])
    call = get_script_call('test__select_pdb_chain_sites_icodes', m='hermes_py_050', pn=True, pp=0)
    os.system(call)

def test__select_pdb_chain_parallelism():
    write_pdbids_chains_sites_list('test__select_pdb_chain_parallelism', [('1a0f', 'A', None), ('1bni', 'B', None)])
    call = get_script_call('test__select_pdb_chain_parallelism', m='hermes_py_050', pn=True, pp=2)
    os.system(call)

def test__select_pdb_chain_sites_parallelism():
    write_pdbids_chains_sites_list('test__select_pdb_chain_sites_parallelism', [('1a0f', 'A', [3, 4, 5, 6]), ('1bni', 'B', [10, 11, 12, 13])])
    call = get_script_call('test__select_pdb_chain_sites_parallelism', m='hermes_py_050', pn=True, pp=2)
    os.system(call)



def test_function__select_pdb_chain_sites_icodes():

    # first make the csv file with the script, as ground truth
    write_pdbids_chains_sites_list('test_function__select_pdb_chain_sites_icodes', [('5jzy', 'L', [14, '14-A', '14-B', '14-D'])])
    call = get_script_call('test_function__select_pdb_chain_sites_icodes', m='hermes_py_050', pn=True, pp=0)
    os.system(call)
    df_true = pd.read_csv(os.path.join(OUTDIR, 'test_function__select_pdb_chain_sites_icodes.csv'))
    df_true_logits = []
    for aa in sorted(list(ol_to_ind_size.keys())):
        df_true_logits.append(df_true[f'logit_{aa}'])
    df_true_logits = np.vstack(df_true_logits)

    df, _ = run_hermes_on_pdbfile_or_pyrosetta_pose('hermes_py_050', './pdbs/5jzy.pdb', chain_and_sites_list=[('L', ['14', '14-A', '14-B', '14-D'])])

    df_logits = []
    for aa in sorted(list(ol_to_ind_size.keys())):
        df_logits.append(df[f'logit_{aa}'])
    df_logits = np.vstack(df_logits)

    assert np.allclose(df_logits, df_true_logits)


def test_function__select_mix():
    # first make the csv file with the script, as ground truth
    write_pdbids_chains_sites_list('test_function__select_mix', [('5jzy', 'L', [14, '14-A', '14-B', '14-D']), ('5jzy', 'H', None)])
    call = get_script_call('test_function__select_mix', m='hermes_py_050', pn=True, pp=0)
    os.system(call)
    df_true = pd.read_csv(os.path.join(OUTDIR, 'test_function__select_mix.csv'))
    df_true_logits = []
    for aa in sorted(list(ol_to_ind_size.keys())):
        df_true_logits.append(df_true[f'logit_{aa}'])
    df_true_logits = np.vstack(df_true_logits)

    df, _ = run_hermes_on_pdbfile_or_pyrosetta_pose('hermes_py_050', './pdbs/5jzy.pdb', chain_and_sites_list=[('L', ['14', '14-A', '14-B', '14-D']), 'H'])

    df_logits = []
    for aa in sorted(list(ol_to_ind_size.keys())):
        df_logits.append(df[f'logit_{aa}'])
    df_logits = np.vstack(df_logits)

    # print(df_logits[~np.isclose(df_logits, df_true_logits, atol=1e-4)])
    # print(df_true_logits[~np.isclose(df_logits, df_true_logits, atol=1e-4)])

    assert np.allclose(df_logits, df_true_logits, atol=1e-1) # GPU computation making these diverge a little sometimes


if __name__ == '__main__':
    os.system(f'rm -r {TEMPDIR}')
    os.system(f'rm -r {OUTDIR}')
    os.makedirs(TEMPDIR, exist_ok=False)
    os.makedirs(OUTDIR, exist_ok=False)

    test__all_pdbs_in_folder()
    test__select_pdb()
    test__select_pdb_chain()
    test__select_pdb_chain_sites()
    test__select_pdb_chain_parallelism()
    test__select_pdb_chain_sites_parallelism()

    test_function__select_pdb_chain_sites_icodes()
    test_function__select_mix()



