

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from zernikegrams.structural_info import get_structural_info_fn


def get_residues_info(pdb_dir: str) -> pd.DataFrame:

    new_df = pd.DataFrame()

    ## columns: pdb,chain,resname,resnum,insertion_code

    pdb_files = os.listdir(pdb_dir)

    for pdb_file in tqdm(pdb_files):
        pdb_file = os.path.join(pdb_dir, pdb_file)
        if not os.path.exists(pdb_file):
            print(f'{pdb_file} does not exist')
            continue
        pdb = pdb_file.split('/')[-1].split('.')[0]

        # get structural information
        try:
            structural_info = get_structural_info_fn(pdb_file, parser='pyrosetta', hydrogens=True, SASA=True, DSSP=True, angles=False)
        except Exception as e:
            print(f'{pdb_file} failed')
            print(e)
            continue

        res_ids = structural_info['res_ids'][0]
        SASAs = structural_info['SASAs'][0]

        chains = np.array([res_id[2].decode('utf-8') for res_id in res_ids])
        resnames = np.array([res_id[0].decode('utf-8') for res_id in res_ids])
        resnums = np.array([int(res_id[3].decode('utf-8')) for res_id in res_ids])
        icodes = np.array([res_id[4].decode('utf-8') for res_id in res_ids])

        chain_resname_resnum_icode_unique_list = sorted(list(set([(chain, resname, resnum, icode) for chain, resname, resnum, icode in zip(chains, resnames, resnums, icodes)])))

        ss_s = []
        sasa_s = []
        pdbs_aa = []
        chains_aa = []
        resnames_aa = []
        resnums_aa = []
        icodes_aa = []
        for chain, resname, resnum, icode in chain_resname_resnum_icode_unique_list:

            # identify indices of chain and residue number, each index identifies an atom
            indices = np.where((resnums == resnum) & (chains == chain) & (icode == icodes))[0]

            ss = res_ids[indices][0][-1].decode('utf-8')
            sasa = np.mean(SASAs[indices])

            ss_s.append(ss)
            sasa_s.append(sasa)
            pdbs_aa.append(pdb)
            chains_aa.append(chain)
            resnames_aa.append(resname)
            resnums_aa.append(resnum)
            icodes_aa.append(icode)
        
        curr_df = pd.DataFrame()
        curr_df['pdb'] = pdbs_aa
        curr_df['chain'] = chains_aa
        curr_df['resname'] = resnames_aa
        curr_df['resnum'] = resnums_aa
        curr_df['insertion_code'] = icodes_aa
        curr_df['sec_struc'] = ss_s
        curr_df['sasa'] = sasa_s

        new_df = pd.concat([new_df, curr_df])

    return new_df