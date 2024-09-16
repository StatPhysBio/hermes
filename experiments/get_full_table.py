

import os
import pandas as pd

METADATA_COLUMNS = ['dataset', 'use MT structures', 'measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']

HERMES_MODELS = 'hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st hermes_py_000_untrained_small_ft_cdna117k_ddg_st hermes_py_050_untrained_small_ft_cdna117k_ddg_st'.split()

HERMES_MODELS_SUP_SKEMPI = 'hermes_bp_000_ft_skempi_hard hermes_bp_050_ft_skempi_hard hermes_py_000_ft_skempi_hard hermes_py_050_ft_skempi_hard hermes_bp_000_ft_skempi_medium hermes_bp_050_ft_skempi_medium hermes_py_000_ft_skempi_medium hermes_py_050_ft_skempi_medium hermes_bp_000_ft_skempi_easy hermes_bp_050_ft_skempi_easy hermes_py_000_ft_skempi_easy hermes_py_050_ft_skempi_easy'.split()

PROTEINMPNN_MODELS = ['proteinmpnn_v_48_002', 'proteinmpnn_v_48_020', 'proteinmpnn_v_48_030']

ESM_MODELS = ['esm_1v_wt_marginals']


if __name__ == '__main__':

    # make the tables
    os.system('cd Protein_G && python make_results_table.py && cd ..')
    os.system('cd VAMP && python make_results_table.py && cd ..')
    os.system('cd ProTherm && python make_results_table.py && cd ../..')
    os.system('cd S669 && python make_results_table.py && cd ..')
    os.system('cd Ssym && python make_results_table.py && cd ..')
    os.system('cd t4_lysozyme && python make_results_table.py && cd ..')
    os.system('cd skempi && python make_results_table.py && cd ..')
    os.system('cd atlas && python make_results_table.py && cd ..')
    os.system('cd T2837 && python make_results_table.py && cd ..')
    os.system('cd T2837_esmfold && python make_results_table.py && cd ..')
    os.system('cd T2837_tp && python make_results_table.py && cd ..')
    os.system('cd T2837_tp_esmfold && python make_results_table.py && cd ..')

    # combine the tables
    table_files = ['Protein_G/protein_g_ddg_experimental-results_table.csv',
                    'VAMP/vamp_ddg_experimental-results_table.csv',
                    'ProTherm/protherm_targets_ddg_experimental-results_table.csv',
                    'S669/s669_ddg_experimental-results_table.csv',
                    'Ssym/Ssym_dir/ssym_dir_ddg_experimental-results_table.csv',
                    'Ssym/Ssym_inv/ssym_inv_ddg_experimental-results_table.csv',
                    't4_lysozyme/T4_mutant_ddG_standardized-results_table.csv',
                    'skempi/skempi_v2_cleaned_NO_1KBH-results_table.csv',
                    'atlas/ATLAS_cleaned-results_table.csv',
                    'T2837/T2837_ddg_experimental-results_table.csv',
                    'T2837_esmfold/T2837_esmfold_ddg_experimental-results_table.csv',
                    'T2837_tp/T2837_tp_ddg_experimental-results_table.csv',
                    'T2837_tp_esmfold/T2837_tp_esmfold_ddg_experimental-results_table.csv']

    dfs = [pd.read_csv(f) for f in table_files]

    df = pd.concat(dfs, ignore_index=True)

    df.to_csv('full_results_table.csv', index=False)


        






