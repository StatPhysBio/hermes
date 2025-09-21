


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import spearmanr

from aa_properties.constants import HYDROPHOBIC_PROPS, STERIC_PROPS, ELECTRONIC_PROPS, PROP_TYPE_TO_PROPS, PROP_TYPES, ALL_PROPS, PROP_TYPE_TO_COLOR

def create_white_to_color_cmap(color_name, cmap_name='custom_cmap'):
    """Creates a colormap from white to the specified color."""
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, ['white', color_name])


# MODEL_TO_PRETTY_NAME_SR = {
#     'hermes_py_000': r'$\rho_{HERMES-fixed_{000}}$',
#     'hermes_py_050': r'$\rho_{HERMES-fixed_{050}}$',
#     'hermes_py_000__relaxed_nrep1_ens1_nowt_side12_bb0': r'$\rho_{HERMES-relaxed_{000}}$',
#     'hermes_py_050__relaxed_nrep1_ens1_nowt_side12_bb0': r'$\rho_{HERMES-relaxed_{050}}$',
#     'hermes_py_000_ft_cdna117k_relaxed_pred': r'$\rho_{HERMES-FTrelaxed_{000}}$',
#     'hermes_py_050_ft_cdna117k_relaxed_pred': r'$\rho_{HERMES-FTrelaxed_{050}}$',
#     'hermes_py_000_ft_cdna117k_ddg_st': r'$\rho_{HERMES-fixed_{000}^{ft}}$',
#     'hermes_py_050_ft_cdna117k_ddg_st': r'$\rho_{HERMES-fixed_{050}^{ft}}$',
#     'blosum62': r'$\rho_{B_{62}}$',
#     'neg_ddg': r'$\rho_{- avg \Delta\Delta G}$',
#     'neg_ddg_binary': r'$\rho_{- avg 1[\Delta\Delta G > 0]}$',
#     'proteinmpnn_v_48_002': r'$\rho_{ProteinMPNN_{002}}$',
#     'proteinmpnn_v_48_030': r'$\rho_{ProteinMPNN_{030}}',
# }

# MODEL_TO_PRETTY_NAME = {
#     'hermes_py_000': r'$HERMES-fixed_{000}$',
#     'hermes_py_050': r'$HERMES-fixed_{050}$',
#     'hermes_py_000__relaxed_nrep1_ens1_nowt_side12_bb0': r'$HERMES-relaxed_{000}$',
#     'hermes_py_050__relaxed_nrep1_ens1_nowt_side12_bb0': r'$HERMES-relaxed_{050}$',
#     'hermes_py_000_ft_cdna117k_relaxed_pred': r'$HERMES-FTrelaxed_{000}$',
#     'hermes_py_050_ft_cdna117k_relaxed_pred': r'$HERMES-FTrelaxed_{050}$',
#     'hermes_py_000_ft_cdna117k_ddg_st': r'$HERMES-fixed_{000}^{ft cdna117k}$',
#     'hermes_py_050_ft_cdna117k_ddg_st': r'$HERMES-fixed_{050}^{ft}$',
#     'blosum62': r'$B_{62}$',
#     'neg_ddg': r'$- avg \Delta\Delta G$',
#     'neg_ddg_binary': r'$- avg 1[\Delta\Delta G > 0]$',
#     'proteinmpnn_v_48_002': r'$ProteinMPNN_{002}$',
#     'proteinmpnn_v_48_030': r'$ProteinMPNN_{030}$',
# }

# MODEL_TO_PRETTY_NAME_MINUS_AVERAGE = {
#     'hermes_py_000': r'$HERMES_{000} - \overline{HERMES_{000}}$',
#     'hermes_py_050': r'$HERMES_{050} - \overline{HERMES_{050}}$',
#     'hermes_py_000__relaxed_nrep1_ens1_nowt_side12_bb0': r'$HERMES-relaxed_{000} - \overline{HERMES-relaxed_{000}}$',
#     'hermes_py_050__relaxed_nrep1_ens1_nowt_side12_bb0': r'$HERMES-relaxed_{050} - \overline{HERMES-relaxed_{050}}$',
#     'hermes_py_000_ft_cdna117k_relaxed_pred': r'$HERMES-FT\_relaxed_{000} - \overline{HERMES-FTrelaxed_{000}}$',
#     'hermes_py_050_ft_cdna117k_relaxed_pred': r'$HERMES-FT\_relaxed_{050} - \overline{HERMES-FTrelaxed_{050}}$',
#     'hermes_py_000_ft_cdna117k_ddg_st': r'$HERMES_{000}^{ft} - \overline{HERMES_{000}^{ft}}$',
#     'hermes_py_050_ft_cdna117k_ddg_st': r'$HERMES_{050}^{ft} - -\overline{HERMES_{050}^{ft}}$',
#     'blosum62': r'$B_{62} - \overline{B_{62}}$',
#     'neg_ddg': r'$- avg \Delta\Delta G - \overline{- avg \Delta\Delta G}$',
#     'neg_ddg_binary': r'$- avg \Delta\Delta G - \overline{- avg 1[\Delta\Delta G > 0]}$',
#     'proteinmpnn_v_48_002': r'$ProteinMPNN_{002} - \overline{ProteinMPNN_{002}}$',
#     'proteinmpnn_v_48_030': r'$ProteinMPNN_{030} - \overline{ProteinMPNN_{030}}$',
# }




MODEL_TO_PRETTY_NAME = {
    'stability_oracle': 'Stability-Oracle + cDNA117k',

    'proteinmpnn_v_48_002': 'ProteinMPNN 0.02',
    'proteinmpnn_v_48_030': 'ProteinMPNN 0.30',

    'hermes_py_000': 'HERMES-fixed 0.00',
    'hermes_py_000_ft_cdna117k_relaxed_pred': 'HERMES-FT_relaxed 0.00',
    'hermes_py_000__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed 0.00',
    'hermes_py_000_ft_cdna117k_ddg_st': 'HERMES-fixed 0.00\n+ cDNA117k',
    'hermes_py_000_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed 0.00\n+ cDNA117k',
    'hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': 'HERMES-FT_relaxed 0.00\n+ cDNA117k',

    'hermes_py_050': 'HERMES-fixed 0.50',
    'hermes_py_050_ft_cdna117k_relaxed_pred': 'HERMES-FT_relaxed 0.50',
    'hermes_py_050__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed 0.50',
    'hermes_py_050_ft_cdna117k_ddg_st': 'HERMES-fixed 0.50\n+ cDNA117k',
    'hermes_py_050_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed 0.50\n+ cDNA117k',
    'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': 'HERMES-FT_relaxed 0.50\n+ cDNA117k',

    'blosum62': 'BLOSUM62',
    'neg_ddg': r'$- \Delta\Delta G$',
    'neg_ddg_binary': r'$- 1[\Delta\Delta G > 0]$',


    'hermes_bp_000': 'HERMES-fixed Bp 0.00',
    'hermes_bp_000_ft_cdna117k_relaxed_pred': 'HERMES-FT_relaxed Bp 0.00',
    'hermes_bp_000__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed Bp 0.00',
    'hermes_bp_000_ft_cdna117k_ddg_st': 'HERMES-fixed Bp 0.00\n+ cDNA117k',
    'hermes_bp_000_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed Bp 0.00\n+ cDNA117k',
    'hermes_bp_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': 'HERMES-FT_relaxed Bp 0.00\n+ cDNA117k',

    'hermes_bp_050': 'HERMES-fixed 0.50',
    'hermes_bp_050_ft_cdna117k_relaxed_pred': 'HERMES-FT_relaxed Bp 0.50',
    'hermes_bp_050__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed Bp 0.50',
    'hermes_bp_050_ft_cdna117k_ddg_st': 'HERMES-fixed Bp 0.50\n+ cDNA117k',
    'hermes_bp_050_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': 'HERMES-relaxed Bp 0.50\n+ cDNA117k',
    'hermes_bp_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': 'HERMES-FT_relaxed Bp 0.50\n+ cDNA117k',
}


BLOSUM62 = {
    'C': {'C': 9, 'S': -1, 'T': -1, 'A': 0, 'G': -3, 'P': -3, 'D': -3, 'E': -4, 'Q': -3, 'N': -3, 'H': -3, 'R': -3, 'K': -3, 'M': -1, 'I': -1, 'L': -1, 'V': -1, 'W': -2, 'Y': -2, 'F': -2},
    'S': {'C': -1, 'S': 4, 'T': 1, 'A': 1, 'G': 0, 'P': -1, 'D': 0, 'E': 0, 'Q': 0, 'N': 1, 'H': -1, 'R': -1, 'K': 0, 'M': -1, 'I': -2, 'L': -2, 'V': -2, 'W': -3, 'Y': -2, 'F': -2},
    'T': {'C': -1, 'S': 1, 'T': 5, 'A': 0, 'G': -2, 'P': -1, 'D': -1, 'E': -1, 'Q': -1, 'N': 0, 'H': -2, 'R': -1, 'K': -1, 'M': -1, 'I': -1, 'L': -1, 'V': 0, 'W': -2, 'Y': -2, 'F': -2},
    'A': {'C': 0, 'S': 1, 'T': 0, 'A': 4, 'G': 0, 'P': -1, 'D': -2, 'E': -1, 'Q': -1, 'N': -2, 'H': -2, 'R': -1, 'K': -1, 'M': -1, 'I': -1, 'L': -1, 'V': 0, 'W': -3, 'Y': -2, 'F': -2},
    'G': {'C': -3, 'S': 0, 'T': -2, 'A': 0, 'G': 6, 'P': -2, 'D': -1, 'E': -2, 'Q': -2, 'N': 0, 'H': -2, 'R': -2, 'K': -2, 'M': -3, 'I': -4, 'L': -4, 'V': -3, 'W': -2, 'Y': -3, 'F': -3},
    'P': {'C': -3, 'S': -1, 'T': -1, 'A': -1, 'G': -2, 'P': 7, 'D': -1, 'E': -1, 'Q': -1, 'N': -1, 'H': -2, 'R': -2, 'K': -1, 'M': -2, 'I': -3, 'L': -3, 'V': -2, 'W': -4, 'Y': -3, 'F': -4},
    'D': {'C': -3, 'S': 0, 'T': -1, 'A': -2, 'G': -1, 'P': -1, 'D': 6, 'E': 2, 'Q': 0, 'N': 1, 'H': -1, 'R': -2, 'K': -1, 'M': -3, 'I': -3, 'L': -4, 'V': -3, 'W': -4, 'Y': -3, 'F': -3},
    'E': {'C': -4, 'S': 0, 'T': -1, 'A': -1, 'G': -2, 'P': -1, 'D': 2, 'E': 5, 'Q': 2, 'N': 0, 'H': 0, 'R': 0, 'K': 1, 'M': -2, 'I': -3, 'L': -3, 'V': -2, 'W': -3, 'Y': -2, 'F': -3},
    'Q': {'C': -3, 'S': 0, 'T': -1, 'A': -1, 'G': -2, 'P': -1, 'D': 0, 'E': 2, 'Q': 5, 'N': 0, 'H': 0, 'R': 1, 'K': 1, 'M': 0, 'I': -3, 'L': -2, 'V': -2, 'W': -2, 'Y': -1, 'F': -3},
    'N': {'C': -3, 'S': 1, 'T': 0, 'A': -2, 'G': 0, 'P': -2, 'D': 1, 'E': 0, 'Q': 0, 'N': 6, 'H': 1, 'R': 0, 'K': 0, 'M': -2, 'I': -3, 'L': -3, 'V': -3, 'W': -4, 'Y': -2, 'F': -3},
    'H': {'C': -3, 'S': -1, 'T': -2, 'A': -2, 'G': -2, 'P': -2, 'D': -1, 'E': 0, 'Q': 0, 'N': 1, 'H': 8, 'R': 0, 'K': -1, 'M': -2, 'I': -3, 'L': -3, 'V': -3, 'W': -2, 'Y': 2, 'F': -1},
    'R': {'C': -3, 'S': -1, 'T': -1, 'A': -1, 'G': -2, 'P': -2, 'D': -2, 'E': 0, 'Q': 1, 'N': 0, 'H': 0, 'R': 5, 'K': 2, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'W': -3, 'Y': -2, 'F': -3},
    'K': {'C': -3, 'S': 0, 'T': -1, 'A': -1, 'G': -2, 'P': -1, 'D': -1, 'E': 1, 'Q': 1, 'N': 0, 'H': -1, 'R': 2, 'K': 5, 'M': -1, 'I': -3, 'L': -2, 'V': -2, 'W': -3, 'Y': -2, 'F': -3},
    'M': {'C': -1, 'S': -1, 'T': -1, 'A': -1, 'G': -3, 'P': -2, 'D': -3, 'E': -2, 'Q': 0, 'N': -2, 'H': -2, 'R': -1, 'K': -1, 'M': 5, 'I': 1, 'L': 2, 'V': 1, 'W': -1, 'Y': -1, 'F': 0},
    'I': {'C': -1, 'S': -2, 'T': -1, 'A': -1, 'G': -4, 'P': -3, 'D': -3, 'E': -3, 'Q': -3, 'N': -3, 'H': -3, 'R': -3, 'K': -3, 'M': 1, 'I': 4, 'L': 2, 'V': 3, 'W': -3, 'Y': -1, 'F': 0},
    'L': {'C': -1, 'S': -2, 'T': -1, 'A': -1, 'G': -4, 'P': -3, 'D': -4, 'E': -3, 'Q': -2, 'N': -3, 'H': -3, 'R': -2, 'K': -2, 'M': 2, 'I': 2, 'L': 4, 'V': 1, 'W': -2, 'Y': -1, 'F': 0},
    'V': {'C': -1, 'S': -2, 'T': 0, 'A': 0, 'G': -3, 'P': -2, 'D': -3, 'E': -2, 'Q': -2, 'N': -3, 'H': -3, 'R': -3, 'K': -2, 'M': 1, 'I': 3, 'L': 1, 'V': 4, 'W': -3, 'Y': -1, 'F': -1},
    'W': {'C': -2, 'S': -3, 'T': -2, 'A': -3, 'G': -2, 'P': -4, 'D': -4, 'E': -3, 'Q': -2, 'N': -4, 'H': -2, 'R': -3, 'K': -3, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'W': 11, 'Y': 2, 'F': 1},
    'Y': {'C': -2, 'S': -2, 'T': -2, 'A': -2, 'G': -3, 'P': -3, 'D': -3, 'E': -2, 'Q': -1, 'N': -2, 'H': 2, 'R': -2, 'K': -2, 'M': -1, 'I': -1, 'L': -1, 'V': -1, 'W': 2, 'Y': 7, 'F': 3},
    'F': {'C': -2, 'S': -2, 'T': -2, 'A': -2, 'G': -3, 'P': -4, 'D': -3, 'E': -3, 'Q': -3, 'N': -3, 'H': -1, 'R': -3, 'K': -3, 'M': 0, 'I': 0, 'L': 0, 'V': -1, 'W': 1, 'Y': 3, 'F': 6}
}

color_list = plt.get_cmap('tab20').colors
blue = color_list[0]
blue_light = color_list[1]
orange = color_list[2]
orange_light = color_list[3]
green = color_list[4]
green_light = color_list[5]
red = color_list[6]
red_light = color_list[7]
purple = color_list[8]
purple_light = color_list[9]
brown = color_list[10]
brown_light = color_list[11]
pink = color_list[12]
pink_light = color_list[13]
gray = color_list[14]
gray_light = color_list[15]
olive = color_list[16]
olive_light = color_list[17]
cyan = color_list[18]
cyan_light = color_list[19]

MODEL_TO_CMAP = {
    'stability_oracle': create_white_to_color_cmap(brown),

    'proteinmpnn_v_48_002': create_white_to_color_cmap(gray),
    'proteinmpnn_v_48_030': create_white_to_color_cmap(gray),

    'hermes_py_000': create_white_to_color_cmap(blue),
    'hermes_py_000_ft_cdna117k_relaxed_pred': create_white_to_color_cmap(red),
    'hermes_py_000__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(pink),
    'hermes_py_000_ft_cdna117k_ddg_st': create_white_to_color_cmap(purple),
    'hermes_py_000_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(olive),
    'hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': create_white_to_color_cmap(green),

    'hermes_py_050': create_white_to_color_cmap(blue),
    'hermes_py_050_ft_cdna117k_relaxed_pred': create_white_to_color_cmap(red),
    'hermes_py_050__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(pink),
    'hermes_py_050_ft_cdna117k_ddg_st': create_white_to_color_cmap(purple),
    'hermes_py_050_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(olive),
    'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': create_white_to_color_cmap(green),

    'blosum62': create_white_to_color_cmap(gray),
    'neg_ddg': create_white_to_color_cmap(orange),
    'neg_ddg_binary': create_white_to_color_cmap(orange),


    'hermes_bp_000': create_white_to_color_cmap(blue),
    'hermes_bp_000_ft_cdna117k_relaxed_pred': create_white_to_color_cmap(red),
    'hermes_bp_000__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(pink),
    'hermes_bp_000_ft_cdna117k_ddg_st': create_white_to_color_cmap(purple),
    'hermes_bp_000_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(olive),
    'hermes_bp_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': create_white_to_color_cmap(green),

    'hermes_bp_050': create_white_to_color_cmap(blue),
    'hermes_bp_050_ft_cdna117k_relaxed_pred': create_white_to_color_cmap(red),
    'hermes_bp_050__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(pink),
    'hermes_bp_050_ft_cdna117k_ddg_st': create_white_to_color_cmap(purple),
    'hermes_bp_050_ft_cdna117k_ddg_st__relaxed_nrep1_ens1_nowt_side12_bb0': create_white_to_color_cmap(olive),
    'hermes_bp_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st': create_white_to_color_cmap(green),
}

AMINOACIDS = 'GPCAVILMFYWSTNQRHKDE'

special_cases = gray
hydrophobic = orange
polar = purple
positive = red
negative = blue

AA_TYPE_TO_COLOR = {
    'special cases': special_cases,
    'hydrophobic': hydrophobic,
    'polar uncharged': polar,
    'positively charged': positive,
    'negatively charged': negative
}

AA_TO_AA_TYPE = {
    # special cases
    'G': 'special cases',
    'P': 'special cases',
    'C': 'special cases',
    # hydrophobic
    'A': 'hydrophobic',
    'V': 'hydrophobic',
    'I': 'hydrophobic',
    'L': 'hydrophobic',
    'M': 'hydrophobic',
    'F': 'hydrophobic',
    'Y': 'hydrophobic',
    'W': 'hydrophobic',
    # polar, not charged
    'S': 'polar unch.',
    'T': 'polar unch.',
    'N': 'polar unch.',
    'Q': 'polar unch.',
    # positively charged
    'R': 'positive ch.',
    'H': 'positive ch.',
    'K': 'positive ch.',
    # negatively charged
    'D': 'negative ch.',
    'E': 'negative ch.',
}

AA_TO_COLOR = {
    # special cases
    'G': special_cases,
    'P': special_cases,
    'C': special_cases,
    # hydrophobic
    'A': hydrophobic,
    'V': hydrophobic,
    'I': hydrophobic,
    'L': hydrophobic,
    'M': hydrophobic,
    'F': hydrophobic,
    'Y': hydrophobic,
    'W': hydrophobic,
    # polar, not charged
    'S': polar,
    'T': polar,
    'N': polar,
    'Q': polar,
    # positively charged
    'R': positive,
    'H': positive,
    'K': positive,
    # negatively charged
    'D': negative,
    'E': negative,
}

AA_TO_SIZE_BUCKET = {
    'G': 'small',
    'A': 'small',
    'S': 'small',
    'C': 'small',
    'T': 'small',
    'P': 'small',
    'D': 'small',

    'N': 'medium',
    'V': 'medium',
    'E': 'medium',
    'Q': 'medium',
    'L': 'medium',
    'I': 'medium',

    'M': 'large',
    'H': 'large',
    'K': 'large',
    'F': 'large',
    'R': 'large',
    'Y': 'large',
    'W': 'large'
}

def reorder_matrix(df_matrix, aminoacids: str = AMINOACIDS, set_diagonal_to_zero: bool = True): # diagonal should be zero, sometimes previous processing made it be nan
    df_matrix =  df_matrix.loc[list(aminoacids), list(aminoacids)]

    if set_diagonal_to_zero:
        for i in range(20):
            df_matrix.iloc[i, i] = 0.0
    
    return df_matrix


def bucket_matrix_by_categories(df_matrix, aa_to_category, exclude_diagonal: bool = False):

    temp_categories = aa_to_category.values()
    categories = []
    for category in temp_categories:
        if category not in categories:
            categories.append(category)

    category_to_aas = {}
    for aa in aa_to_category:
        category = aa_to_category[aa]
        if category not in category_to_aas:
            category_to_aas[category] = []
        category_to_aas[category].append(aa)
    
    df_matrix_copy = df_matrix.copy()

    if exclude_diagonal:
        for i in range(20):
            df_matrix_copy.iloc[i, i] = np.nan
    
    # Create a new DataFrame to store the bucketed values

    df_matrix_bucketed = pd.DataFrame(index=categories, columns=categories)

    for i, category1 in enumerate(categories):
        for j, category2 in enumerate(categories):

            # Get the indices of the amino acids in the categories
            indices1 = [df_matrix_copy.index.get_loc(aa) for aa in category_to_aas[category1]]
            indices2 = [df_matrix_copy.index.get_loc(aa) for aa in category_to_aas[category2]]

            # Calculate the mean value for the bucketed category
            values = df_matrix_copy.iloc[indices1, :].iloc[:, indices2].values.flatten()
            mean_value = np.nanmean(values)

            # Set the mean value in the new DataFrame
            df_matrix_bucketed.loc[category1, category2] = mean_value

    return df_matrix_bucketed


def robust_spearmanr(df_matrix1, df_matrix2, return_points=False):
    matrix1_flat = df_matrix1.values.flatten()
    matrix2_flat = df_matrix2.values.flatten()

    # they differ sometimes because two properties have a NaN
    mask1 = ~np.isnan(matrix1_flat)
    mask2 = ~np.isnan(matrix2_flat)
    mask = np.logical_and(mask1, mask2)

    matrix1_flat = matrix1_flat[mask]
    matrix2_flat = matrix2_flat[mask]

    return_value = spearmanr(matrix1_flat, matrix2_flat)
    
    if return_points:
        return_value = (return_value, (matrix1_flat, matrix2_flat))

    return return_value

def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Adjusts a colormap to shift its center, useful when data ranges from negative to positive values.

    Parameters:
        cmap (Colormap): Matplotlib colormap to modify.
        start (float): Lower bound of the colormap range (default: 0.0).
        midpoint (float): New center for the colormap (default: 0.5).
        stop (float): Upper bound of the colormap range (default: 1.0).
        name (str): Name of the new colormap.

    Returns:
        Colormap: Adjusted colormap.

    
    Reference
    ---------
    shiftedColorMap function from https://gist.github.com/phobson/7916777, 
    authored by phobson.
    """
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    
    reg_index = np.linspace(start, stop, 257)
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    
    new_cmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.unregister(name)
    matplotlib.colormaps.register(cmap=new_cmap)
    return new_cmap


def spearman_heatmap(df_heatmap, fontsize, figsize, models_to_consider=None, title=None):

    import seaborn as sns

    if models_to_consider is not None:
        # only keep columns and index in models_to_consider
        df_heatmap = df_heatmap.loc[models_to_consider, models_to_consider]

    ticklabels = [df_heatmap.columns[0]] + [MODEL_TO_PRETTY_NAME[model] for model in df_heatmap.columns[1:]]

    # exclude the opper triangle
    for i in range(df_heatmap.shape[0]):
        for j in range(i+1, df_heatmap.shape[1]):
            df_heatmap.iloc[i, j] = np.nan

    # make heatmap of this matrix, set the cmap range between 0 and 1, put title on colorbar, set fontsize of numbers in heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df_heatmap, cmap='hot', vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": fontsize})
    plt.xticks(np.arange(len(ticklabels))+0.5, ticklabels, rotation=70, fontsize=fontsize, ha='right')
    plt.yticks(np.arange(len(ticklabels))+0.5, ticklabels, rotation=0, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize+2)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    prop_to_df = {}
    for prop in ALL_PROPS:
        prop_to_df[prop] = reorder_matrix(pd.read_csv(f'aa_properties/matrices/{prop}.csv', index_col=0))

    handles = [(Patch(color=PROP_TYPE_TO_COLOR[prop_type], alpha=0.6, label=prop_type)) for prop_type in PROP_TYPES]
    plt.figure(figsize=(5, 5))
    plt.legend(handles=handles, loc='center', fontsize=14)
    plt.axis('off')
    plt.savefig(f'aminoacid_prop_types__legend.png')
    plt.savefig(f'aminoacid_prop_types__legend.pdf')
    plt.close()

    handles = [(Patch(color=AA_TYPE_TO_COLOR[aa_type], label=aa_type)) for aa_type in AA_TYPE_TO_COLOR]
    plt.figure(figsize=(5, 5))
    plt.legend(handles=handles, loc='center', fontsize=14)
    plt.axis('off')
    plt.savefig(f'aminoacid_family_colors__legend.png')
    plt.savefig(f'aminoacid_family_colors__legend.pdf')
    plt.close()

    aa_colors = plt.get_cmap('tab20').colors
    handles = [(Line2D([0], [0], marker='o', ls='', color=aa_colors[i], alpha=0.75, label=AMINOACIDS[i])) for i in range(20)]
    plt.figure(figsize=(4, 7))
    plt.legend(handles=handles, loc='center', fontsize=14)
    plt.axis('off')
    plt.savefig(f'aminoacid_individual_colors__legend.png')
    plt.savefig(f'aminoacid_individual_colors__legend.pdf')
    plt.close()


    fontsize = 17


    model_1 = 'neg_ddg' # keep this fixed!!!
    model_2_list = ['hermes_py_050', 'hermes_py_050_ft_cdna117k_relaxed_pred', 'proteinmpnn_v_48_030', 'hermes_py_050_ft_cdna117k_ddg_st', 'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st']

    for subdir in ['all', 'surface', 'core']:

        for model_2 in model_2_list:

            matrix_1 = reorder_matrix(pd.read_csv(f'./cdna117k/{subdir}/matrices/{model_1}.csv', index_col=0))
            matrix_2 = reorder_matrix(pd.read_csv(f'./T2837_all_sites/{subdir}/matrices/{model_2}.csv', index_col=0))

            colors_wt = np.full_like(matrix_1.values, np.nan) # rows
            colors_mt = np.full_like(matrix_1.values, np.nan) # cols
            for i, aa in enumerate(matrix_1.index):
                for j in range(20):
                    colors_wt[i, j] = i
                    colors_mt[j, i] = i
            
            matrix_1_flat = matrix_1.values.flatten()
            matrix_2_flat = matrix_2.values.flatten()
            colors_wt_flat = colors_wt.flatten()
            colors_mt_flat = colors_mt.flatten()

            mask = ~np.isnan(matrix_1_flat) & ~np.isnan(matrix_2_flat) & ~np.isnan(colors_wt_flat) & ~np.isnan(colors_mt_flat)
            matrix_1_flat = matrix_1_flat[mask]
            matrix_2_flat = matrix_2_flat[mask]
            colors_wt_flat = colors_wt_flat[mask]
            colors_mt_flat = colors_mt_flat[mask]
            colors_wt_flat = [aa_colors[int(idx)] for idx in colors_wt_flat]
            colors_mt_flat = [aa_colors[int(idx)] for idx in colors_mt_flat]

            ncols = 2
            nrows = 1
            colsize = 4.5
            rowsize = 4.5
            fig, axs = plt.subplots(figsize=(colsize*ncols, rowsize*nrows), ncols=ncols, nrows=nrows, sharex=False, sharey=False)

            ax = axs[0] # wt
            ax.scatter(matrix_1_flat, matrix_2_flat, c=colors_wt_flat, alpha=0.75)
            ax.set_xlabel(MODEL_TO_PRETTY_NAME[model_1], fontsize=fontsize-1)
            ax.set_ylabel(MODEL_TO_PRETTY_NAME[model_2], fontsize=fontsize-1)
            ax.set_title('Colored by Wild-Type AA', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize-3)

            ax = axs[1] # wt
            ax.scatter(matrix_1_flat, matrix_2_flat, c=colors_mt_flat, alpha=0.75)
            ax.set_xlabel(MODEL_TO_PRETTY_NAME[model_1], fontsize=fontsize-1)
            ax.set_ylabel(MODEL_TO_PRETTY_NAME[model_2], fontsize=fontsize-1)
            ax.set_title('Colored by Mutant AA', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize-3)

            plt.tight_layout()
            plt.savefig(f'./T2837_all_sites/{subdir}/plots/matrix_comparison_{model_1}__vs__{model_2}.png')
            plt.savefig(f'./T2837_all_sites/{subdir}/plots/matrix_comparison_{model_1}__vs__{model_2}.pdf')
            plt.close()

    # exit(0)


    matrix_type_to_model_pairs = {
        'skempi': [],
        'cdna117k': [],
        'T2837_all_sites': [('hermes_py_050', 'proteinmpnn_v_48_030'), ('hermes_py_050', 'hermes_py_050_ft_cdna117k_relaxed_pred'), ('hermes_py_050_ft_cdna117k_ddg_st', 'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st')]
    }

    matrix_type_to_correlations = {}
    matrix_type_to_models = {}

    for matrix_type in ['skempi/all_interfaces', 'skempi/from_data', 'T2837_all_sites/core', 'T2837_all_sites/surface', 'T2837_all_sites/all', 'cdna117k/core', 'cdna117k/surface', 'cdna117k/all']: #, 'cdna117k/in_between', 'T2837_all_sites/in_between']:

        print(matrix_type)

        matrices_dir = f'./{matrix_type}/matrices/'
        plotsdir = f'./{matrix_type}/plots'
        os.makedirs(plotsdir, exist_ok=True)
        
        models = [filepath[:-4] for filepath in  os.listdir(matrices_dir)]
        matrix_type_to_models[matrix_type] = models

        model_to_df = {model: reorder_matrix(pd.read_csv(f'./{matrices_dir}/{model}.csv', index_col=0)) for model in models}

        correlations = {model: {prop: robust_spearmanr(model_to_df[model], prop_to_df[prop]) for prop in ALL_PROPS} for model in model_to_df}
        matrix_type_to_correlations[matrix_type] = correlations

        # # compute spearman correlations between matrices
        # model_correlations_df = pd.DataFrame(np.full((len(models), len(models)), np.nan), index=models, columns=models)
        # for i, model1 in enumerate(models):
        #     for model2 in models[i+1:]:
        #         model_correlations_df.loc[model1, model2] = robust_spearmanr(model_to_df[model1], model_to_df[model2])
        
        # model_correlations_df.to_csv(f'./{plotsdir}/correlations.csv')


        for model_1, model_2 in matrix_type_to_model_pairs[matrix_type.split('/')[0]]:

            matrix_1 = model_to_df[model_1]
            matrix_2 = model_to_df[model_2]

            colors_wt = np.full_like(matrix_1.values, np.nan) # rows
            colors_mt = np.full_like(matrix_1.values, np.nan) # cols
            for i, aa in enumerate(matrix_1.index):
                for j in range(20):
                    colors_wt[i, j] = i
                    colors_mt[j, i] = i
            
            matrix_1_flat = matrix_1.values.flatten()
            matrix_2_flat = matrix_2.values.flatten()
            colors_wt_flat = colors_wt.flatten()
            colors_mt_flat = colors_mt.flatten()

            mask = ~np.isnan(matrix_1_flat) & ~np.isnan(matrix_2_flat) & ~np.isnan(colors_wt_flat) & ~np.isnan(colors_mt_flat)
            matrix_1_flat = matrix_1_flat[mask]
            matrix_2_flat = matrix_2_flat[mask]
            colors_wt_flat = colors_wt_flat[mask]
            colors_mt_flat = colors_mt_flat[mask]
            colors_wt_flat = [aa_colors[int(idx)] for idx in colors_wt_flat]
            colors_mt_flat = [aa_colors[int(idx)] for idx in colors_mt_flat]

            ncols = 2
            nrows = 1
            colsize = 4.5
            rowsize = 4.5
            fig, axs = plt.subplots(figsize=(colsize*ncols, rowsize*nrows), ncols=ncols, nrows=nrows, sharex=False, sharey=False)

            ax = axs[0] # wt
            ax.scatter(matrix_1_flat, matrix_2_flat, c=colors_wt_flat, alpha=0.75)
            ax.set_xlabel(MODEL_TO_PRETTY_NAME[model_1], fontsize=fontsize-1)
            ax.set_ylabel(MODEL_TO_PRETTY_NAME[model_2], fontsize=fontsize-1)
            ax.set_title('Colored by Wild-Type AA', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize-3)

            ax = axs[1] # wt
            ax.scatter(matrix_1_flat, matrix_2_flat, c=colors_mt_flat, alpha=0.75)
            ax.set_xlabel(MODEL_TO_PRETTY_NAME[model_1], fontsize=fontsize-1)
            ax.set_ylabel(MODEL_TO_PRETTY_NAME[model_2], fontsize=fontsize-1)
            ax.set_title('Colored by Mutant AA', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize-3)

            plt.tight_layout()
            plt.savefig(f'{plotsdir}/matrix_comparison_{model_1}__vs__{model_2}.png')
            plt.savefig(f'{plotsdir}/matrix_comparison_{model_1}__vs__{model_2}.pdf')
            plt.close()


        for model in models:

            print('\t' + model)

            # ## matrix as heatmap, centered at zero

            # matrix = model_to_df[model].values
            # # manually put the zeros along the diagonal here, because zero is meaningful here, and we need them to make the heatmap
            # for i in range(20):
            #     matrix[i, i] = 0.0

            # mmin = np.nanmin(matrix)
            # mmax = np.nanmax(matrix)

            # midpoint = (1 - ((mmax + 1e-9) / (mmax + np.abs(mmin))))
            # print(mmax, mmin, midpoint)

            # fig = plt.figure(figsize=(6, 6))
            # data = plt.imshow(matrix, cmap=shifted_color_map(matplotlib.colormaps['coolwarm'], midpoint = midpoint))
            # plt.title(MODEL_TO_PRETTY_NAME[model], fontsize=fontsize+2)
            # plt.xticks(np.arange(20), list(AMINOACIDS), fontsize=fontsize-2)
            # plt.yticks(np.arange(20), list(AMINOACIDS), fontsize=fontsize-2)
            # plt.xlabel('mutant', fontsize=fontsize)
            # plt.ylabel('wildtype', fontsize=fontsize)
            # colorbar = fig.colorbar(data)
            # colorbar.ax.tick_params(labelsize=fontsize)
            # # colorbar.set_label(MODEL_TO_PRETTY_NAME_MINUS_AVERAGE[model], fontsize=fontsize+10)
            # plt.tight_layout()
            # plt.savefig(f'{outdir}/average_matrix_{model}.png')
            # plt.savefig(f'{outdir}/average_matrix_{model}.pdf')
            # plt.close()


            ## matrix as heatmap - centered at the average value
            matrix = model_to_df[model].values

            # matrix -= np.nanmean(matrix)
            # mmin = np.nanmin(matrix)
            # mmax = np.nanmax(matrix)
            # vmin = min(mmin, -mmax)
            # vmax = max(mmax, -mmin)

            ## get the vmin and vmax for all subsets (core, surface, and all)
            ## stupid workaround but whattayagonnado?

            # means, mins, maxs = [], [], []
            # for subdir in ['core', 'surface', 'all']:
            #     df = reorder_matrix(pd.read_csv(f'./{matrix_type.split("/")[0]}/{subdir}/matrices/{model}.csv', index_col=0))
            #     means.append(np.nanmean(df.values))
            # vmean = np.mean(means)
            # for subdir in ['core', 'surface', 'all']:
            #     df = reorder_matrix(pd.read_csv(f'./{matrix_type.split("/")[0]}/{subdir}/matrices/{model}.csv', index_col=0))
            #     mins.append(np.nanmin(df.values) - vmean)
            #     maxs.append(np.nanmax(df.values) - vmean)
            # mmin = np.min(mins)
            # mmax = np.max(maxs)
            # vmin = min(mmin, -mmax)
            # vmax = max(mmax, -mmin)
            # matrix -= vmean
            # cmap = 'coolwarm'

            if 'skempi' not in matrix_type:
                mins, maxs = [], []
                for matrix_type_full_dir in [f'{matrix_type.split("/")[0]}/core', f'{matrix_type.split("/")[0]}/surface', f'{matrix_type.split("/")[0]}/all']:
                    df = reorder_matrix(pd.read_csv(f'./{matrix_type_full_dir}/matrices/{model}.csv', index_col=0))
                    mins.append(np.nanmin(df.values))
                    maxs.append(np.nanmax(df.values))
                if model != 'neg_ddg':
                    if os.path.exists('skempi/all_interfaces/matrices/' + model + '.csv'):
                        df = reorder_matrix(pd.read_csv(f'./skempi/all_interfaces/matrices/{model}.csv', index_col=0))
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
                    elif os.path.exists('skempi/from_data/matrices/' + model + '.csv'):
                        df = reorder_matrix(pd.read_csv(f'./skempi/from_data/matrices/{model}.csv', index_col=0))
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
            else:
                mins, maxs = [], []
                df = reorder_matrix(pd.read_csv(f'{matrix_type}/matrices/{model}.csv', index_col=0))
                mins.append(np.nanmin(df.values))
                maxs.append(np.nanmax(df.values))
                if model != 'neg_ddg':
                    if os.path.exists('T2837_all_sites/all/matrices/' + model + '.csv'):
                        matrix_type_base = 'T2837_all_sites'
                    elif os.path.exists('cdna117k/all/matrices/' + model + '.csv'):
                        matrix_type_base = 'cdna117k'
                    for matrix_type_full_dir in [f'{matrix_type_base}/core', f'{matrix_type_base}/surface', f'{matrix_type_base}/all']:
                        df = reorder_matrix(pd.read_csv(f'./{matrix_type_full_dir}/matrices/{model}.csv', index_col=0))
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))

            mmin = np.min(mins)
            mmax = np.max(maxs)
            vmin = mmin
            vmax = mmax
            # cmap = MODEL_TO_CMAP[model]
            cmap = 'coolwarm'

            fig = plt.figure(figsize=(5.5, 5.5))
            data = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(MODEL_TO_PRETTY_NAME[model], fontsize=fontsize+2)
            plt.xticks(np.arange(20), list(AMINOACIDS), fontsize=fontsize)
            plt.yticks(np.arange(20), list(AMINOACIDS), fontsize=fontsize)
            for aa_i, aa in enumerate(AMINOACIDS):
                plt.setp(plt.gca().get_xticklabels()[aa_i], color=AA_TO_COLOR[aa])
                plt.setp(plt.gca().get_yticklabels()[aa_i], color=AA_TO_COLOR[aa])
            
            plt.xlabel('mutant', fontsize=fontsize+1)
            plt.ylabel('wildtype', fontsize=fontsize+1)
            # colorbar = fig.colorbar(data)
            # colorbar.ax.tick_params(labelsize=fontsize)
            # colorbar.set_label(MODEL_TO_PRETTY_NAME_MINUS_AVERAGE[model], fontsize=fontsize+10)
            # plt.tight_layout()
            plt.savefig(f'{plotsdir}/average_matrix_normalized_{model}.png')
            plt.savefig(f'{plotsdir}/average_matrix_normalized_{model}.pdf')
            plt.close()

            # Create a separate colorbar figure
            fig_cb, ax_cb = plt.subplots(figsize=(2, 5))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
            cb.ax.tick_params(labelsize=fontsize+2)
            plt.tight_layout()
            plt.savefig(f'{plotsdir}/colorbar_{model}.png')
            plt.savefig(f'{plotsdir}/colorbar_{model}.pdf')
            plt.close()


            ## matrix as heatmap, bucketed by AA type
            bucketed_matrix_df = bucket_matrix_by_categories(model_to_df[model], AA_TO_AA_TYPE, exclude_diagonal=True)

            if 'skempi' not in matrix_type:
                mins, maxs = [], []
                for matrix_type_full_dir in [f'{matrix_type.split("/")[0]}/core', f'{matrix_type.split("/")[0]}/surface', f'{matrix_type.split("/")[0]}/all']:
                    df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./{matrix_type_full_dir}/matrices/{model}.csv', index_col=0)), AA_TO_AA_TYPE)
                    mins.append(np.nanmin(df.values))
                    maxs.append(np.nanmax(df.values))
                if model != 'neg_ddg':
                    if os.path.exists('skempi/all_interfaces/matrices/' + model + '.csv'):
                        df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./skempi/all_interfaces/matrices/{model}.csv', index_col=0)), AA_TO_AA_TYPE)
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
                    elif os.path.exists('skempi/from_data/matrices/' + model + '.csv'):
                        df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./skempi/from_data/matrices/{model}.csv', index_col=0)), AA_TO_AA_TYPE)
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
            else:
                mins, maxs = [], []
                df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'{matrix_type}/matrices/{model}.csv', index_col=0)), AA_TO_AA_TYPE)
                mins.append(np.nanmin(df.values))
                maxs.append(np.nanmax(df.values))
                if model != 'neg_ddg':
                    if os.path.exists('T2837_all_sites/all/matrices/' + model + '.csv'):
                        matrix_type_base = 'T2837_all_sites'
                    elif os.path.exists('cdna117k/all/matrices/' + model + '.csv'):
                        matrix_type_base = 'cdna117k'
                    for matrix_type_full_dir in [f'{matrix_type_base}/core', f'{matrix_type_base}/surface', f'{matrix_type_base}/all']:
                        df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./{matrix_type_full_dir}/matrices/{model}.csv', index_col=0)), AA_TO_AA_TYPE)
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
            
            mmin = np.min(mins)
            mmax = np.max(maxs)
            vmin = mmin
            vmax = mmax
            # cmap = MODEL_TO_CMAP[model]
            cmap = 'viridis'

            fig = plt.figure(figsize=(6.2, 6.2))
            categories = bucketed_matrix_df.index
            data = plt.imshow(bucketed_matrix_df.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(MODEL_TO_PRETTY_NAME[model], fontsize=fontsize+2)
            plt.xticks(np.arange(len(categories)), categories, fontsize=fontsize, rotation=70, ha='right')
            plt.yticks(np.arange(len(categories)), categories, fontsize=fontsize)
            
            plt.xlabel('mutant', fontsize=fontsize+2)
            plt.ylabel('wildtype', fontsize=fontsize+2)
            # colorbar = fig.colorbar(data)
            # colorbar.ax.tick_params(labelsize=fontsize)
            # colorbar.set_label(MODEL_TO_PRETTY_NAME_MINUS_AVERAGE[model], fontsize=fontsize+10)
            # plt.tight_layout()
            plt.subplots_adjust(
                left=0.35,
                right=0.97,
                bottom=0.35,
                top=0.90
            )
            plt.savefig(f'{plotsdir}/bucketed_by_type__average_matrix_normalized_{model}.png')
            plt.savefig(f'{plotsdir}/bucketed_by_type__average_matrix_normalized_{model}.pdf')
            plt.close()

            # Create a separate colorbar figure
            fig_cb, ax_cb = plt.subplots(figsize=(2, 5))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
            cb.ax.tick_params(labelsize=fontsize+2)
            plt.tight_layout()
            plt.savefig(f'{plotsdir}/bucketed_by_type__colorbar_{model}.png')
            plt.savefig(f'{plotsdir}/bucketed_by_type__colorbar_{model}.pdf')
            plt.close()


            ## matrix as heatmap, bucketed by size
            bucketed_matrix_df = bucket_matrix_by_categories(model_to_df[model], AA_TO_SIZE_BUCKET, exclude_diagonal=True)

            if 'skempi' not in matrix_type:
                mins, maxs = [], []
                for matrix_type_full_dir in [f'{matrix_type.split("/")[0]}/core', f'{matrix_type.split("/")[0]}/surface', f'{matrix_type.split("/")[0]}/all']:
                    df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./{matrix_type_full_dir}/matrices/{model}.csv', index_col=0)), AA_TO_SIZE_BUCKET)
                    mins.append(np.nanmin(df.values))
                    maxs.append(np.nanmax(df.values))
                if model != 'neg_ddg':
                    if os.path.exists('skempi/all_interfaces/matrices/' + model + '.csv'):
                        df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./skempi/all_interfaces/matrices/{model}.csv', index_col=0)), AA_TO_SIZE_BUCKET)
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
                    elif os.path.exists('skempi/from_data/matrices/' + model + '.csv'):
                        df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./skempi/from_data/matrices/{model}.csv', index_col=0)), AA_TO_SIZE_BUCKET)
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
            else:
                mins, maxs = [], []
                df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'{matrix_type}/matrices/{model}.csv', index_col=0)), AA_TO_SIZE_BUCKET)
                mins.append(np.nanmin(df.values))
                maxs.append(np.nanmax(df.values))
                if model != 'neg_ddg':
                    if os.path.exists('T2837_all_sites/all/matrices/' + model + '.csv'):
                        matrix_type_base = 'T2837_all_sites'
                    elif os.path.exists('cdna117k/all/matrices/' + model + '.csv'):
                        matrix_type_base = 'cdna117k'
                    for matrix_type_full_dir in [f'{matrix_type_base}/core', f'{matrix_type_base}/surface', f'{matrix_type_base}/all']:
                        df = bucket_matrix_by_categories(reorder_matrix(pd.read_csv(f'./{matrix_type_full_dir}/matrices/{model}.csv', index_col=0)), AA_TO_SIZE_BUCKET)
                        mins.append(np.nanmin(df.values))
                        maxs.append(np.nanmax(df.values))
                
            mmin = np.min(mins)
            mmax = np.max(maxs)
            vmin = mmin
            vmax = mmax
            # cmap = MODEL_TO_CMAP[model]
            cmap = 'viridis'


            fig = plt.figure(figsize=(5.7, 5.7))
            categories = bucketed_matrix_df.index
            data = plt.imshow(bucketed_matrix_df.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(MODEL_TO_PRETTY_NAME[model], fontsize=fontsize+2)
            plt.xticks(np.arange(len(categories)), categories, fontsize=fontsize, rotation=70, ha='right')
            plt.yticks(np.arange(len(categories)), categories, fontsize=fontsize)
            
            plt.xlabel('mutant', fontsize=fontsize+2)
            plt.ylabel('wildtype', fontsize=fontsize+2)
            # colorbar = fig.colorbar(data)
            # colorbar.ax.tick_params(labelsize=fontsize)
            # colorbar.set_label(MODEL_TO_PRETTY_NAME_MINUS_AVERAGE[model], fontsize=fontsize+10)
            # plt.tight_layout()
            plt.subplots_adjust(
                left=0.25,
                right=0.97,
                bottom=0.25,
                top=0.90
            )
            plt.savefig(f'{plotsdir}/bucketed_by_size__average_matrix_normalized_{model}.png')
            plt.savefig(f'{plotsdir}/bucketed_by_size__average_matrix_normalized_{model}.pdf')
            plt.close()

            # Create a separate colorbar figure
            fig_cb, ax_cb = plt.subplots(figsize=(2, 5))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
            cb.ax.tick_params(labelsize=fontsize+2)
            plt.tight_layout()
            plt.savefig(f'{plotsdir}/bucketed_by_size__colorbar_{model}.png')
            plt.savefig(f'{plotsdir}/bucketed_by_size__colorbar_{model}.pdf')
            plt.close()



            ## violinplot vs. aminoacid properties
            plt.figure(figsize=(4, 4))
            for i, prop_type in enumerate(PROP_TYPES):
                m_values = [correlations[model][prop][0] for prop in PROP_TYPE_TO_PROPS[prop_type]]
                parts = plt.violinplot([m_values], positions=[i], widths=0.6, showmedians=True, showextrema=False)
                parts['cmedians'].set_color(PROP_TYPE_TO_COLOR[prop_type])
                parts['cmedians'].set_linewidth(2)
                for pb, color in zip(parts['bodies'], [PROP_TYPE_TO_COLOR[prop_type]]):
                    pb.set_facecolor(color)
                    pb.set_edgecolor(color)
                    pb.set_alpha(0.5)
            
            plt.axhline(0, ls='--', color='black')
            plt.grid(axis='y', ls='--', alpha=0.5)
            plt.xticks(range(len(PROP_TYPES)), PROP_TYPES, rotation=70, fontsize=fontsize-3)
            plt.yticks(fontsize=fontsize-3)
            plt.title(f'{MODEL_TO_PRETTY_NAME[model]}', fontsize=fontsize)
            plt.ylabel('Spearman r\nto amino-acid properties', fontsize=fontsize-2)

            plt.ylim([-0.29, 0.79])

            plt.tight_layout()
            plt.savefig(f'{plotsdir}/violinplot_{model}_vs_aminoacid_properties.png')
            plt.savefig(f'{plotsdir}/violinplot_{model}_vs_aminoacid_properties.pdf')
            plt.close()


    ## barplot vs. aminoacid properties across all/core/surface
    subdirs = ['all', 'surface', 'core']
    subdir_to_hatch = {
        'all': '',
        'surface': '...',
        'core': '///'
    }
    for base_matrix_type in ['T2837_all_sites', 'cdna117k']:

        for model in matrix_type_to_models[f'{base_matrix_type}/all']:

            all_means = []
            all_stddevs = []
            all_values = []
            for subdir in subdirs:
                matrix_type = f'{base_matrix_type}/{subdir}'

                means = []
                stddevs = []
                for prop_type in PROP_TYPES:
                    curr_corrs = []
                    for prop in PROP_TYPE_TO_PROPS[prop_type]:
                        curr_corrs.append(matrix_type_to_correlations[matrix_type][model][prop][0])
                    means.append(np.mean(curr_corrs))
                    stddevs.append(np.std(curr_corrs))
                    all_values.append(np.array(curr_corrs))
                all_means.append(means)
                all_stddevs.append(stddevs)
            
            all_means = np.array(all_means)
            all_stddevs = np.array(all_stddevs)

            N, K = all_means.shape

            # Bar height and spacing
            bar_thickness = 0.3  # Height of individual bars
            group_spacing = 0.4  # Space between model groups
            y = np.arange(N) * (K * bar_thickness + group_spacing)  # Y positions for models

            # Plot
            fig, ax = plt.subplots(figsize=(5, 4))

            # Plot each category's bars, shifting them within each group
            # for j in range(K):
            #     ax.barh(y + j * bar_height, all_means[:, j], height=bar_height, xerr=all_stddevs[:, j], 
            #             capsize=0, label=PROP_TYPES[j], color=PROP_TYPE_TO_COLOR[PROP_TYPES[j]], alpha=0.8)

            positions = []
            subdir_indices = []
            category_indices = []
            for i, yy in enumerate(y):
                for j in range(K):
                    positions.append(yy + j * bar_thickness)
                    category_indices.append(j)
                    subdir_indices.append(i)

            boxplots = ax.boxplot(all_values, vert=True, positions=positions, widths=bar_thickness, patch_artist=True, showfliers=False)

            for patch, median, whisker, cat_idx, subdir_idx in zip(boxplots['boxes'], boxplots['medians'], boxplots['whiskers'][::2], category_indices, subdir_indices):
                patch.set_facecolor(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]]) # set box color
                patch.set_alpha(0.6)
                patch.set_edgecolor(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])
                patch.set_hatch(subdir_to_hatch[subdirs[subdir_idx]])

                median.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])  # Set median color
                median.set_linewidth(1.5)  # Make median line thicker for better visibility

                whisker.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])

            for i, whisker in enumerate(boxplots['whiskers']):
                cat_idx = category_indices[i // 2]  # Every two whiskers belong to the same category
                whisker.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])
            

            for i, cap in enumerate(boxplots['caps']):
                cat_idx = category_indices[i // 2]  # Every two caps belong to the same category
                cap.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])
            
            # ax.axvline(0, ls='--', color='black')
            # ax.grid(axis='x', alpha=0.6)
            # ax.set_yticks(y + (K-1) * bar_height / 2)  # Center ticks under groups
            # ax.set_yticklabels(subdirs)
            # ax.set_xlabel('Spearman r\nto amino-acid properties', fontsize=fontsize)
            # ax.set_xlim([-0.29, 0.79])
            # ax.tick_params(axis='y', labelsize=fontsize)
            # ax.tick_params(axis='x', labelsize=fontsize-2)

            ax.axhline(0, ls='--', color='black')
            ax.grid(axis='y', alpha=0.6)
            ax.set_xticks(y + (K-1) * bar_thickness / 2)  # Center ticks under groups
            ax.set_xticklabels(subdirs)
            ax.set_ylabel('Spearman r\nto amino-acid properties', fontsize=fontsize)
            ax.set_ylim([-0.29, 0.79])
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-2)
            
            ax.set_title(MODEL_TO_PRETTY_NAME[model], fontsize=fontsize)
            # ax.set_ylabel("Models", fontsize=fontsize)
            # ax.legend(title="Categories", loc="best")
            # plt.title("Grouped Horizontal Bar Plot with Error Bars")

            # plt.tight_layout()
            # Manually adjust spacing: left, right, bottom, top (values between 0 and 1)
            plt.subplots_adjust(
                left=0.25,   # more space on the left
                right=0.95,  # less space on the right
                bottom=0.10, # more space below
                top=0.85      # more space above
            )
            plt.savefig(f'{base_matrix_type}/all/plots/comparison_{model}_vs_aminoacid_properties.png')
            plt.savefig(f'{base_matrix_type}/all/plots/comparison_{model}_vs_aminoacid_properties.pdf')
            plt.close()



    ## barplot vs. aminoacid properties across all/core/surface
    subdirs = ['all', 'surface', 'core']
    subdir_to_hatch = {
        'all': '',
        'surface': '...',
        'core': '///'
    }

    for horizontal in [True]:

        nrows = 1
        ncols = len(subdirs)
        figsize = (ncols * 4, nrows * 5)
        fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, sharex=True, sharey=True)

        for ax, subdir in zip(axs, subdirs):

            all_means = []
            all_stddevs = []
            all_values = []

            # models = matrix_type_to_models[f'{base_matrix_type}/{subdir}']
            models = ['neg_ddg', 'proteinmpnn_v_48_030', 'hermes_py_050', 'hermes_py_050_ft_cdna117k_relaxed_pred', 'hermes_py_050_ft_cdna117k_ddg_st', 'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st']
            if horizontal:
                models = models[::-1]
            model_pretty_names = [MODEL_TO_PRETTY_NAME[model].replace('\n', '') for model in models]

            for model in models:

                if model == 'neg_ddg':
                    base_matrix_type = 'cdna117k'
                else:
                    base_matrix_type = 'T2837_all_sites'

                matrix_type = f'{base_matrix_type}/{subdir}'

                means = []
                stddevs = []
                for prop_type in PROP_TYPES:
                    curr_corrs = []
                    for prop in PROP_TYPE_TO_PROPS[prop_type]:
                        curr_corrs.append(matrix_type_to_correlations[matrix_type][model][prop][0])
                    means.append(np.mean(curr_corrs))
                    stddevs.append(np.std(curr_corrs))
                    all_values.append(np.array(curr_corrs))
                all_means.append(means)
                all_stddevs.append(stddevs)
            
            all_means = np.array(all_means)
            all_stddevs = np.array(all_stddevs)

            N, K = all_means.shape

            # Bar height and spacing
            bar_thickness = 0.3  # Height of individual bars
            group_spacing = 0.4  # Space between model groups
            y = np.arange(N) * (K * bar_thickness + group_spacing)  # Y positions for models

            # Plot each category's bars, shifting them within each group
            # for j in range(K):
            #     ax.barh(y + j * bar_height, all_means[:, j], height=bar_height, xerr=all_stddevs[:, j], 
            #             capsize=0, label=PROP_TYPES[j], color=PROP_TYPE_TO_COLOR[PROP_TYPES[j]], alpha=0.8)

            positions = []
            model_indices = []
            category_indices = []
            for i, yy in enumerate(y):
                for j in range(K):
                    positions.append(yy + j * bar_thickness)
                    category_indices.append(j)
                    model_indices.append(i)

            boxplots = ax.boxplot(all_values, vert=False, positions=positions, widths=bar_thickness, patch_artist=True, showfliers=False)

            for patch, median, whisker, cat_idx, model_idx in zip(boxplots['boxes'], boxplots['medians'], boxplots['whiskers'][::2], category_indices, model_indices):
                patch.set_facecolor(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]]) # set box color
                patch.set_alpha(0.6)
                patch.set_edgecolor(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])
                patch.set_hatch(subdir_to_hatch[subdir])

                median.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])  # Set median color
                median.set_linewidth(1.5)  # Make median line thicker for better visibility

                whisker.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])

            for i, whisker in enumerate(boxplots['whiskers']):
                cat_idx = category_indices[i // 2]  # Every two whiskers belong to the same category
                whisker.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])
            

            for i, cap in enumerate(boxplots['caps']):
                cat_idx = category_indices[i // 2]  # Every two caps belong to the same category
                cap.set_color(PROP_TYPE_TO_COLOR[PROP_TYPES[cat_idx]])
            
            # if horizontal:
            ax.axvline(0, ls='--', color='black')
            ax.grid(axis='x', alpha=0.6)
            ax.set_yticks(y + (K-1) * bar_thickness / 2)  # Center ticks under groups
            ax.set_yticklabels(model_pretty_names)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6])
            if subdir == 'surface':
                ax.set_xlabel('Spearman r to amino-acid properties', fontsize=fontsize)
            ax.set_xlim([-0.19, 0.79])
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize-2)
            # else:
            #     ax.axhline(0, ls='--', color='black')
            #     ax.grid(axis='y', alpha=0.6)
            #     ax.set_xticks(y + (K-1) * bar_thickness / 2)  # Center ticks under groups
            #     ax.set_xticklabels(model_pretty_names, rotation=70, ha='right')
            #     ax.set_ylabel('Spearman r\nto amino-acid properties', fontsize=fontsize)
            #     ax.set_ylim([-0.29, 0.79])
            #     ax.tick_params(axis='x', labelsize=fontsize)
            #     ax.tick_params(axis='y', labelsize=fontsize-2)
            
            ax.set_title(subdir, fontsize=fontsize)
            # ax.set_ylabel("Models", fontsize=fontsize)
            # ax.legend(title="Categories", loc="best")
            # plt.title("Grouped Horizontal Bar Plot with Error Bars")

            # plt.tight_layout()
            # Manually adjust spacing: left, right, bottom, top (values between 0 and 1)
            # if horizontal:
            #     plt.subplots_adjust(
            #         left=0.55,   # more space on the left
            #         right=0.97,  # less space on the right
            #         bottom=0.20, # more space below
            #         top=0.902     # more space above
            #     )
            # else:
            #     plt.subplots_adjust(
            #         left=0.22,   # more space on the left
            #         right=0.95,  # less space on the right
            #         bottom=0.45, # more space below
            #         top=0.92      # more space above
            #     )
        
        plt.tight_layout()
        plt.savefig(f'comparison_all_models_vs_aminoacid_properties__horizontal={True}.png')
        plt.savefig(f'comparison_all_models_vs_aminoacid_properties__horizontal={True}.pdf')
        plt.close()


