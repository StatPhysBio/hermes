

import os
import numpy as np

# import Bio.PDB as pdb
# import h5py
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import auc

from hermes.utils.protein_naming import *

markersize=4
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# TICK_SIZE = 32
# plt.rc('xtick', labelsize=TICK_SIZE)    
# plt.rc('ytick', labelsize=TICK_SIZE) 
from matplotlib.gridspec import GridSpec
plt.rcParams['font.size'] = 6.
plt.rcParams['font.family'] = "Arial"

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def x_cell(x,y, ax=None, **kwargs):
    ax.scatter(x, y, marker='x', **kwargs)
    ax = ax or plt.gca()
    
import matplotlib.colors as mcolors
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def keep_center_colormap(vmin, vmax, center=0):
    vmin = vmin - center
    vmax = vmax - center
    dv = max(-vmin, vmax) * 2
    N = int(256 * dv / (vmax-vmin))
    RdBu = cm.get_cmap('RdBu', N)
    newcolors = RdBu(np.linspace(0, 1, N))
    beg = int((dv / 2 + vmin) * N / dv)
    end = N - int((dv / 2 - vmax) * N / dv)
    newmap = mcolors.ListedColormap(newcolors[beg:end])
    return newmap


def plot_heatmap(df,hm_value,hm_title,ax,vmin=None,vmax=None,vcenter=0.,cax=None):
        if vmin == None:
            vmin,vmax = (
                np.min(
                    df[[hm_value.format(aa_to_one_letter[ind_to_aa_ward[x]])  
                   for x in range(20) ]].to_numpy()
                ),
                np.max(
                    df[[hm_value.format(aa_to_one_letter[ind_to_aa_ward[x]])  
                   for x in range(20) ]].to_numpy()
                )
            )

        newmap = keep_center_colormap(vmin,vmax)
        hm_im = ax.imshow(
            df[[hm_value.format(aa_to_one_letter[ind_to_aa_ward[x]])  
                   for x in range(20) ]],
            cmap='RdBu',
#             vmin=vmin,vmax=vmax
            norm=mcolors.TwoSlopeNorm(vmin=vmin,vmax=vmax,vcenter=vcenter)
        )
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels=df['mutant'])
        ax.set_xticks(range(20))
        ax.set_xticklabels(labels=[aa_to_one_letter[ind_to_aa_ward[x]] for x in range(20)])
        ax.set_title(hm_title)

        for j,row in df.iterrows():
            wt_idx = aa_to_ind_ward[one_letter_to_aa[row['wt']]]
            mut_idx = aa_to_ind_ward[one_letter_to_aa[row['mut']]]
            x_cell(
                wt_idx,
                j,
                color='black',
                ax=ax,
                s=1,
                linewidth=1,
            )
            mut_color = 'black'
            highlight_cell(
                mut_idx,
                j,
                ax=ax,
                color=mut_color,
                linestyle=':',
                linewidth=0.5
            )
        ax.plot(
            [-0.5,19.5],[19.5,19.5],
            color='k',
            linestyle='--',
            linewidth=0.5
        )
        if cax==None:
            pos = ax.get_position()
            fig = ax.get_figure()
            cax = fig.add_axes(
                [pos.x0-pos.width*0.1,pos.y0-pos.height*0.2,
                 pos.width,pos.height*0.02]
            )
        plt.colorbar(
            hm_im,
            cax=cax,
            orientation='horizontal',
            shrink=0.3,
        )
            
def plot_heatmaps(df,hm_values,hm_titles,filename=None,vcenter=0.):
    n = len(hm_values)
    fig = plt.figure(figsize=(4*n,4),dpi=300,constrained_layout=True)
    gs = GridSpec(
        1,n,
        figure=fig,
        width_ratios=[20]*n,
         #height_ratios=[20,20],
#         wspace=0.5,
#         hspace=0.5
    )
    hm_axes = [fig.add_subplot(gs[:,i]) for i in range(n)]
    
    plotted_fields = np.concatenate(
        [
            *[[x for x in df.columns if y.strip('{}') in x] for y in hm_values]
        ]
    )
    all_data = df[
        plotted_fields
    ]
    vmin,vmax = np.min(all_data.to_numpy()),np.max(all_data.to_numpy())
#     print(vmin,vmax)
#     print(all_data)
    for i,hm_value in enumerate(hm_values):
        plot_heatmap(df,hm_value,hm_titles[i],hm_axes[i],vmin,vmax,vcenter=vcenter,)
#         divider = make_axes_locatable(hm_axes[i])
#         cax = divider.append_axes("bottom", size="2%", pad=0.5,)

        
#     fig.text(-.05,0.280,'Neutral',fontsize=26,rotation=90)
#     fig.text(-.05,0.643,'Destabilizing',fontsize=26,rotation=90)
#     for ax in hm_axes:
#         ax.plot(
#             [-0.5,19.5],[19.5,19.5],
#             color='k',
#             linestyle='--',
#             linewidth=0.5
#         )
    
    
    fig.tight_layout()
    if isinstance(filename,str):
        plt.savefig(filename,bbox_inches='tight')
    plt.show()

def scatter_plots_ddG(df,value,coords,filename=None,lines=True,legend=True,ylabel=None,fontsize=8):
    fig = plt.figure(figsize=(2,2),dpi=300,constrained_layout=True)
    scatter_ax = fig.add_subplot()
    plot_scatter_ddG(
        df,value,coords,scatter_ax,
        filename=filename,lines=lines,legend=legend,
        ylabel=ylabel,
        fontsize=fontsize
    )
    if isinstance(filename,str):
        plt.savefig(filename,bbox_inches='tight')
    plt.show()
    
def plot_scatter_ddG(
    df, value, coords, scatter_ax,
    filename=None, lines=True, legend=True,
    ylabel=None,
    markersize=markersize,
    fontsize=8
):
    pearson_r,pearson_pval = pearsonr(
        df[~np.isnan(df['ddG'])][value],
        df[~np.isnan(df['ddG'])]['ddG'])
    spearman_r,spearman_pval = spearmanr(
        df[~np.isnan(df['ddG'])][value],
        df[~np.isnan(df['ddG'])]['ddG'])
    effect_to_c = {
        'Destabilizing':'#EC9C2C',
        'Neutral':'#374D89'
    }
    scatter_im = scatter_ax.scatter(
        df['ddG'],df[value],
        s=markersize,
        c=[effect_to_c[x] for x in df['effect']],
        zorder=2
    )
    if lines:
        scatter_ax.axvline(0,linestyle='--',color='k',zorder=1,linewidth=1)
        scatter_ax.axhline(0,linestyle='--',color='k',zorder=1,linewidth=1)
    scatter_ax.set_xlabel(
        r'stability effect, $\Delta\Delta G$',
        fontsize=fontsize,
#         rotation=90,
    )
    if ylabel == None:
        scatter_ax.set_ylabel(
            r'H-CNN prediction',
            fontsize=fontsize
        )
    else:
        scatter_ax.set_ylabel(
            ylabel,
            fontsize=fontsize,
        )
    if coords != None:
        scatter_ax.text(
            *coords,
            'Pearson corr. = {:.2f} \np-val = {:.2e}'.format(
                pearson_r,pearson_pval
            ) + '\n' + 'Spearman corr. = {:.2f} \np-val = {:.2e}'.format(
                spearman_r,spearman_pval
            )
        )


    if legend:
        scatter_ax.legend(
            handles=[
                mlines.Line2D(
                    [0], 
                    [0], 
                    marker='.', 
                    color='w', 
                    label='Scatter',
                    markerfacecolor='#EC9C2C', 
                    markersize=markersize
                ),
                mlines.Line2D(
                    [0], 
                    [0], 
                    marker='.', 
                    color='w', 
                    label='Scatter',
                    markerfacecolor='#374D89', 
                    markersize=markersize
                )
            ],
            labels=['destabilizing','neutral'],
            loc='center right',
            bbox_to_anchor=(2.1,.5)
        )



def dms_scatter_plot(
    df,
    dms_column, pred_column,
    dms_label=None, pred_label=None,
    filename=None, lines=False, legend=True,
    color='#84649B',
    markersize=4,
    fontsize=8
):
    fig = plt.figure(figsize=(2,2), dpi=300, constrained_layout=True)
    scatter_ax = fig.add_subplot()

    pearson_r, pearson_pval = pearsonr(
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][pred_column],
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][dms_column])
    spearman_r, spearman_pval = spearmanr(
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][pred_column],
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][dms_column])
    
    scatter_ax.scatter(
        df[dms_column], df[pred_column],
        s=markersize,
        c=color,
        alpha=0.1,
        zorder=2
    )

    if lines:
        scatter_ax.axvline(0, linestyle='--', color='k', zorder=1, linewidth=1)
        scatter_ax.axhline(0, linestyle='--', color='k', zorder=1, linewidth=1)
    
    dms_label = dms_label if dms_label is not None else dms_column
    scatter_ax.set_xlabel(
        dms_label,
        fontsize=fontsize
    )

    pred_label = pred_label if pred_label is not None else pred_column
    scatter_ax.set_ylabel(
        pred_label,
        fontsize=fontsize
    )

    if legend:
        scatter_ax.legend(
            handles=[
                mlines.Line2D(
                    [0], 
                    [0], 
                    marker='o', 
                    color='w', 
                    label='Scatter',
                    markerfacecolor=color, 
                    markersize=markersize
                )
            ],
            labels=['Pearson corr. = {:.2f} \np-val = {:.2e}'.format(pearson_r, pearson_pval)
                    + '\n' +
                    'Spearman corr. = {:.2f} \np-val = {:.2e}'.format(spearman_r, spearman_pval)],
            loc='center right',
            bbox_to_anchor=(2.1,.5)
        )
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

    return (pearson_r, pearson_pval), (spearman_r, spearman_pval)


def get_roc_curve(df,label_field,label,value_field):
    
    data_vals = df[value_field]
    pos_vals = df[df[label_field] == label][value_field]
    neg_vals = df[df[label_field] != label][value_field]
    
    n_pos = len(pos_vals)
    n_neg =len(neg_vals)
    n = n_pos + n_neg
    
    tp = np.zeros(shape=len(df)+2)
    fp = np.zeros(shape=len(df)+2)
    
    for i,thresh in enumerate(np.sort(data_vals)):
        tp[i+1] = np.count_nonzero(pos_vals <= thresh)/n_pos
        fp[i+1] = np.count_nonzero(neg_vals <= thresh)/n_neg
    tp[-1],fp[-1] = 1.,1.

    return fp,tp

def dms_roc_plot(df, dms_column, pred_column, dms_pos_value=None, dms_label=None, filename=None):

    dms_pos_value = dms_pos_value if dms_pos_value is not None else 1
    roc_wt_struct = get_roc_curve(
        df, dms_column, dms_pos_value, pred_column
    )
    auc_wt_struct = auc(*roc_wt_struct,)

    fig = plt.figure(figsize=(2,2),dpi=300,constrained_layout=True)

    roc_ax = fig.add_subplot()

    roc_ax.set_xlim(-0.05,1.05)
    roc_ax.set_ylim(-0.05,1.05)
    roc_ax.set_xlabel('FPR',fontsize=8)
    roc_ax.set_ylabel('TPR',fontsize=8)
    dms_label = dms_label if dms_label is not None else dms_column
    roc_ax.set_title(dms_label)

    roc_ax.plot([0,1],[0,1],linestyle=(0,(2,2,2)),c='k',linewidth=1.)

    roc_ax.plot(
        *roc_wt_struct,
        label="AUC = {:.2f}".format(auc_wt_struct),
        linewidth=2,
        c='#84649B',
        zorder=4
    #     alpha=0.7
    )

    roc_ax.set_xlim(-0.05,1.05)
    roc_ax.set_ylim(-0.05,1.05)
    roc_ax.legend(loc='center right') #,bbox_to_anchor=(2.5,.5))

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

    return auc_wt_struct



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
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

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def saturation_mutagenesis_heatmap(input_csv_file: str,
                                   mutant_score_column: str,
                                   mutant_column: str = 'mutant',
                                   aa_order: str = 'AILMFVPGWRHKDENQSTYC',
                                   show_mutants_in: str = 'columns',
                                   model_type: str = None):

    '''
    The code will output the heatmap in a png file with the name "heatmap_{input_csv_file.strip('.csv')}.png".
    Make the heatmap with mutations in the columns, then transpose it if it is requested differently.

    Parameters
    ----------
    input_csv_file : str
        Name of the csv file containing the data.
    mutant_score_column : str
        Name of the column in the input csv file that contains the predicted mutant scores.
    mutant_column : str
        Name of the column in the input csv file that contains the mutant names.
    aa_order : str
        Left-to-Right or Top-to-Bottom of the mutant amino-acids to show in the heatmap.
    show_mutants_in : str
        Whether to show the mutants in rows or columns.
    model_type : str
        Only used to choose the name on the prediction colorbar.
    '''

    aa_to_idx_in_order = {aa: i for i, aa in enumerate(aa_order)}

    df = pd.read_csv(input_csv_file)

    mutants = df[mutant_column].values

    # get native peptide, with "holes" if they occur
    # accomodate for resnums being arbitrary, not starting necessarily at 1
    pep_aa_and_resnum_strings = list(set([mut[:-1] for mut in mutants]))
    pep_aa, pep_resnum = [], []
    for string in pep_aa_and_resnum_strings:
        pep_aa.append(string[0])
        pep_resnum.append(int(string[1:]))
    pep_aa = np.array(pep_aa)
    pep_resnum = np.array(pep_resnum)
    start_resnum = np.min(pep_resnum)
    pep_resnum = pep_resnum - start_resnum
    native_peptide = np.array(['-'] * (np.max(pep_resnum) + 1))
    for aa, resnum in zip(pep_aa, pep_resnum):
        native_peptide[resnum] = aa
    # the native peptide is not even used...
    # "start_resnum" is used though and it's important

    ncols = 20
    nrows = native_peptide.shape[0]
    heatmap = np.full((nrows, ncols), np.nan)
    patches_mutants_in_columns, patches_mutants_in_rows = [], []
    for i, df_row in df.iterrows():
        aa_wt = df_row[mutant_column][0]
        resnum = int(df_row[mutant_column][1:-1])
        aa_mt = df_row[mutant_column][-1]

        col = aa_to_idx_in_order[aa_mt] # mutant
        row = resnum - start_resnum # position

        heatmap[row, col] = df_row[mutant_score_column]

        if aa_wt == aa_mt:

            patches_mutants_in_columns.append(mpl.patches.Rectangle(
                    (-0.5 + col, -0.5 + row),
                    1.0,
                    1.0,
                    edgecolor='black',
                    fill=False,
                    lw=2
                ))
            
            patches_mutants_in_rows.append(mpl.patches.Rectangle(
                    (-0.5 + row, -0.5 + col),
                    1.0,
                    1.0,
                    edgecolor='black',
                    fill=False,
                    lw=2
                ))
    

    if show_mutants_in == 'columns':

        colsize = 0.25 * ncols
        rowsize = 0.50 * nrows
        plt.figure(figsize=(colsize, rowsize))
        plt.imshow(heatmap, cmap='Blues')

        plt.xticks(np.arange(ncols), aa_order)
        plt.yticks(np.arange(nrows), np.arange(1, nrows + 1))

        plt.xlabel('Amino acid', fontsize=12)
        plt.ylabel('Position within peptide', fontsize=12)

        ax = plt.gca()
        ax.set_aspect(2) # change aspect ratio
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False) # move xticks to the top
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.xaxis.set_label_coords(0.5, 1.19)
        ax.yaxis.set_label_coords(-0.1, 0.5)

        for patch in patches_mutants_in_columns:
            ax.add_patch(patch)
    
    else:
        
        temp = nrows
        nrows = ncols
        ncols = temp
        heatmap = heatmap.T

        colsize = 0.50 * ncols
        rowsize = 0.25 * nrows
        plt.figure(figsize=(colsize, rowsize))
        plt.imshow(heatmap, cmap='Blues')

        plt.yticks(np.arange(nrows), aa_order)
        plt.xticks(np.arange(ncols), np.arange(1, ncols + 1))

        ax = plt.gca()
        ax.set_aspect(0.5) # change aspect ratio
        # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False) # move xticks to the top
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        for patch in patches_mutants_in_rows:
            ax.add_patch(patch)
    
    LABEL_TABLE = {
        'hcnn': r'HCNN prediction, $\Delta logP$',
        'proteinmpnn': r'ProteinMPNN prediction, $\Delta logP$',
        None: r'Prediction'
    }

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(LABEL_TABLE[model_type], rotation=270, fontsize=12)
    cbar.ax.yaxis.set_label_coords(4.2, 0.5)
    plt.tight_layout()

    dir = '/'.join(input_csv_file.split('/')[:-1])
    file = input_csv_file.split('/')[-1]
    plt.savefig(os.path.join(dir, 'heatmap_' + file.replace('.csv', '.png')), dpi=300)
    plt.savefig(os.path.join(dir, 'heatmap_' + file.replace('.csv', '.pdf')), dpi=300)
    plt.close()

        









