import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Mapping metric names to human-readable titles for plot labels
METRIC_TITLE_MAP = {
    'logprobas': 'log-probability',
    'probas': 'probability',
    'logits': 'logit value'
}

METRIC_TITLE_MAP_PLURAL = {
    'logprobas': 'log-probabilities',
    'probas': 'probabilities',
    'logits': 'logit values'
}


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


def plot_hermes_predictions(metric, df, name, output, center_wt):
    """
    Generates a heatmap of HERMES predictions for a given metric.
    Cells with an "x" indicate the wild-type amino-acid at that position.
    
    Parameters:
        metric (str): The metric to plot.
        df (DataFrame): Data containing predictions.
        name (str): Pdbid w/ or w/o chain identifier.
        output (str or None): Output directory.
        center_wt (bool): If True, centers data around wild-type values.
    """
    metric_columns = [col for col in df.columns if col.startswith(metric[:-1] + "_")]
    positions = df['resnum'].astype(str) + df['insertion_code'].fillna('')
    chain_per_position = df['chain']
    unique_chains = chain_per_position.unique()
    
    xtick_labels = df['resnum'].astype(str) + (df['chain'] if len(unique_chains) > 1 else '')
    heatmap_data = df[metric_columns].T
    heatmap_data.columns = positions
    heatmap_array = heatmap_data.to_numpy()
    
    wildtype_indices = [metric_columns.index(f"{metric[:-1]}_{res}") for res in df['resname']]
    wildtype_mask = np.zeros_like(heatmap_array, dtype=bool)
    wildtype_mask[wildtype_indices, np.arange(heatmap_array.shape[1])] = True
    
    plt.figure(figsize=(75, 12))
    
    if center_wt:
        wildtype_values = heatmap_array[wildtype_indices, np.arange(heatmap_array.shape[1])]
        shifted_heatmap = heatmap_array - wildtype_values
        orig_cmap = matplotlib.colormaps['RdBu_r']
        midpoint_shift = 1 - np.max(shifted_heatmap) / (np.max(shifted_heatmap) + np.abs(np.min(shifted_heatmap)))
        shifted_cmap = shifted_color_map(orig_cmap, midpoint=midpoint_shift)
        im = plt.imshow(shifted_heatmap, cmap=shifted_cmap, aspect="auto")
    else:
        im = plt.imshow(heatmap_array, cmap="RdBu_r", vmin=np.min(heatmap_array), vmax=np.max(heatmap_array), aspect="auto")
    
    # put an "x" by the heatmap cells corresponding to the wild-type amino-acid
    for i, j in zip(*wildtype_mask.nonzero()):
        plt.text(j, i, 'x', ha='center', va='center', color='black', fontsize=36)
    
    cbar = plt.colorbar(im, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=42)
    plt.xticks(np.arange(len(positions)), labels=[label if i % 10 == 0 else '' for i, label in enumerate(xtick_labels)], fontsize=42)
    plt.tick_params(axis='x', length=10)
    plt.yticks(np.arange(len(metric_columns)), [col.replace(metric[:-1] + "_", "") for col in metric_columns], fontsize=36)
    plt.xlabel("Sequence Position", fontsize=48, labelpad=15)
    plt.ylabel("Amino Acid", fontsize=48, labelpad=15)
    
    title_suffix = "Deviation from WT " if center_wt else "Predicted "
    plt.title(f"{title_suffix}{METRIC_TITLE_MAP_PLURAL[metric]} ({name})", fontsize=52, pad=20)
    cbar.set_label(("Deviation from WT " if center_wt else "") + METRIC_TITLE_MAP[metric], fontsize=42, labelpad=20)
    plt.tight_layout()
    
    output_path = f"{output or ''}aa_{metric}_per_pos_{name}{'_centeredWT' if center_wt else ''}.png"
    plt.savefig(output_path)
    plt.close()


def filter_df_by_pdb_and_chain(df, pdbid):
    """Helper function to filter dataframe by pdb and chain if specified."""
    if "_" in pdbid:  # Chain is specified
        pdb, chain = pdbid.split("_", 1)
        return df[(df['pdb'] == pdb) & (df['chain'] == chain)], f"{pdb}_{chain}"
    return df[df['pdb'] == pdbid], pdbid

def process_df(df, request, output, center_wt, chain_sep, pdbid=None):
    """Process dataframe by pdbid or all pdbs."""
    if pdbid:  # If pdbid is provided, filter dataframe accordingly
        for id in pdbid:
            print(f"Processing pdbid: {id}")
            pdb_df, name = filter_df_by_pdb_and_chain(df, id)

            if chain_sep:
                unique_chains = pdb_df['chain'].unique()
                for chain in unique_chains:
                    full_name = f"{name}_{chain}"
                    chain_df = pdb_df[pdb_df['chain'] == chain]
                    plot_request(request, chain_df, full_name, output, center_wt)
            else:
                plot_request(request, pdb_df, name, output, center_wt)
    else:  # Process all pdbs in dataframe
        unique_pdbs = df['pdb'].unique()
        print("Processing unique pdbs: ", unique_pdbs)
        for pdb in unique_pdbs:
            pdb_df = df[df['pdb'] == pdb]
            if chain_sep:
                unique_chains = pdb_df['chain'].unique()
                for chain in unique_chains:
                    chain_df = pdb_df[pdb_df['chain'] == chain]
                    name = f"{pdb}_{chain}"
                    plot_request(request, chain_df, name, output, center_wt)
            else:
                name = pdb
                plot_request(request, pdb_df, name, output, center_wt)


def plot_request(request, df, name, output, center_wt):
    """Plot for a single metric."""
    for req in request:
        plot_hermes_predictions(req, df, name, output, center_wt)


def main(csv_file, request, pdbid=None, output='', chain_sep=False, center_wt=False):
    df = pd.read_csv(csv_file)
    print("request: ", request)
    print("pdbid: ", pdbid)
    print("df.shape: ", df.shape)

    process_df(df, request, output, center_wt, chain_sep, pdbid)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmap plots of HERMES inference results")
    parser.add_argument("--csv_file", type=str, help="Path to CSV file containing HERMES inference results")
    parser.add_argument("--request", choices=['logprobas', 'probas', 'logits'], nargs='+', help="HERMES-predicted metrics to be plotted")
    parser.add_argument("--pdbid", type=str, nargs='+', help="PDB IDs to filter (use 'pdbid' or 'pdbid_CHAIN')")
    parser.add_argument("--chain_sep", action='store_true', help="Generate separate plots for each chain")
    parser.add_argument("--center_wt", action='store_true', help="Subtract the wild-type value within each site")
    parser.add_argument("--output", type=str, help="Output directory, otherwise plots are saved in the current directory")
    args = parser.parse_args()
    main(args.csv_file, args.request, args.pdbid, args.output, args.chain_sep, args.center_wt)

