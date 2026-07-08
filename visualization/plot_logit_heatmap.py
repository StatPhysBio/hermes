#!/usr/bin/env python
"""
plot_logit_heatmap.py

Plot a heatmap of resname-centered logit values from a per-residue logit CSV
(the same CSV format used by color_by_value.py).

Expected CSV columns (one header row):

    pdb, chain, resname, resnum, insertion_code,
    logit_G, logit_A, logit_C, logit_S, logit_P, logit_T, logit_V, logit_D,
    logit_I, logit_L, logit_N, logit_M, logit_Q, logit_K, logit_E, logit_H,
    logit_F, logit_R, logit_Y, logit_W

For each residue, the 20 logits are centered by subtracting the wild-type
logit (the column matching `resname`), so the wild-type cell is exactly 0
and every other cell reads as "how much the model prefers this AA over
the wild type at this position".

Output: a 20 x N heatmap split into multiple panels of <= max-per-panel
residues each, grouped by chain. Wild-type cells are marked with a small
black 'x'. The colormap is divergent red-white-blue with white at 0.

USAGE
-----
    python plot_logit_heatmap.py logits.csv
    python plot_logit_heatmap.py logits.csv -o heatmap.pdf --chains A B
    python plot_logit_heatmap.py logits.csv --max-per-panel 150 --tick-every 5
    python plot_logit_heatmap.py logits.csv --vmax-clip 5.0  --aa-order property
"""

import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# 20 standard amino-acid one-letter codes (must match the logit_* columns).
AA_LETTERS = list("GACSPTVDILNMQKEHFRYW")

# Default row order on the heatmap y-axis: alphabetical.
DEFAULT_AA_ROW_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# Optional alternative: grouped by chemical property
# (special | hydrophobic | polar | + charged | - charged).
PROPERTY_AA_ROW_ORDER = list("GPCAVILMFYWSTNQRHKDE")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_csv(csv_file, chains=None):
    """
    Read the logit CSV and return a dict mapping chain_id to a list of
    residue records. Each record:
        {'resname': 'A', 'resnum': 42, 'icode': '',
         'centered': {'G': float, 'A': float, ...}}
    """
    chain_data = {}
    bad_resname = 0
    bad_logit = 0

    with open(csv_file, 'r') as fh:
        reader = csv.DictReader(fh)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]

        def find_col(*aliases):
            for a in aliases:
                if a in fieldnames:
                    return a
            return None

        chain_col   = find_col('chain', 'chainid')
        resnum_col  = find_col('resnum', 'resi', 'residue_number')
        resname_col = find_col('resname', 'aa')
        icode_col   = find_col('insertion_code', 'icode', 'ins_code')

        missing = []
        if not chain_col:   missing.append('chain')
        if not resnum_col:  missing.append('resnum')
        if not resname_col: missing.append('resname')
        for aa in AA_LETTERS:
            if 'logit_' + aa not in fieldnames:
                missing.append('logit_' + aa)
        if missing:
            raise ValueError("CSV missing required columns: "
                             + ", ".join(missing))

        for row in reader:
            ch = (row.get(chain_col) or '').strip()
            if chains is not None and ch not in chains:
                continue

            resname = (row.get(resname_col) or '').strip().upper()
            if resname not in AA_LETTERS:
                bad_resname += 1
                continue

            try:
                ref_val = float(row['logit_' + resname])
                if math.isnan(ref_val):
                    bad_logit += 1
                    continue
            except (TypeError, ValueError):
                bad_logit += 1
                continue

            centered = {}
            bad = False
            for aa in AA_LETTERS:
                try:
                    v = float(row['logit_' + aa])
                except (TypeError, ValueError):
                    bad = True
                    break
                if math.isnan(v):
                    bad = True
                    break
                centered[aa] = v - ref_val
            if bad:
                bad_logit += 1
                continue

            icode = (row.get(icode_col) or '').strip() if icode_col else ''
            if icode in ('?', '.', '-'):
                icode = ''

            try:
                resnum_int = int(str(row.get(resnum_col, '')).strip())
            except (TypeError, ValueError):
                continue

            chain_data.setdefault(ch, []).append({
                'resname': resname,
                'resnum':  resnum_int,
                'icode':   icode,
                'centered': centered,
            })

    if bad_resname or bad_logit:
        print("[plot_logit_heatmap] Skipped %d non-standard-resname rows, "
              "%d rows with missing/non-numeric logits."
              % (bad_resname, bad_logit), file=sys.stderr)

    return chain_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(csv_file,
                 output_file=None,
                 chains=None,
                 max_residues_per_panel=250,
                 aa_row_order=None,
                 tick_every=10,
                 vmax_clip=None,
                 cell_width=0.06,
                 cell_height=0.18,
                 dpi=200):
    """
    Build the figure. Returns the matplotlib Figure object (and writes to
    `output_file` if given).
    """
    if aa_row_order is None:
        aa_row_order = DEFAULT_AA_ROW_ORDER
    aa_row_order = list(aa_row_order)
    if set(aa_row_order) != set(AA_LETTERS):
        raise ValueError("aa_row_order must contain exactly the 20 standard "
                         "amino-acid one-letter codes.")

    chain_data = parse_csv(csv_file, chains=chains)
    if not chain_data:
        raise ValueError("No data after filtering. "
                         "Are the requested chains present in the CSV?")

    if chains is not None:
        not_found = set(chains) - set(chain_data.keys())
        if not_found:
            print("[plot_logit_heatmap] Warning: requested chain(s) not "
                  "found in CSV: %s" % ", ".join(sorted(not_found)),
                  file=sys.stderr)

    # Build panels: split each chain into balanced chunks. A chain of length
    # L gets ceil(L / max_per_panel) chunks of ~equal size (ceil(L/n_chunks)).
    panels = []  # list of (chain_id, residue_records, start_index_in_chain)
    for ch in sorted(chain_data.keys()):
        residues = chain_data[ch]
        L = len(residues)
        n_chunks = max(1, math.ceil(L / max_residues_per_panel))
        chunk_size = math.ceil(L / n_chunks)
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, L)
            if start < L:
                panels.append((ch, residues[start:end], start))

    # All panels get the same x-axis extent so cell size is consistent.
    max_panel_width = max(len(p[1]) for p in panels)
    n_aas = len(aa_row_order)

    # Symmetric color range across the whole figure.
    all_vals = []
    for ch in chain_data:
        for r in chain_data[ch]:
            all_vals.extend(r['centered'].values())
    m = max(abs(v) for v in all_vals) if all_vals else 1.0
    if vmax_clip is not None:
        m = min(m, abs(float(vmax_clip)))

    # Figure size: based on the widest panel + panel count.
    fig_w = max_panel_width * cell_width + 2.5
    fig_h = len(panels) * (n_aas * cell_height + 0.6) + 1.0

    fig, axes = plt.subplots(
        nrows=len(panels), ncols=1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    cmap = plt.cm.RdBu  # low = red, mid = white, high = blue
    norm = mcolors.Normalize(vmin=-m, vmax=m)

    last_im = None
    for panel_idx, (ch, chunk, start_idx) in enumerate(panels):
        ax = axes[panel_idx, 0]
        n = len(chunk)

        # Build matrix: rows = aa_row_order, cols = residues in chunk.
        matrix = np.full((n_aas, n), np.nan)
        for j, r in enumerate(chunk):
            for i, aa in enumerate(aa_row_order):
                matrix[i, j] = r['centered'][aa]

        last_im = ax.imshow(
            matrix, aspect='auto', cmap=cmap, norm=norm,
            interpolation='nearest', origin='upper',
        )

        # Mark wild-type cells with a small black 'x'.
        for j, r in enumerate(chunk):
            wt_row = aa_row_order.index(r['resname'])
            ax.plot(j, wt_row, marker='x', color='black',
                    markersize=4.5, markeredgewidth=0.9,
                    linestyle='None')

        # Y axis: amino-acid labels.
        ax.set_yticks(range(n_aas))
        ax.set_yticklabels(aa_row_order, fontsize=12)
        ax.tick_params(axis='y', length=0, pad=2)

        # X axis: a tick at every residue, but a label only at every
        # `tick_every` residues (plus the first and last residue of the
        # chunk for context).
        positions = list(range(n))
        labels = [''] * n
        for j, r in enumerate(chunk):
            if r['resnum'] % tick_every == 0:
                labels[j] = str(r['resnum']) + (r['icode'] or '')
        if not labels[0]:
            labels[0] = str(chunk[0]['resnum']) + (chunk[0]['icode'] or '')
        if not labels[-1]:
            labels[-1] = str(chunk[-1]['resnum']) + (chunk[-1]['icode'] or '')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=12)
        ax.tick_params(axis='x', length=2, pad=1)

        # Pad x-axis so all panels visually share the same width.
        ax.set_xlim(-0.5, max_panel_width - 0.5)

        # Thin white grid between cells for legibility.
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_aas, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.3)
        ax.tick_params(which='minor', length=0)

        # Chain label on the left.
        ax.set_ylabel("Chain %s" % ch, fontsize=18,
                      fontweight='bold', labelpad=8)

    # Shared colorbar to the right of all panels.
    cbar = fig.colorbar(
        last_im, ax=axes.ravel().tolist(),
        orientation='vertical',
        fraction=0.012, pad=0.012,
        shrink=0.85,
    )
    cbar.set_label('Centered logit\n(relative to wild type)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print("[plot_logit_heatmap] Saved to %s" % output_file)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_aa_order(arg):
    if arg is None or arg.lower() in ('alpha', 'alphabetical'):
        return DEFAULT_AA_ROW_ORDER
    if arg.lower() == 'property':
        return PROPERTY_AA_ROW_ORDER
    return list(arg.upper())


def main():
    p = argparse.ArgumentParser(
        description="Plot a heatmap of resname-centered logits from a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('csv',
                   help='CSV with per-residue logits (logit_G ... logit_W)')
    p.add_argument('-o', '--output',
                   help='Output image (.png/.pdf/.svg). '
                        'Default: <csv basename>_heatmap.png')
    p.add_argument('--chains', nargs='+', default=None,
                   help='Restrict to these chain IDs '
                        '(e.g. --chains A B   or   --chains A,B)')
    p.add_argument('--max-per-panel', type=int, default=250,
                   help='Max residues per heatmap panel before splitting '
                        '(default: 250)')
    p.add_argument('--tick-every', type=int, default=5,
                   help='Residue-number tick every N residues (default: 5)')
    p.add_argument('--vmax-clip', type=float, default=None,
                   help='Clip the symmetric color scale to +/- this value '
                        '(helpful if a few sites have very extreme logits)')
    p.add_argument('--aa-order', default='property',
                   help='AA row order: "property" (default), "alpha", or a '
                        '20-letter custom ordering like ACDEFGHIKLMNPQRSTVWY')
    p.add_argument('--cell-width', type=float, default=0.08,
                   help='Width of each heatmap cell in inches (default: 0.08)')
    p.add_argument('--cell-height', type=float, default=0.16,
                   help='Height of each heatmap cell in inches (default: 0.16)')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    # Accept comma-separated chains too: --chains A,B,C
    chains = None
    if args.chains:
        chains = []
        for c in args.chains:
            chains.extend(c.split(','))
        chains = [c.strip() for c in chains if c.strip()]

    output = args.output or (os.path.splitext(args.csv)[0] + '_heatmap.png')
    aa_order = _parse_aa_order(args.aa_order)

    plot_heatmap(
        args.csv,
        output_file=output,
        chains=chains,
        max_residues_per_panel=args.max_per_panel,
        aa_row_order=aa_order,
        tick_every=args.tick_every,
        vmax_clip=args.vmax_clip,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()