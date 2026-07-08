"""
color_by_value.py

PyMOL script: color a structure (PDB or CIF) by per-residue scores derived
from a per-residue amino-acid logit table (e.g. an inverse-folding or
masked-language-model output), using a divergent red--white--blue colormap
centered at 0.

Expected CSV format (one header row):

    pdb, chain, resname, resnum, insertion_code,
    logit_G, logit_A, logit_C, logit_S, logit_P, logit_T, logit_V, logit_D,
    logit_I, logit_L, logit_N, logit_M, logit_Q, logit_K, logit_E, logit_H,
    logit_F, logit_R, logit_Y, logit_W

`resname` is the wild-type residue as a single-letter code (G, A, C, ...).
`insertion_code` may be blank, ' ', '?', '.', or '-' for "no insertion code".

For each row, every logit is first centered by subtracting the wild-type
logit (the column matching `resname`); so the wild-type entry becomes 0,
and every other entry reads as "how much the model prefers AA X over the
wild type at this position".

The 20 centered logits are then condensed into a single score, controlled
by `value_col`:

    median, mean, max     aggregate across all 20 amino acids
    G, A, C, S, P, T,
    V, D, I, L, N, M,     a specific AA's centered logit
    Q, K, E, H, F,        (i.e. "how preferred is mutation to X")
    R, Y, W

The resulting score is mapped to the cartoon: red = wild type preferred,
white = neutral (~0), blue = substitution preferred. The color range is
symmetric around 0 by default so that white always lands exactly on 0.

USAGE (inside PyMOL):

    run color_by_value.py
    color_by_value mystructure.pdb, myvalues.csv

    # max relative logit (the most-preferred substitution at each site):
    color_by_value mystructure.pdb, myvalues.csv, value_col=max

    # pick a specific substitution (centered logit for proline):
    color_by_value mystructure.pdb, myvalues.csv, value_col=P

    # mean centered logit, with an explicit symmetric clip:
    color_by_value mystructure.pdb, myvalues.csv, value_col=mean, vmax=3.0

USAGE (from the shell):

    pymol -d "run color_by_value.py; color_by_value mystructure.pdb, myvalues.csv, value_col=max"
"""

import csv
import math
import os

from pymol import cmd


# 20 standard amino-acid one-letter codes (order matches the CSV columns,
# but the order itself doesn't affect any computation).
AA_LETTERS = list("GACSPTVDILNMQKEHFRYW")
LOGIT_COLS = ["logit_" + aa for aa in AA_LETTERS]


def _parse_value_col(value_col):
    """
    Resolve `value_col` into (mode, key):
      ('aggregate', 'median'|'mean'|'max')
      ('single',    one of the 20 AA letters)
    Returns (None, None) if unparseable.
    """
    vc = str(value_col).strip()
    if vc.lower().startswith("logit_"):
        vc = vc[6:]

    if vc.lower() in ("median", "mean", "max"):
        return ("aggregate", vc.lower())

    vc_upper = vc.upper()
    if vc_upper in AA_LETTERS:
        return ("single", vc_upper)

    return (None, None)


def _median(values):
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return float("nan")
    if n % 2:
        return vs[n // 2]
    return 0.5 * (vs[n // 2 - 1] + vs[n // 2])


def _aggregate(centered, mode, key):
    """Combine the 20 centered logits (dict AA -> value) into one score."""
    if mode == "single":
        return centered[key]
    vals = list(centered.values())
    if key == "median":
        return _median(vals)
    if key == "mean":
        return sum(vals) / len(vals)
    if key == "max":
        return max(vals)
    raise ValueError("Unrecognized value_col mode/key: %s/%s" % (mode, key))


def color_by_value(structure_file,
                   csv_file,
                   value_col='median',
                   vmin=None,
                   vmax=None,
                   obj_name=None,
                   missing_color='gray70',
                   palette='red white blue',
                   bg='white'):
    """
    Load `structure_file`, read the per-residue logit table from `csv_file`,
    center the logits by the wild-type residue, condense to one score per
    residue, and color the cartoon with a divergent colormap centered at 0.

    Arguments
    ---------
    structure_file : str
        Path to a .pdb or .cif structure file.
    csv_file : str
        Path to a CSV with columns
            pdb, chain, resname, resnum, insertion_code,
            logit_G, logit_A, ..., logit_W
        `resname` is the wild-type residue's one-letter code.
    value_col : str, optional
        How to condense the 20 centered logits per residue:
            'median', 'mean', 'max'              -- aggregate across all 20
            single AA letter ('G','A',...,'W')   -- that AA's centered logit
        Default 'median'.
    vmin, vmax : float, optional
        Color range. Default: symmetric +/- max(|score|) so white sits at 0.
        If only one is given, the other is set to its negative.
    obj_name : str, optional
        PyMOL object name. Defaults to the structure file's basename.
    missing_color : str
        Color for residues with no CSV entry. Default 'gray70'.
    palette : str
        Spectrum palette. Default 'red white blue' (red = wild type
        preferred, white = 0, blue = substitution preferred).
        Pass 'blue white red' to flip the direction.
    bg : str
        Background color. Default 'white'.
    """

    # ---- 0. Resolve value_col -------------------------------------------
    mode, key = _parse_value_col(value_col)
    if mode is None:
        print("[color_by_value] ERROR: value_col must be 'median', 'mean', "
              "'max', or one of the 20 amino-acid one-letter codes "
              "(%s). Got: %r" % (",".join(AA_LETTERS), value_col))
        return

    # ---- 1. Load structure ----------------------------------------------
    if not obj_name:
        obj_name = os.path.splitext(os.path.basename(structure_file))[0]
        obj_name = obj_name.replace('.', '_')  # safe for .pdb.gz etc.
    cmd.load(structure_file, obj_name)

    # ---- 2. Parse CSV ----------------------------------------------------
    rows = []
    bad_resname_rows = 0
    bad_logit_rows = 0
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
        if not chain_col:   missing.append('chain/chainid')
        if not resnum_col:  missing.append('resnum')
        if not resname_col: missing.append('resname')
        for c in LOGIT_COLS:
            if c not in fieldnames:
                missing.append(c)
        if missing:
            print("[color_by_value] ERROR: CSV missing required column(s): %s"
                  % ", ".join(missing))
            print("[color_by_value] Available columns: %s"
                  % ", ".join(fieldnames))
            return

        for row in reader:
            resname = (row.get(resname_col) or '').strip().upper()
            if resname not in AA_LETTERS:
                # Non-standard residue (e.g. modified AA, ligand row): skip.
                bad_resname_rows += 1
                continue
            try:
                ref_val = float(row['logit_' + resname])
                if math.isnan(ref_val):
                    bad_logit_rows += 1
                    continue
            except (TypeError, ValueError):
                bad_logit_rows += 1
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
                bad_logit_rows += 1
                continue

            score = _aggregate(centered, mode, key)

            icode = (row.get(icode_col) or '').strip() if icode_col else ''
            if icode in ('?', '.', '-'):
                icode = ''

            rows.append({
                'chain': (row.get(chain_col) or '').strip(),
                'resi':  str(row.get(resnum_col) or '').strip(),
                'icode': icode,
                'value': score,
            })

    if not rows:
        print("[color_by_value] No usable rows in CSV.")
        return

    # ---- 3. Decide color range ------------------------------------------
    values = [r['value'] for r in rows]
    if vmin is None and vmax is None:
        m = max(abs(min(values)), abs(max(values)), 1e-12)
        vmin, vmax = -m, m
    else:
        if vmin is None:
            vmin = -abs(float(vmax))
        if vmax is None:
            vmax = abs(float(vmin))
        vmin = float(vmin)
        vmax = float(vmax)

    label = key if mode == "aggregate" else ("logit_%s (centered)" % key)
    print("[color_by_value] Score: %s  |  range: vmin=%.4g, vmax=%.4g  |  palette: '%s'"
          % (label, vmin, vmax, palette))
    if bad_resname_rows or bad_logit_rows:
        print("[color_by_value] Skipped %d row(s) with non-standard resname, "
              "%d row(s) with missing/non-numeric logits."
              % (bad_resname_rows, bad_logit_rows))

    # ---- 4. Display setup -----------------------------------------------
    cmd.bg_color(bg)
    cmd.hide('everything', obj_name)
    cmd.show('cartoon', obj_name)
    cmd.color(missing_color, obj_name)

    # Use the occupancy column (q) as a "has data" flag and stash the score
    # in the B-factor (b) column. Then spectrum() does the gradient.
    cmd.alter(obj_name, "q=0.0")

    # ---- 5. Push scores into B-factor, flag matched residues -----------
    matched = 0
    not_found = 0
    for r in rows:
        resi_token = r['resi'] + r['icode']
        if resi_token.startswith('-'):
            resi_token = "\\" + resi_token  # escape leading minus for PyMOL

        if r['chain']:
            sel = "%s and chain %s and resi %s" % (obj_name, r['chain'], resi_token)
        else:
            sel = "%s and resi %s" % (obj_name, resi_token)

        n = cmd.alter(sel, "b=%f" % r['value'])
        if n == 0:
            not_found += 1
            continue
        cmd.alter(sel, "q=1.0")
        matched += 1

    cmd.rebuild()  # commit alter() changes

    if matched == 0:
        print("[color_by_value] WARNING: No CSV residues matched the structure.")
        return

    # ---- 6. Apply divergent colormap ------------------------------------
    cmd.spectrum('b',
                 palette,
                 "%s and q > 0.5" % obj_name,
                 minimum=vmin,
                 maximum=vmax)

    print("[color_by_value] Colored %d residues; %d CSV rows had no match in structure."
          % (matched, not_found))

    cmd.orient(obj_name)


# Register as a PyMOL command so you can call it from the command line:
#   color_by_value structure.pdb, values.csv, value_col=median
cmd.extend('color_by_value', color_by_value)