"""
expand_mutations.py

Expand a CSV of single-residue mutations into all 20 amino-acid substitutions per row.

Input CSV must have a column with mutations in single-letter format like: S190F (original S at position 190 -> mutant F).
The script will produce one row per possible target amino-acid (A,C,D,...,Y), keep all other metadata columns,
and add a 'desired_one' column set to 1 only for the mutant observed in the input file.

Usage:
    python expand_mutations.py -i before.csv -o after.csv
    python expand_mutations.py --input before.csv --output after.csv --mutcol mutation

Options:
    -i/--input      input CSV path
    -o/--output     output CSV path
    --mutcol        name of mutation column (default: "mutation")
    --aalist        comma-separated list of amino acids to use (default: standard 20)
"""
import argparse
import logging
import re
import sys
from typing import List

import pandas as pd

# Standard 20 amino acids (single-letter)
DEFAULT_AAS = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

# Regex that accepts forms like "S190F" or "p.S190F" (optionally with whitespace)
MUT_REGEX = re.compile(r'^(?:p\.)?\s*([A-Za-z])\s*(\d+)\s*([A-Za-z])$', flags=re.IGNORECASE)


def parse_mutation(mut: str):
    """
    Parse mutation string like 'S190F' or 'p.S190F' -> returns (orig, pos, mutant)
    Raises ValueError if parsing fails.
    """
    if pd.isna(mut):
        raise ValueError("mutation is NaN/empty")
    s = str(mut).strip()
    m = MUT_REGEX.match(s)
    if not m:
        raise ValueError(f"cannot parse mutation '{mut}' (expected like 'S190F' or 'p.S190F')")
    orig = m.group(1).upper()
    pos = m.group(2)  # keep as string so formatting remains identical (no leading zeros expected)
    target = m.group(3).upper()
    return orig, pos, target


def expand_row(row: pd.Series, mutcol: str, aas: List[str]):
    """
    Given a pandas Series (row), parse its mutation, and return a list of dicts
    each representing the row expanded for one amino-acid substitution.
    """
    base_mut = row.get(mutcol)
    orig, pos, target = parse_mutation(base_mut)
    out_rows = []
    for aa in aas:
        new_mutation = f"{orig}{pos}{aa}"
        new_row = row.to_dict()  # copy all metadata columns
        new_row[mutcol] = new_mutation
        new_row["desired_one"] = int(aa == target)
        out_rows.append(new_row)
    return out_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Expand single-residue mutations into all 20 AA substitutions")
    parser.add_argument("-i", "--input", required=True, help="input CSV file")
    parser.add_argument("-o", "--output", required=True, help="output CSV file")
    parser.add_argument("--mutcol", default="mutation", help="name of the mutation column (default: 'mutation')")
    parser.add_argument("--aalist", default=",".join(DEFAULT_AAS),
                        help="comma-separated list of AAs to expand to (default: standard 20)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    aas = [a.strip().upper() for a in args.aalist.split(",") if a.strip()]
    if not aas:
        logging.error("No amino acids provided in --aalist")
        sys.exit(1)

    # Read input CSV
    try:
        df = pd.read_csv(args.input, dtype=str)  # read everything as string to preserve formatting
    except Exception as e:
        logging.error(f"Failed to read input CSV '{args.input}': {e}")
        sys.exit(1)

    if args.mutcol not in df.columns:
        logging.error(f"Mutation column '{args.mutcol}' not found in input CSV columns: {list(df.columns)}")
        sys.exit(1)

    out_rows = []
    errors = []
    total = len(df)
    logging.info(f"Expanding {total} input rows into {len(aas)} substitutions each (total ~ {total * len(aas)} rows)")

    for idx, row in df.iterrows():
        try:
            expanded = expand_row(row, args.mutcol, aas)
            out_rows.extend(expanded)
        except Exception as e:
            errmsg = f"Row {idx} (mutation='{row.get(args.mutcol)}') skipped: {e}"
            errors.append(errmsg)
            logging.warning(errmsg)

    if not out_rows:
        logging.error("No expanded rows produced. Exiting.")
        sys.exit(1)

    out_df = pd.DataFrame(out_rows)

    # Ensure desired_one is last column (optional)
    cols = [c for c in df.columns if c in out_df.columns]  # preserve original column order when possible
    if "desired_one" not in cols:
        cols.append("desired_one")
    else:
        # ensure it's at the end
        cols = [c for c in cols if c != "desired_one"] + ["desired_one"]

    # Add any other columns that were introduced or existed but not in original order
    for c in out_df.columns:
        if c not in cols:
            cols.append(c)

    out_df = out_df[cols]

    try:
        out_df.to_csv(args.output, index=False)
    except Exception as e:
        logging.error(f"Failed to write output CSV '{args.output}': {e}")
        sys.exit(1)

    logging.info(f"Wrote expanded CSV to '{args.output}' ({len(out_df)} rows).")
    if errors:
        logging.info(f"{len(errors)} rows could not be parsed; see warnings above.")


if __name__ == "__main__":
    main()
