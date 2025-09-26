
import csv
from dateutil import parser
import pandas as pd


def write_fasta(sequences, output_file, line_width=80):
    with open(output_file, "w") as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n")
            # Wrap sequence to line_width
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + "\n")

def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test


if __name__ == '__main__':

    data_path = "./"
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : 3.5, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }

    train, valid, test = build_training_clusters(params, False)

    # each item has two entries: "pdbid_chainid" and "hash"
    # use the hash to get the sequences
    train_hashes = list(train.keys())
    print(train_hashes[:4])
    
    df = pd.read_csv(params['LIST'])

    # Filter rows where hash is in the list
    train_df = df[df['HASH'].isin(train_hashes)]

    # Keep only the first occurrence of each hash
    train_df = train_df.drop_duplicates(subset='HASH', keep='first')

    print(len(train_df))

    ## make fasta file
    pdbid_to_seq = {}
    for _, row in train_df.iterrows():
        pdbid_to_seq[row['CHAINID']] = row['SEQUENCE']
    
    write_fasta(pdbid_to_seq, "../proteinmpnn.fasta")
    



