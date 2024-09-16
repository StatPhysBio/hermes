
import os
import numpy as np
import pandas as pd





def preprocess_atlas(df):

    ## filter out the rows with missing Kd value (set to "n.d.")
    ## filter out rows that don't have a Kd_wt/Kd-mt ratio. NOTE: this excludes any row with Kd values present
    ##      but that are set to "greater than some large Kd". We could keep these ut re-computing the ratios would be a bit annoying and I don't want to right now
    ##      regardless, the resulting values would be imperfect. Still good for the purposes of classification though. Will update this in the future
    new_ratio_values = []
    for x in df['Kd_wt/Kd_mut']:
        try:
            x = float(x)
            new_ratio_values.append(x)
        except:
            new_ratio_values.append(np.nan)
    df['Kd_wt/Kd_mut'] = new_ratio_values

    bad_rows = np.logical_or.reduce([df['Kd_microM'] == 'n.d.', np.isnan(df['Kd_wt/Kd_mut'])])
    # bad_rows = df['Kd_microM'] == 'n.d.'
    df = df[~bad_rows]

    ## create PEPname column
    PEPname_list = []
    rows_to_drop = []
    for i, row in df.iterrows():
        PEPseq = row['PEPseq']
        PEP_mut = row['PEP_mut']
        if PEP_mut == 'WT':
            PEPname = PEPseq
        else:
            PEPname_arr = np.array([aa for aa in PEPseq])
            for mutation in PEP_mut.split('|'):
                mutation =  mutation.strip()
                aa_wt = mutation[0]
                aa_mut = mutation[-1]
                pos = int(mutation[1:-1])
                if aa_mut != PEPname_arr[pos-1]:
                    print('Mutation {} does not match PEPseq {} at position {} for TCRname {}. Dropping row.'.format(mutation, PEPseq, pos, row['TCRname']))
                    rows_to_drop.append(i)
                PEPname_arr[pos-1] = aa_wt
            PEPname = ''.join(PEPname_arr)
                
        PEPname_list.append(PEPname)
    df['PEPname'] = PEPname_list

    ## drop rows with bad mutations
    df = df.drop(rows_to_drop)

    ## collapse rows that have the same TCR, MHC, peptide, and respective mutations
    ## for now, just keep one of them, just for the purpose of figuring out statistics on the number of mutations etc
    ## TODO: change this when actually taking out binding affinity!!!
    # df = df.drop_duplicates(subset=['TCRname', 'MHCname', 'PEPname', 'TCR_mut', 'MHC_mut', 'PEP_mut'], keep='first')

    ## remove stupid space in mutations
    df['TCR_mut'] = [x.strip() for x in df['TCR_mut']]
    df['MHC_mut'] = [x.strip() for x in df['MHC_mut']]
    df['PEP_mut'] = [x.strip() for x in df['PEP_mut']]

    ## clean Kd
    ## NOTE: keeping values that are "greater than some large Kd". Just stripping the "greater than" sign. Imperfect but it's a starting point
    Kd_microM_clean = np.array([float(str(kd).split('+/-')[0].strip().strip('>')) for kd in df['Kd_microM']])
    df['-log(Kd)'] = -np.log(Kd_microM_clean)

    df['log(Kd_wt/Kd_mt)'] = np.log(df['Kd_wt/Kd_mut']) # higher --> stronger binding
    df['ddg'] = (8.314/4184)*(273.15 + 25.0) * (-np.log(df['Kd_wt/Kd_mut'])) # lower --> stronger binding

    ## make wt_pdb and mt_pdb columns
    ## also make column of kd_wt and kd_mt, leveraging the fat that the data_frame is sorted, with the wildtype system always coming before the mutant system
    wt_pdbs, mt_pdbs = [], []
    for i, row in df.iterrows():
        true_pdb = row['true_PDB']
        template_pdb = row['template_PDB']
        if isinstance(true_pdb, str) and not isinstance(template_pdb, str): # wildtype
            wt_pdbs.append(true_pdb)
            mt_pdbs.append(np.nan)


        elif not isinstance(true_pdb, str) and isinstance(template_pdb, str): # in-silico mutation
            wt_pdbs.append(template_pdb)
            def clean_mut(x):
                return str(x).replace(' ', '').replace('|', '.')
            mt_pdbs.append(f"{template_pdb}-{clean_mut(row['MHC_mut'])}-{clean_mut(row['MHC_mut_chain'])}-{clean_mut(row['TCR_mut'])}-{clean_mut(row['TCR_mut_chain'])}-{clean_mut(row['PEP_mut'])}")
            # assert os.path.exists(f'structures/designed_pdb/{mt_pdbs[-1]}.pdb')

        elif isinstance(true_pdb, str) and isinstance(template_pdb, str): # mutation with ground truth structure
            wt_pdbs.append(template_pdb)
            mt_pdbs.append(true_pdb)
    
    df['wt_pdb'] = wt_pdbs
    df['mt_pdb'] = mt_pdbs

    return df.reset_index(drop=True)






atlas = pd.read_excel('ATLAS.xlsx')
atlas = preprocess_atlas(atlas)



## make general "mutant" and "chain" columns for compatibility with HCNN code
atlas_cleaned = atlas.copy()

## make a single column for the mutations and a single column for the chain. the other columns can be used to identify where the mutation is
mutations = []
chains = []
for i, row in atlas_cleaned.iterrows():
    curr_mutations = []
    curr_chains = []

    if row['TCR_mut'] != 'WT':
        curr_mutations.append(row['TCR_mut'])
        curr_chains.append(row['TCR_PDB_chain'])

    if row['MHC_mut'] != 'WT':
        curr_mutations.append(row['MHC_mut'])
        curr_chains.append(row['MHC_mut_chain'])

    if row['PEP_mut'] != 'WT':
        curr_mutations.append(row['PEP_mut'])
        num_mutations = len(row['PEP_mut'].split('|'))
        curr_chains.append('|'.join(['C' for _ in range(num_mutations)]))
    
    mutations.append('|'.join(curr_mutations).replace(' ', ''))
    chains.append('|'.join(curr_chains).replace(' ', ''))

atlas_cleaned['mutant'] = mutations
atlas_cleaned['chain'] = chains

atlas_cleaned.to_csv('ATLAS_cleaned.csv', index=False)



