
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.stats import gaussian_kde
from scipy.stats import shapiro, normaltest

AMINOACIDS = 'GPCAVILMFYWSTNQRHKDE'

model_version_list = ['hermes_py_000',
                      'hermes_py_050',
                      'hermes_py_000_ft_cdna117k_relaxed_pred',
                      'hermes_py_050_ft_cdna117k_relaxed_pred',
                      'hermes_py_000_ft_cdna117k_ddg_st',
                      'hermes_py_050_ft_cdna117k_ddg_st',
                      'hermes_py_000_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st',
                      'hermes_py_050_ft_cdna117k_relaxed_pred_ft_cdna117k_ddg_st']

outdir = './all/distributions/'
os.makedirs(outdir, exist_ok=True)

for model_version in model_version_list:

    print(model_version)

    df = pd.read_csv(f'/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/csv_files/{model_version}.csv')

    ## gather all distributions
    distributions = {'all': []}

    for i, row in tqdm(df.iterrows(), total=len(df)):

        aa_wt = row['resname']

        if aa_wt not in distributions:
            distributions[aa_wt] = {}

        for aa_mt in AMINOACIDS:

            if aa_mt not in distributions[aa_wt]:
                distributions[aa_wt][aa_mt] = []

            score = row[f'logit_{aa_mt}'] - row[f'logit_{aa_wt}']
            distributions[aa_wt][aa_mt].append(score)
            distributions['all'].append(score)
    
    ## now, plot them all!
    ## keep track of percentiles as well

    fontsize = 14
    percentiles_to_consider = [50, 60, 70, 75, 80, 85, 90, 95, 97.5, 99]

    plt.figure(figsize=(5, 3.5))
    scores = np.array(distributions['all'])
    kde = gaussian_kde(scores)
    x_vals = np.linspace(scores.min(), scores.max(), 200)
    y_vals = kde(x_vals)
    plt.plot(x_vals, y_vals, color="steelblue", lw=1.5)
    plt.axvline(0, color='black')

    for perc in percentiles_to_consider:
        perc_val = np.percentile(scores, perc)
        plt.axvline(perc_val, color="red", linestyle="--", lw=1)

    plt.title(f'any -> any', fontsize=fontsize+3)
    plt.tight_layout()

    outfile = os.path.join(outdir, f'{model_version}__distribution_all_mutations.png')
    plt.savefig(outfile)
    # plt.savefig(outfile.replace('.png', '.pdf'))
    plt.close()


    percentiles = {'all': {}}
    for perc in percentiles_to_consider:
        perc_val = np.percentile(np.array(distributions['all']), perc)
        # ax.axvline(perc_val, color="red", linestyle="--", lw=1)
        percentiles['all'][str(perc)] = perc_val

    # ncols = 20
    # nrows = 20
    # colsize = 3.2
    # rowsize = 2.5
    # fig, axs = plt.subplots(figsize=(ncols*colsize, nrows*rowsize), ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    for i_wt, aa_wt in tqdm(enumerate(AMINOACIDS), total=len(AMINOACIDS)):
        
        if aa_wt not in percentiles:
            percentiles[aa_wt] = {}
        
        for i_mt, aa_mt in enumerate(AMINOACIDS):

            if aa_wt == aa_mt: continue

            # ax = axs[i_wt, i_mt]
            scores = np.array(distributions[aa_wt][aa_mt])

            # stat, p = normaltest(scores)
            # print(aa_wt, aa_mt, stat, p)

            # kde = gaussian_kde(scores)
            # x_vals = np.linspace(scores.min(), scores.max(), 200)
            # y_vals = kde(x_vals)
            # ax.plot(x_vals, y_vals, color="steelblue", lw=1.5)

            # ax.axvline(0, color='black')

            percentiles[aa_wt][aa_mt] = {}
            for perc in percentiles_to_consider:
                perc_val = np.percentile(scores, perc)
                # ax.axvline(perc_val, color="red", linestyle="--", lw=1)
                percentiles[aa_wt][aa_mt][str(perc)] = perc_val

            # ax.set_title(f'{aa_wt} -> {aa_mt}', fontsize=fontsize)
        
    # plt.tight_layout()

    # outfile = os.path.join(outdir, f'{model_version}__distributions.png')
    # plt.savefig(outfile)
    # # plt.savefig(outfile.replace('.png', '.pdf'))
    # plt.close()

    with open(os.path.join(outdir, f'{model_version}__percentiles.json'), 'w+') as f:
        json.dump(percentiles, f, indent=4)

    

