
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.stats import gaussian_kde
from scipy.stats import shapiro, normaltest

AMINOACIDS = 'GPCAVILMFYWSTNQRHKDE'

model_version_list = ['proteinmpnn_v_48_002',
                      'proteinmpnn_v_48_030']

outdir = './all/distributions/'
os.makedirs(outdir, exist_ok=True)

csv_dir = '/gscratch/spe/gvisan01/hermes/experiments/T2837/results_all_sites/thermompnn'


## gather all distributions
distributions = {'all': []}

for csv_file in os.listdir(csv_dir):
    df = pd.read_csv(os.path.join(csv_dir, csv_file))

    for i, row in df.iterrows():
        aa_wt = row['wildtype']
        aa_mt = row['mutation']
        score = -row['ddG_pred']

        if not np.isnan(score):
            if aa_wt not in distributions:
                distributions[aa_wt] = {}
            if aa_mt not in distributions[aa_wt]:
                distributions[aa_wt][aa_mt] = []
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

outfile = os.path.join(outdir, f'thermompnn__distribution_all_mutations.png')
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

with open(os.path.join(outdir, f'thermompnn__percentiles.json'), 'w+') as f:
    json.dump(percentiles, f, indent=4)



