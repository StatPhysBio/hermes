
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./Ssym_dir/ssym_dir_ddg_experimental.csv')

ddg_scores = df['score']

fontsize = 17

stab_color = 'red'

plt.figure(figsize=(7, 5))
plt.hist(ddg_scores, bins=30, color='skyblue')
plt.axvline(0, color=stab_color, linewidth=1.5)
plt.title('ssym dataset', fontsize=fontsize+2)
plt.xlabel('$\Delta\Delta G$ of wild-type $\\to$ mutant', fontsize=fontsize)
plt.ylabel('number of mutations', fontsize=fontsize)
plt.xticks(fontsize=fontsize-2)
plt.yticks(fontsize=fontsize-2)
plt.text(0.05, 0.95, 'stabilizing', fontsize=fontsize-2, color=stab_color, transform=plt.gca().transAxes, ha='left', va='top')
plt.text(0.95, 0.95, 'de-stabilizing', fontsize=fontsize-2, color=stab_color, transform=plt.gca().transAxes, ha='right', va='top')
plt.tight_layout()
plt.savefig('ddg_scores_distribution.png')
plt.savefig('ddg_scores_distribution.pdf')
plt.show()
