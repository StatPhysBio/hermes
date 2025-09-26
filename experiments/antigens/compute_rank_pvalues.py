import numpy as np
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests


max_mt_rank_to_xs = {
    3: np.array([15, 11, 8, 15, 15, 13]),
    6: np.array([16, 16, 11, 20, 16, 17]),
    20: np.array([16, 18, 12, 21, 19, 22])
}

n = 33

def get_p(max_mt_rank):
    num = np.sum(np.arange(19, 19 - max_mt_rank, -1))
    denom = 19 * 20
    return np.sum(num) / denom

def get_binom_pvalue(x, n, p):
    return 1 - binom.cdf(x - 1, n, p)


for max_mt_rank, xs in max_mt_rank_to_xs.items():
    print(f'max mt rank: {max_mt_rank}')

    p = get_p(max_mt_rank)

    pvs = [get_binom_pvalue(x, n, p) for x in xs]

    # Holm–Bonferroni correction
    _, pvs_corrected, _, _ = multipletests(pvs, alpha=0.05, method="holm")

    pvs_corrected_str = [f'{pv:.3f}' for pv in pvs_corrected]

    print('\t', pvs_corrected_str)
    print()
