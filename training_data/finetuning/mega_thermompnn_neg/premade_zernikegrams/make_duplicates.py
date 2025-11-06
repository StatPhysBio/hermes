
import os

new_model = 'hermes_py_050_ft_casp12_perc0p5_relaxed_pred'

for seed in range(10):
    os.system(f'cp hermes_py_050_seed=1000{seed}.npz.gz {new_model}_seed=1000{seed}.npz.gz')
