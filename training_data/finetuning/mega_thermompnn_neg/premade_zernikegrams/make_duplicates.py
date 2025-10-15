
import os

new_model = 'hermes_py_050_untrained'

for seed in range(10):
    os.system(f'cp hermes_py_050_seed=1000{seed}.npz.gz {new_model}_seed=1000{seed}.npz.gz')
