
# Submitting slurm jobs made easy

In this folder, you can find scripts that make it easy to submit slurm jobs. Emphasis on submitting multiple jobs that use the same split/pipeline, but with different arguments.

### `submit.py`

All you have to do is specify a config file in `.yaml` format within `config/runtime/`. The config file should contain the following keys:

```
account: ACCOUNT
partition: PARITION
nodes: NUM_NODES
cores: NUM_CORES
walltime: WALLTIME
mem: MEMORY
jobname: 'job_{param_1}_{param_2}' # you can put parameter values in the job name!
logdir: '/path/to/slurm/logs' # the log files will be saved here, with the jobname as the filename

parameters:
  param_1: [val_1, val_2, val_3]
  param_2: [val_1, val_2]
  
command: |
  cd /path/to/script/
  python -u my_spript.py \
                --param_1 {param_1} \
                --param_2 {param_2}
```

Then, run:

```
python submit.py --config CONFIGNAME.yaml [--gpu] [--email]
```

The `--gpu` flag will request one gpu. The `--email` flag will toggle email notifications to your UW email.
