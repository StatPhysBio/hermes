

import os
import json
import argparse

## vary this for different models
MODELS = 'hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'

# MODELS = 'hermes_bp_000 hermes_bp_000_ft_cdna117k_ddg_st hermes_bp_000_ft_ros_ddg_st hermes_bp_050 hermes_bp_050_ft_cdna117k_ddg_st'
# MODELS = 'hermes_bp_050_ft_ros_ddg_st hermes_py_000 hermes_py_000_ft_cdna117k_ddg_st hermes_py_000_ft_ros_ddg_st hermes_py_000_untrained_ft_cdna117k_ddg_st'
# MODELS = 'hermes_py_050 hermes_py_050_ft_cdna117k_ddg_st hermes_py_050_ft_ros_ddg_st hermes_py_050_untrained_ft_cdna117k_ddg_st hermes_py_000_ft_esmfold_cdna117k_ddg_st'
# MODELS = 'hermes_py_050_ft_esmfold_cdna117k_ddg_st hermes_bp_000_ft_esmfold_cdna117k_ddg_st hermes_bp_050_ft_esmfold_cdna117k_ddg_st'


SLURM_SETUP = "#!/bin/bash\n\
#SBATCH --job-name={system_identifier}\n\
#SBATCH --account={account}\n\
#SBATCH --partition={partition}\n{gpu_text}\
#SBATCH --nodes=1\n\
#SBATCH --ntasks-per-node={num_cores}\n\
#SBATCH --time={walltime}\n\
#SBATCH --mem={memory}\n{email_text}\
#SBATCH -e {errfile}\n\
#SBATCH -o {outfile}"

# systems is all directories in current folder that do not start with "__"
SYSTEMS = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith("__")]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A',  '--account', type=str, default='stf')
    parser.add_argument('-P',  '--partition', type=str, default='compute-hugemem')
    parser.add_argument('-G',  '--use_gpu', type=int, default=0, choices=[0, 1])
    parser.add_argument('-C',  '--num_cores', type=int, default=1)
    parser.add_argument('-W',  '--walltime', type=str, default='2-00:00:00')
    parser.add_argument('-M',  '--memory', type=str, default='44G')
    parser.add_argument('-E',  '--send_emails', type=int, default=0, choices=[0, 1])
    parser.add_argument('-EA', '--email_address', type=str, default='gvisan01@uw.edu')
    args = parser.parse_args()


    logs_path = './__slurm_logs'
    os.makedirs(logs_path, exist_ok=True)
    
    if args.use_gpu:
        gpu_text = '#SBATCH --gres=gpu:1\n'
    else:
        gpu_text = ''
    
    if args.send_emails:
        email_text = f'#SBATCH --mail-type=ALL\n#SBATCH --mail-user={args.email_address}\n#SBATCH --export=all\n'
    else:
        email_text = ''
    
    for system_identifier in SYSTEMS:

        # make the hcnn.sh file in the system's directory
        with open(f'{system_identifier}/info.json', 'r') as f:
            info = json.load(f)

        with open('sbatch_hermes_base_text.txt', 'r') as f:
            hcnn_sh_text = f.read().format(model_version_list=MODELS, systems=' '.join(info['systems']), dms_columns=' '.join(info['dms_columns']))
        
        with open(f'{system_identifier}/hermes.sh', 'w+') as f:
            f.write(hcnn_sh_text)
    
        # make slrum job for it!
        slurm_text = SLURM_SETUP.format(system_identifier=system_identifier,
                                    account=args.account,
                                    partition=args.partition,
                                    gpu_text=gpu_text,
                                    num_cores=args.num_cores,
                                    walltime=args.walltime,
                                    memory=args.memory,
                                    email_text=email_text,
                                    errfile=os.path.join(logs_path, f"{system_identifier}.err"),
                                    outfile=os.path.join(logs_path, f"{system_identifier}.out"))

        slurm_text += '\n\n' + f'cd {system_identifier}\n' + 'bash hermes.sh\n'

        slurm_file = 'job.slurm'
        with open(slurm_file, 'w') as f:
            f.write(slurm_text)

        os.system(f"sbatch {slurm_file}")
