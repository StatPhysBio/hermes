

import os
import argparse


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
    parser.add_argument('-P',  '--partition', type=str, default='gpu-2080ti')
    parser.add_argument('-G',  '--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('-C',  '--num_cores', type=int, default=1)
    parser.add_argument('-W',  '--walltime', type=str, default='23:00:00')
    parser.add_argument('-M',  '--memory', type=str, default='44G')
    parser.add_argument('-E',  '--send_emails', type=int, default=0, choices=[0, 1])
    parser.add_argument('-EA', '--email_address', type=str, default=None)
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
    
        slurm_text = SLURM_SETUP.format(system_identifier=system_identifier,
                                    account=args.account,
                                    partition=args.partition,
                                    gpu_text=gpu_text,
                                    num_cores=args.num_cores,
                                    walltime=args.walltime,
                                    memory=args.memory,
                                    email_text=email_text,
                                    errfile=os.path.join(logs_path, f"{system_identifier}_proteinmpnn.err"),
                                    outfile=os.path.join(logs_path, f"{system_identifier}_proteinmpnn.out"))

        slurm_text += '\n\n' + f'cd {system_identifier}\n' + 'bash proteinmpnn.sh\n'

        slurm_file = 'job.slurm'
        with open(slurm_file, 'w') as f:
            f.write(slurm_text)

        os.system(f"sbatch {slurm_file}")
