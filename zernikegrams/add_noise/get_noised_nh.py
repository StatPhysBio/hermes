
import numpy as np
import h5py
from hdf5plugin import LZ4
import time
import argparse

from rich.progress import Progress

from zernikegrams.add_noise.noise_core import add_noise
from zernikegrams.utils import log_config as logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_in', type=str, required=True)
    parser.add_argument('--hdf5_out', type=str, required=True)
    parser.add_argument('--input_dataset_name', type=str, default='data')
    parser.add_argument('--output_dataset_name', type=str, default='data')
    parser.add_argument('--noise', type=float, default=0.5, help='Noise Stddev, in Angstroms')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time())
    
    rng = np.random.default_rng(args.seed)

    time.sleep(1)

    start = time.time()
    
    with h5py.File(args.hdf5_in, 'r') as f_in:

        N = len(f_in[args.input_dataset_name])

        dt = f_in[args.input_dataset_name].dtype
                
        with h5py.File(args.hdf5_out, 'w') as f_out:
            # Initialize dataset
            f_out.create_dataset(args.output_dataset_name,
                         shape=(N,),
                         maxshape=(None,),
                         dtype=dt,
                         compression=LZ4())
            
            f_out.create_dataset('seed',
                         shape=(1,),
                         maxshape=(None,),
                         dtype=np.float32,
                         compression=LZ4())
        
        with h5py.File(args.hdf5_out, 'r+') as f_out:
            with Progress() as bar:
                task = bar.add_task(f"Noise {args.seed}", total=N)
                for n in range(N):
                    np_protein = f_in[args.input_dataset_name][n]
                    np_protein_with_noise = add_noise(np_protein, args.noise, rng=rng)
                    f_out[args.output_dataset_name][n] = np_protein_with_noise
                    
                    bar.update(
                        task,
                        advance=1
                    )
            
            f_out['seed'][0] = args.seed
    
    logger.info(f"Done adding noise with seed {args.seed}")


if __name__ == "__main__":
    main()