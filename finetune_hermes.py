


import os, sys
import yaml
import json

import argparse

from hermes.finetuning.finetuning_hermes import finetune_single_model

this_file_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the finetuning config .yaml file.')
    parser.add_argument('-i', '--model_index', type=int, default=None, help='Index of the model to finetune. If None, all models in the model directory will be finetuned.')
    args = parser.parse_args()

    # load finetuning params
    with open(args.config, 'r') as f:
        finetuning_params = yaml.load(f, Loader=yaml.FullLoader)
    
    
    # get model directories to finetune
    hermes_models_dir = os.path.join(this_file_dir, 'trained_models', finetuning_params['model_version'])
    single_model_dirs = sorted(os.listdir(hermes_models_dir))

    if args.model_index is not None:
        single_model_dirs = [single_model_dirs[args.model_index]]

    for i, model_dir_name in enumerate(single_model_dirs):
        print(f'Finetuning model {i+1}/{len(single_model_dirs)}: {model_dir_name}')

        input_model_dir = os.path.join(hermes_models_dir, model_dir_name)
        output_model_dir = os.path.join(this_file_dir, 'trained_models', finetuning_params['model_version'] + f"_finetuned_with_{finetuning_params['finetuning_version']}", model_dir_name)

        finetune_single_model(input_model_dir, output_model_dir, finetuning_params)


