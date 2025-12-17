
import os
import yaml
import json
import argparse

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from hermes.pretraining.pretraining_hermes_divided_dataset import pretraining_hermes_divided_dataset, hermes_inference_on_test_data
from hermes.utils.argparse import optional_str

def confusion_matrix(args, targets, best_indices, accuracy):
    from hermes.utils.protein_naming import ind_to_ol_size
    original_order__ind_to_aa = [ind_to_ol_size[i] for i in range(20)]
    clustering_order__ind_to_aa = ['G', 'P', 'W', 'F', 'Y', 'A', 'S', 'K', 'R', 'Q', 'E', 'M', 'H', 'C', 'T', 'D', 'N', 'L', 'V', 'I']
    clustering_order__aa_to_ind = {aa: i for i, aa in enumerate(clustering_order__ind_to_aa)}

    indices_original_to_clustering = np.array([original_order__ind_to_aa.index(ind) for ind in clustering_order__ind_to_aa])

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, best_indices, labels=indices_original_to_clustering)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(20), clustering_order__ind_to_aa, fontsize=12)
    plt.yticks(np.arange(20), clustering_order__ind_to_aa, fontsize=12)
    plt.colorbar()
    plt.xlabel('Predicted Highest Proba AA', fontsize=14)
    plt.ylabel('True AA', fontsize=14)
    plt.title(f'Network Confusion (Acc: {accuracy:.3f})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'confusion_matrix.png'))
    plt.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--model_dir', type=str, default='runs/so3_convnet-zernike-ns-lmax=6-rst_norm=square__4', help='Where to store the model and related output')
    parser.add_argument('-c', '--config', type=optional_str, default='config/so3_convnet.yaml', help='Path to training config file. Ignored when --eval_only is set.')
    parser.add_argument('-n', '--train_with_noise', type=int, default=0, help='Whether to train with 0.5 Angstrom noise (data must have already been made)')
    parser.add_argument('-i', '--model_index', type=int, default=0, help='Index of the model to pretrain, will effectively change the noise seed of both the data and of the training (data must have already been made).')

    parser.add_argument('--pretrained_model_dir', type=optional_str, default=None, help='Path to optional already-pretrained model directory. Ignored when --eval_only is set.')
    parser.add_argument('--single_training_dataset', action='store_true', default=False, help='currently not implemented')
    parser.add_argument('--eval_only', action='store_true', default=False, help='for debugging purposes')
    args = parser.parse_args()

    if args.train_with_noise:
        args.model_dir = os.path.join(args.model_dir, f"so3_convnet_noise=0.5_seed={10000 + args.model_index}")
    else:
        args.model_dir = os.path.join(args.model_dir, f"so3_convnet_seed={10000 + args.model_index}")

    # make directory if it does not already exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # load config if requested, if config is None, then use hparams within model_dir
    if args.config is not None and not args.eval_only:
        with open(args.config, 'r') as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        
        # set the noise and seed parameters
        hparams['noise'] = 0.5 if args.train_with_noise else 0
        hparams['noise_seed'] = 10000 + args.model_index if args.train_with_noise else 0
        hparams['seed'] = 10000 + args.model_index if args.train_with_noise else 10000

        # save hparams as json file within model_dir
        with open(os.path.join(args.model_dir, 'hparams.json'), 'w+') as f:
            json.dump(hparams, f, indent=4)
        
    else:
        with open(os.path.join(args.model_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)

    if not args.eval_only:
        # launch training script
        if args.single_training_dataset:
            raise NotImplementedError()
        else:
            pretraining_hermes_divided_dataset(args.model_dir, pretrained_model_dir=args.pretrained_model_dir)

    # perform inference, on test data, with basic results
    predictions = hermes_inference_on_test_data(args.model_dir, model_name='lowest_valid_loss_model', emb_i=None, verbose=True, loading_bar=True)

    predictions = np.load(os.path.join(args.model_dir, 'test_data_results-lowest_valid_loss_model.npz'))

    logits = predictions['logits']
    best_indices = predictions['best_indices']
    targets = predictions['targets']
    res_ids = predictions['res_ids']

    accuracy = accuracy_score(targets, best_indices)

    print(f'Accuracy: {accuracy:.3f}')
    confusion_matrix(args, targets, best_indices, accuracy)


