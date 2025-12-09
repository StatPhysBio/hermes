'''

General classification training loop using CGNet-derived architectures

Assumes dataset giving data and labels are given
Takes as input a directory, which contains a json file with the model hyperparameters,
and in which the model will save checkpoints and all the good stuff

'''

import os, sys
import gzip, pickle
import json
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from typing import *

from hermes.utils.data import put_dict_on_device
from hermes.models import CGNet, SO3_ConvNet
from hermes.cg_coefficients import get_w3j_coefficients


def one_hot_encode(x, n_classes):
    return torch.eye(n_classes, device=x.device)[x.to(torch.long)]

def general_model_init(hparams, data_irreps):

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device), flush=True)

    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients(hparams['lmax'])
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False
    
    if hparams['model_type'] == 'cgnet':
        model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    elif hparams['model_type'] == 'so3_convnet':
        model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    else:
        raise NotImplementedError()
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params), flush=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    return model, loss_fn, device




def hermes_inference_on_test_data(model_dir: Union[List[str], str], # either path to a single model or to multiple models, assuming that all models have the same input
                                 emb_i: Optional[Union[int, str]] = -1, # type of embeddings to get out of HCNN. "None" returns dummy zeros.
                                 batch_size: int = 256,
                                 model_name: str = 'lowest_valid_loss_model',
                                 verbose: bool = False,
                                 loading_bar: bool = False
                                 ):
    '''
    Exists only for the purpose of testing a model right after pre-training
    '''
    if isinstance(model_dir, str):
        model_dir_list = [model_dir]
    else:
        model_dir_list = model_dir

    # get hparams from json for data purposes
    with open(os.path.join(model_dir_list[0], 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device), flush=True)

    ########## THE CODE BLOCK BELOW MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    # from protein_holography_pytorch.utils.data import load_data
    from .data import load_data
    output_filepath = os.path.join(model_dir_list[0], 'test_data_results-{}.npz'.format(model_name))
    datasets, data_irreps, _ = load_data(hparams, splits=['test'])
    dataset = datasets['test']

    if hparams['normalize_input']:
        normalize_input_at_runtime = True
    else:
        normalize_input_at_runtime = False
    
    if verbose: print('Done preprocessing.')
    sys.stdout.flush()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    if verbose: print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()

    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients()
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False

    ## ensemble models!
    predictions_trace = []
    for model_dir in model_dir_list:

        with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)
    
        # create model and load weights
        if hparams['model_type'] == 'cgnet':
            model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=normalize_input_at_runtime).to(device)
        elif hparams['model_type'] == 'so3_convnet':
            model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=normalize_input_at_runtime).to(device)
        else:
            raise NotImplementedError()
    
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt'), map_location=torch.device(device)))
    
        num_params = 0
        for param in model.parameters():
            num_params += torch.flatten(param.data).shape[0]
        if verbose: print('There are %d parameters' % (num_params), flush=True)

        predictions = model.predict(dataloader, device=device, verbose=verbose, loading_bar=loading_bar, emb_i = emb_i)

        predictions_trace.append(predictions)
    
    if len(predictions_trace) == 1:
        predictions = predictions_trace[0]
        if verbose: print('Accuracy: %.3f' % (accuracy_score(predictions['targets'], predictions['best_indices'])))
    else:
        assert len(predictions_trace) != 0, 'Something has gotten terribly wrong'
        assert np.allclose(predictions_trace[0]['targets'], predictions_trace[1]['targets']) # simple, non-exhaustive sanity check

        for pred in predictions_trace:
            assert np.all(pred['targets'] == predictions_trace[0]['targets'])
            assert np.all(pred['res_ids'] == predictions_trace[0]['res_ids'])

        predictions = {
            'logits': np.stack([pred['logits'] for pred in predictions_trace], axis=-1),
            'best_indices': np.stack([pred['best_indices'] for pred in predictions_trace], axis=-1),
            'targets': predictions_trace[0]['targets'],
            'res_ids': predictions_trace[0]['res_ids']
        }
        if np.all(np.array(['embeddings' in pred for pred in predictions_trace])):
            predictions['embeddings'] = np.stack([pred['embeddings'] for pred in predictions_trace], axis=-1)

    if output_filepath is None: # then just return
        return predictions
    
    elif output_filepath[-4:] == '.npz':
        np.savez_compressed(output_filepath,
                            logits=predictions['logits'],
                            best_indices=predictions['best_indices'],
                            targets=predictions['targets'],
                            res_ids=predictions['res_ids'])
        
        return predictions
    
    else:
        raise NotImplementedError('Invalid format for output_filepath')




def pretraining_hermes_divided_dataset(model_dir: str, pretrained_model_dir: Optional[str] = None, pretrained_model_name: str = 'lowest_valid_loss_model'):
    '''
    Assumes that directory 'model_dir' exists and contains json file with data and model hyperprameters 
    '''

    # get hparams from json
    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    if hparams['seed'] is not None:
        rng = torch.Generator().manual_seed(hparams['seed'])
    else:
        rng = torch.Generator() # random seed

    print('Loading data...', flush=True)
    
    ########## THE CODE BLOCK BELOW MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    ## get data and make dataloaders
    from .data import load_single_split_data
    train_dataset, data_irreps, norm_factor = load_single_split_data(hparams, 'training__0', get_norm_factor_if_training=True)
    valid_dataset, _, _ = load_single_split_data(hparams, 'validation')

    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True), drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, generator=rng, shuffle=False, drop_last=False)

    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########
    
    print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()

    # set norm factor in hparams, save new hparams
    hparams['model_hparams']['input_normalizing_constant'] = norm_factor
    with open(os.path.join(model_dir, 'hparams.json'), 'w+') as f:
        json.dump(hparams, f, indent=4)

    model, loss_fn, device = general_model_init(hparams, data_irreps)

    if pretrained_model_dir is not None:
        print('Loading pretrained model...', flush=True)
        model.load_state_dict(torch.load(os.path.join(pretrained_model_dir, f'{pretrained_model_name}.pt')))

    # setup learning algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    if hparams['lr_scheduler'] is None:
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    else:
        raise NotImplementedError()

    logfile = open(os.path.join(model_dir, 'log.txt'), 'w+')

    # training loop!
    num_validation_steps_per_epoch = 5
    divisor_for_validation_step = hparams['num_train_datasets'] // num_validation_steps_per_epoch
    train_loss_trace = []
    valid_loss_trace = []
    valid_acc_trace = []
    lowest_valid_loss = np.inf
    for epoch in range(hparams['n_epochs']):
        print('Epoch %d/%d\t\ttrain loss\t\tvalid loss\t\tvalid acc\t\ttime (s)' % (epoch+1, hparams['n_epochs']), flush=True)
        print('Epoch %d/%d\t\ttrain loss\t\tvalid loss\t\tvalid acc\t\ttime (s)' % (epoch+1, hparams['n_epochs']), file=logfile, flush=True)
        sys.stdout.flush()
        temp_train_loss_trace = []
        start_time = time.time()

        if epoch == 0:
            shuffled_train_dataset_indices = np.hstack([np.array([0]), np.random.permutation(hparams['num_train_datasets']-1)+1]) # put the zero index first, as that's the first dataset no matter what
        else:
            shuffled_train_dataset_indices = np.random.permutation(hparams['num_train_datasets'])

        for train_dataset_i in range(hparams['num_train_datasets']):

            # get new training dataset
            if train_dataset_i > 0 or epoch > 0:
                # print('Loading new training dataset...', flush=True)
                del train_dataset
                del train_dataloader
                train_dataset, _, _ = load_single_split_data(hparams, f'training__{shuffled_train_dataset_indices[train_dataset_i]}', get_norm_factor_if_training=False)
                train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True), drop_last=False)
                # print('Done!')
        
            for train_i, (X_train, X_train_vec, y_train, (rot, data_ids)) in enumerate(train_dataloader):
                X_train = put_dict_on_device(X_train, device)
                y_train = y_train.to(device)
                model.train()

                optimizer.zero_grad()
                
                y_train_hat = model(X_train)
                loss_train = loss_fn(y_train_hat, y_train)
                temp_train_loss_trace.append(loss_train.item())
                
                loss_train.backward()
                optimizer.step()


            if not (train_dataset_i % divisor_for_validation_step == 0 or train_dataset_i == hparams['num_train_datasets'] - 1):
                continue

            # record train and validation loss
            temp_valid_loss_trace = []
            y_valid_all = []
            y_valid_hat_all = []
            pseudoenergies_all = []
            for valid_i, (X_valid, X_valid_vec, y_valid, (rot, data_ids)) in enumerate(valid_dataloader):
                X_valid = put_dict_on_device(X_valid, device)
                y_valid = y_valid.to(device)
                model.eval()
                
                pseudoenergies = model(X_valid)
                loss_valid = loss_fn(pseudoenergies, y_valid)
                temp_valid_loss_trace.append(loss_valid.item())
                y_valid_all.append(y_valid.detach().cpu().numpy())
                y_valid_hat_all.append(np.argmax(pseudoenergies.detach().cpu().numpy(), axis=1))
                pseudoenergies_all.append(pseudoenergies.detach().cpu().numpy())

            y_valid_all = np.hstack(y_valid_all)
            y_valid_hat_all = np.hstack(y_valid_hat_all)
            pseudoenergies_all = np.vstack(pseudoenergies_all)
            
            curr_train_loss = np.mean(temp_train_loss_trace)
            curr_valid_loss = np.mean(temp_valid_loss_trace)

            curr_valid_acc = accuracy_score(y_valid_all, y_valid_hat_all)

            end_time = time.time()
            print('%d/%d:\t\t\t%.5f\t\t\t%.5f\t\t\t%.3f\t\t\t%.1f' % (train_dataset_i+1, hparams['num_train_datasets'], curr_train_loss, curr_valid_loss, curr_valid_acc, (end_time - start_time)), flush=True)
            print('%d/%d:\t\t\t%.5f\t\t\t%.5f\t\t\t%.3f\t\t\t%.1f' % (train_dataset_i+1, hparams['num_train_datasets'], curr_train_loss, curr_valid_loss, curr_valid_acc, (end_time - start_time)), file=logfile, flush=True)
            
            # update lr with scheduler
            if lr_scheduler is not None:
                lr_scheduler.step(curr_valid_loss)

            # record best model so far
            if curr_valid_loss < lowest_valid_loss:
                lowest_valid_loss = curr_valid_loss
                torch.save(model.state_dict(), os.path.join(model_dir, 'lowest_valid_loss_model.pt'))

            train_loss_trace.append(curr_train_loss)
            valid_loss_trace.append(curr_valid_loss)
            valid_acc_trace.append(curr_valid_acc)

            temp_train_loss_trace = []
            temp_valid_loss_trace = []
            start_time = time.time()

    ## save loss traces

    ## as plots
    import matplotlib.pyplot as plt

    iterations = np.arange(len(train_loss_trace))

    plt.figure(figsize=(10, 4))
    plt.plot(iterations, train_loss_trace, label='train')
    plt.plot(iterations, valid_loss_trace, label='valid')
    plt.ylabel('Cross-Entropy loss')
    plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'loss_trace.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(iterations, valid_acc_trace, label='valid')
    plt.ylabel('Accuracy')
    plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'acc_trace.png'))
    plt.close()
