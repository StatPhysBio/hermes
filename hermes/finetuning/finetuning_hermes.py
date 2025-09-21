

import os, sys
import json
import gzip, pickle

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
import time
from copy import deepcopy

from torch.utils.data import DataLoader

import argparse

from hermes.models import CGNet, SO3_ConvNet
from hermes.cg_coefficients import get_w3j_coefficients

from hermes.inference.inference_hermes import get_channels, get_data_irreps

from sklearn.metrics import classification_report, roc_auc_score
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

from hermes.utils.data import put_dict_on_device
from hermes.utils.protein_naming import ol_to_ind_size

from e3nn import o3
from torch.utils.data import Dataset
from typing import *


def hermes_model_init_for_finetuning(model_dir, finetuning_params):

    with open(f'{model_dir}/hparams.json', 'r') as f:
        hparams = json.load(f)

    data_irreps, ls_indices = get_data_irreps(hparams)

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device), flush=True)

    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients() # get_w3j_coefficients(hparams['lmax'])
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False
    
    # load model and pre-trained weights
    if hparams['model_type'] == 'cgnet':
        model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    elif hparams['model_type'] == 'so3_convnet':
        model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    else:
        raise NotImplementedError()
    
    if os.path.exists(os.path.join(model_dir, 'lowest_valid_loss_model.pt')):
        model.load_state_dict(torch.load(os.path.join(model_dir, 'lowest_valid_loss_model.pt'), map_location=device))
    else:
        print('-----------------------------------------------------------------', flush=True)
        print('WARNING: No pre-trained model found. Using random initialization.', flush=True)
        print('-----------------------------------------------------------------', flush=True)
    
    if finetuning_params['finetuning_depth'] == 'all':
        trainable_names = [name for name, _ in model.named_parameters()]
    elif finetuning_params['finetuning_depth'] == 'invariant_mlp':
        ## only let the SO(3)-invariant classification head be trainable
        trainable_names = ['fc_blocks.0.0.weight', 'fc_blocks.0.0.bias', 'output_head.weight', 'output_head.bias']
    else:
        raise ValueError('finetuning_depth must be either "all" or "invariant_mlp"')

    for name, param in model.named_parameters():
        if name in trainable_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params), flush=True)

    return hparams, data_irreps, model, device



def load_data(finetuning_params: Dict, hparams: Dict, data_irreps: o3.Irreps):

    if finetuning_params['finetune_with_noise']:
        zgrams_path = finetuning_params['zernikegrams_npz_gz_template'].format(model_version=finetuning_params['model_version'] + f"_seed={hparams['noise_seed']}")
    else:
        zgrams_path = finetuning_params['zernikegrams_npz_gz_template'].format(model_version=finetuning_params['model_version'])
    
    with gzip.open(zgrams_path, 'rb') as f:
        zgrams = np.load(f, allow_pickle=True)
    
    # print(set(list(zgrams['pdbid'])))
    
    # convert parallel arrays to dictionary (pdbid, chainid, resnum) -> zernikegram
    # convert all pdbids to lowercase for assured compatibility
    all_res_ids_to_zgrams = {(pdbid.lower(), chainid, resnum): zernikegram for pdbid, chainid, resnum, zernikegram in zip(zgrams['pdbid'], zgrams['chainid'], zgrams['resnum'], zgrams['zernikegram'])}
    
    dataloaders = {}
    for split in ['train', 'valid', 'test']:
        targets = pd.read_csv(finetuning_params['targets_csv_template'].format(split=split))

        if split == 'train':
            mean_train_score, std_train_score = targets['score'].mean(), targets['score'].std()

        dataset = FinetuningZernikegramsDataset(all_res_ids_to_zgrams, data_irreps, targets)

        if split == 'train':
            batch_size = finetuning_params['batch_size']
        else:
            batch_size = 512

        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))

    return dataloaders, mean_train_score, std_train_score
    


class FinetuningZernikegramsDataset(Dataset):
    def __init__(self, all_res_ids_to_zgrams: Dict, irreps: o3.Irreps, targets: pd.DataFrame):

        self.targets = targets

        # construct resnums from targets variants
        self.targets['resnum'] = self.targets['variant'].apply(lambda x: int(x[1:-1]))

        # turn all pidbids to lowercase for assured compatibility
        self.targets['pdbid'] = self.targets['pdbid'].str.lower()

        # print(set(list(self.targets['pdbid'])))

        # construct unique (pdbid, chainid, resnum) from targets
        res_ids_set = set(list(zip(self.targets['pdbid'], self.targets['chainid'], self.targets['resnum'])))

        # construct internal dictionary of (pdbid, chainid, resnum) -> zgram, only with res_ids that are in res_ids_set
        num_fail = 0
        self.res_id_to_zgram = {}
        for res_id in res_ids_set:
            if res_id in all_res_ids_to_zgrams: # NOTE: this should not happen
                self.res_id_to_zgram[res_id] = all_res_ids_to_zgrams[res_id]
            else:
                num_fail += 1

        if num_fail > 0:
            print(f'WARNING: Failed to find {num_fail}/{len(res_ids_set)} res_ids in all_res_ids_to_zgrams.', flush=True)
        
            # remove rows from self.targets that do not have a corresponding zgram
            length_before = len(self.targets)
            self.targets = self.targets[self.targets.apply(lambda row: (row['pdbid'], row['chainid'], row['resnum']) in self.res_id_to_zgram, axis=1)]
            print(f'Removed {length_before - len(self.targets)}/{length_before} rows from self.targets.', flush=True)
        else:
            print(f'All {len(res_ids_set)} res_ids found in zgrams_parallel_arrays.', flush=True)
        
        self.ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps.ls])
        self.unique_ls = sorted(list(set(irreps.ls)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        row = self.targets.iloc[idx]

        res_id = (row['pdbid'], row['chainid'], row['resnum'])
        zgram = self.res_id_to_zgram[res_id]
        zgram_fiber = {}
        for l in self.unique_ls:
            zgram_fiber[l] = torch.tensor(zgram[self.ls_indices == l]).view(-1, 2*l+1).float()

        aa_idx_wt = ol_to_ind_size[row['variant'][0]]
        aa_idx_mt = ol_to_ind_size[row['variant'][-1]]

        score = row['score']

        return zgram_fiber, aa_idx_wt, aa_idx_mt, score, res_id


def loss_function(pred_logits_B20, target_aa_wt_idx_B, target_aa_mt_idx_B, target_score_B, device,
                  alpha_cls = 0.5):

    # compute classification loss (need it as regularizer, to not forget the original task)
    loss_cls = torch.nn.CrossEntropyLoss()(pred_logits_B20, target_aa_wt_idx_B)

    # compute regression loss
    B = pred_logits_B20.shape[0]
    pred_score_B = pred_logits_B20[torch.arange(B), target_aa_wt_idx_B] - pred_logits_B20[torch.arange(B), target_aa_mt_idx_B]

    # loss_reg = torch.nn.MSELoss()(pred_score_B, target_score_B)
    loss_reg = torch.nn.HuberLoss(delta=1.0)(pred_score_B, target_score_B)

    return alpha_cls * loss_cls + (1 - alpha_cls) * loss_reg, loss_cls, loss_reg


def training_epoch(finetuning_params, model, optimizer, device, dataloader):
    model.train()

    loss_trace, loss_cls_trace, loss_reg_trace = [], [], []
    for zgram, aa_idx_wt, aa_idx_mt, score, res_id in tqdm(dataloader, total=len(dataloader)):
        zgram = put_dict_on_device(zgram, device)
        aa_idx_wt = aa_idx_wt.long().to(device)
        aa_idx_mt = aa_idx_mt.long().to(device)
        score = score.float().to(device)

        optimizer.zero_grad()

        pred_logits_B20 = model(zgram)

        loss, loss_cls, loss_reg = loss_function(pred_logits_B20, aa_idx_wt, aa_idx_mt, score, device, alpha_cls=finetuning_params['alpha_cls'])

        loss.backward()
        optimizer.step()

        loss_trace.append(loss.item())
        loss_cls_trace.append(loss_cls.item())
        loss_reg_trace.append(loss_reg.item())
    
    last_perc_to_keep = 0.1 # just so that it's a better estimate of the loss on the whole training data by the end of the epoch
    loss_trace = loss_trace[int(len(loss_trace) * last_perc_to_keep):]
    loss_cls_trace = loss_cls_trace[int(len(loss_cls_trace) * last_perc_to_keep):]
    loss_reg_trace = loss_reg_trace[int(len(loss_reg_trace) * last_perc_to_keep):]
    
    return np.mean(loss_trace), np.mean(loss_cls_trace), np.mean(loss_reg_trace)


def validation_epoch(finetuning_params, model, device, dataloader, return_logits_stats=False):
    model.eval()

    loss_trace, loss_cls_trace, loss_reg_trace = [], [], []
    logsumexp_trace = []
    for zgram, aa_idx_wt, aa_idx_mt, score, res_id in tqdm(dataloader, total=len(dataloader)):
        zgram = put_dict_on_device(zgram, device)
        aa_idx_wt = aa_idx_wt.long().to(device)
        aa_idx_mt = aa_idx_mt.long().to(device)
        score = score.float().to(device)

        pred_logits_B20 = model(zgram)

        if return_logits_stats:
            from scipy.special import logsumexp
            logsumexp_trace.append(logsumexp(pred_logits_B20.detach().cpu().numpy(), axis=1))

        loss, loss_cls, loss_reg = loss_function(pred_logits_B20, aa_idx_wt, aa_idx_mt, score, device, alpha_cls=finetuning_params['alpha_cls'])

        loss_trace.append(loss.item())
        loss_cls_trace.append(loss_cls.item())
        loss_reg_trace.append(loss_reg.item())
        
    if return_logits_stats:
        return np.mean(loss_trace), np.mean(loss_cls_trace), np.mean(loss_reg_trace), (np.mean(np.hstack(logsumexp_trace)), np.std(np.hstack(logsumexp_trace)))
    
    return np.mean(loss_trace), np.mean(loss_cls_trace), np.mean(loss_reg_trace)

def testing(model, device, dataloader, output_model_dir):
    model.eval()

    all_pdbids = []
    all_scores = []
    all_pred_scores = []
    for zgram, aa_idx_wt, aa_idx_mt, score, res_id in tqdm(dataloader, total=len(dataloader)):
        zgram = put_dict_on_device(zgram, device)
        aa_idx_wt = aa_idx_wt.long().to(device)
        aa_idx_mt = aa_idx_mt.long().to(device)
        score = score.float().to(device)

        pred_logits_B20 = model(zgram)

        B = pred_logits_B20.shape[0]
        pred_score_B = pred_logits_B20[torch.arange(B), aa_idx_wt] - pred_logits_B20[torch.arange(B), aa_idx_mt]

        all_pdbids.extend(res_id[0])
        all_scores.extend(score.detach().cpu().numpy())
        all_pred_scores.extend(pred_score_B.detach().cpu().numpy())


    all_pdbids = np.hstack(all_pdbids)
    all_scores = np.hstack(all_scores)
    all_pred_scores = np.hstack(all_pred_scores)

    pr_results, sr_results = [], []
    # add the overall pearson and spearman correlation
    pr_results.append(('overall', pearsonr(all_scores, all_pred_scores)[0]))
    sr_results.append(('overall', spearmanr(all_scores, all_pred_scores)[0]))
    # group by pdbid and compute pearson correlation
    for pdbid in np.unique(all_pdbids):
        mask = all_pdbids == pdbid
        try:
            pr_results.append((pdbid, pearsonr(all_scores[mask], all_pred_scores[mask])[0]))
            sr_results.append((pdbid, spearmanr(all_scores[mask], all_pred_scores[mask])[0]))
        except:
            print(f'Failed to compute correlation for {pdbid}.', flush=True)
            pr_results.append((pdbid, np.nan))
            sr_results.append((pdbid, np.nan))

    return pr_results, sr_results


def finetune_single_model(input_model_dir: str,
                          output_model_dir: str,
                          finetuning_params: Dict):
    
    ## evaluate on test set, reporting pearsonr per pdbid
    
    hparams, data_irreps, model, device = hermes_model_init_for_finetuning(input_model_dir, finetuning_params)

    ## setup output directory
    os.makedirs(output_model_dir, exist_ok=True)

    # dump hparams of original model
    with open(f'{output_model_dir}/hparams.json', 'w') as f:
        json.dump(hparams, f, indent=4)
    
    # dump finetuning hparams
    with open(f'{output_model_dir}/finetuning_params.json', 'w') as f:
        json.dump(finetuning_params, f, indent=4)
    
    ## prepare data (gets repeated for every model but it does not take that long)
    dataloaders, mean_train_score, std_train_score = load_data(finetuning_params, hparams, data_irreps)

    ## score pre-trained model on the test set
    pr_results, sr_results = testing(model, device, dataloaders['test'], output_model_dir)
    print()
    print('pdbid\tpearsonr\tspearmanr\n')
    for pdbid, pr, sr in zip([res[0] for res in pr_results],
                                [res[1] for res in pr_results],
                                [res[1] for res in sr_results]):
        print(f'{pdbid}\t{pr:.3f}\t{sr:.3f}\n')
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=finetuning_params['lr'])

    if finetuning_params['lr_scheduler'] is None:
        lr_scheduler = None
    elif finetuning_params['lr_scheduler'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    else:
        raise NotImplementedError()

    logfile = open(os.path.join(output_model_dir, 'log.txt'), 'w+')

    out_text = f'epoch\ttrain loss\tvalid loss\ttrain cls loss\tvalid cls loss\ttrain regr loss\tvalid regr loss\ttime (s)'
    print(out_text, flush=True)
    print(out_text, file=logfile, flush=True)

    start = time.time()
    train_loss, train_cls_loss, train_regr_loss = np.nan, np.nan, np.nan
    valid_loss, valid_cls_loss, valid_regr_loss, (orig_mean, orig_std) = validation_epoch(finetuning_params, model, device, dataloaders['valid'], return_logits_stats=True) # make this through the validation data just so it takes less time, it doesn't really matgter it's just a rescaling in the ballpark

    ## rescale logits to be in the range of the scores
    model.output_head.weight.data = model.output_head.weight.data * std_train_score / orig_std
    model.output_head.bias.data = model.output_head.bias.data * std_train_score + (mean_train_score - orig_mean * std_train_score / orig_std)

    end = time.time()

    out_text = f'0\t{train_loss:.4f}\t\t{valid_loss:.4f}\t\t{train_cls_loss:.4f}\t\t{valid_cls_loss:.4f}\t\t{train_regr_loss:.4f}\t\t{valid_regr_loss:.4f}\t\t{end-start:.2f}'
    print(out_text, flush=True)
    print(out_text, file=logfile, flush=True)


    # save original model
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), os.path.join(output_model_dir, 'lowest_valid_loss_model.pt'))

    ## score pre-trained model on the test set, after rescaling
    pr_results, sr_results = testing(model, device, dataloaders['test'], output_model_dir)
    print()
    print('pdbid\tpearsonr\tspearmanr\n')
    for pdbid, pr, sr in zip([res[0] for res in pr_results],
                                [res[1] for res in pr_results],
                                [res[1] for res in sr_results]):
        print(f'{pdbid}\t{pr:.3f}\t{sr:.3f}\n')
    print()

    for epoch in range(finetuning_params['n_epochs']):
        train_loss, train_cls_loss, train_regr_loss = training_epoch(finetuning_params, model, optimizer, device, dataloaders['train'])
        valid_loss, valid_cls_loss, valid_regr_loss = validation_epoch(finetuning_params, model, device, dataloaders['valid'])

        out_text = f'{epoch+1}\t\t{train_loss:.4f}\t\t{valid_loss:.4f}\t\t{train_cls_loss:.4f}\t\t{valid_cls_loss:.4f}\t\t{train_regr_loss:.4f}\t\t{valid_regr_loss:.4f}\t\t{end-start:.2f}'
        print(out_text, flush=True)
        print(out_text, file=logfile, flush=True)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_model_dir, 'lowest_valid_loss_model.pt'))

        if lr_scheduler is not None:
            lr_scheduler.step(valid_loss)
    
    logfile.close()

    ## evaluate on test set, reporting pearsonr and spearmanr per pdbid, on best model
    model.load_state_dict(torch.load(os.path.join(output_model_dir, 'lowest_valid_loss_model.pt'), map_location=device))
    pr_results, sr_results = testing(model, device, dataloaders['test'], output_model_dir)

    with open(os.path.join(output_model_dir, 'test_results.txt'), 'w+') as f:
        f.write('pdbid\tpearsonr\tspearmanr\n')
        for pdbid, pr, sr in zip([res[0] for res in pr_results],
                                 [res[1] for res in pr_results],
                                 [res[1] for res in sr_results]):
            f.write(f'{pdbid}\t{pr:.3f}\t{sr:.3f}\n')





