
import sys, os

import numpy as np
import torch
from e3nn import o3
from protein_holography_web import nn

from protein_holography_web.utils.data import put_dict_on_device
from tqdm import tqdm
from copy import deepcopy
from scipy.special import softmax

from torch import Tensor
from typing import *


class SO3_ConvNet(torch.nn.Module):
    '''
    CGNet-like model, but without the invariant skip connections
    '''

    def load_hparams(self, hparams: Dict):

        # this is just a global normalization factor applied to all data (e.g. the mean sqrt power of the training data)
        # useful to store it inside the model for doing inference on new test data
        # we usually don't *need* to do this for supervised models, but I noticed it at least speeds up convergence. so perhaps it is useful, but requires more testing
        self.input_normalizing_constant = torch.tensor(hparams['input_normalizing_constant'], requires_grad=False) if hparams['input_normalizing_constant'] is not None else None

        ## hyperparams of the CG blocks
        self.n_cg_blocks = hparams['n_cg_blocks']

        self.do_initial_linear_projection = hparams['do_initial_linear_projection'] # required to be true when the first value of self.ch_nonlin_rule_list is 'elementwise'
        self.ch_initial_linear_projection = hparams['ch_initial_linear_projection'] #   because the 'elementwise' Tensor Product requires an equal numver of channels per \ell

        # these two control the dimensionality of the CG blocks, in terms of maximum spherical degree \ell, and number of channels (equal for all \ell)
        self.lmax_list = hparams['lmax_list']
        self.ch_size_list = hparams['ch_size_list']

        # these two govern the Tensor Product rules in each block
        self.ls_nonlin_rule_list = hparams['ls_nonlin_rule_list']
        self.ch_nonlin_rule_list = hparams['ch_nonlin_rule_list']

        assert self.n_cg_blocks == len(self.lmax_list)
        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)

        self.use_additive_skip_connections = hparams['use_additive_skip_connections'] # zero-padded on one side if self.ch_size_list[i] > self.ch_size_list[i-1]

        self.weights_initializer = hparams['weights_initializer'] # do not bother with this

        # batch norm, if requested, is applied *at the start* of each CG block
        self.use_batch_norm = hparams['use_batch_norm']

        # hyperparams of the norm layer, outside of batch norm
        # I am listing the default values
        self.norm_type = hparams['norm_type'] # signal [None, layer, signal, layer_and_signal, instance, magnitudes, layer_nonlin], focus on layer and signal
        self.normalization = hparams['normalization'] # component, keep this way [norm, component]
        self.norm_balanced = hparams['norm_balanced'] # False, keep this way [True, False]
        self.norm_affine = hparams['norm_affine'] # per_l [True, False] -> for layer norm ; [unique, per_l, per_feature] -> for signal norm ; Tuple of both kinds --> for layer_and_signal norm
        self.norm_nonlinearity = hparams['norm_nonlinearity'] # None
        self.norm_location = hparams['norm_location'] # between [first, between, last], where to put the norm layer relative to linear and nonlinear layers

        self.linearity_first = hparams['linearity_first'] # whether to apply the linear transformation first (or the nonlinearity first), keep False
        self.filter_symmetric = hparams['filter_symmetric'] # keep True always, no reason to do otherwise. Does not change anything for 'efficient' ls_nonlin_rule, and reduces unnecessary computation for 'full' ls_nonlin_rule

        ## hyperparams of the fully-connected layers on the invariant (\ell=0) output of the CG blocks
        self.n_fc_blocks = hparams['n_fc_blocks']
        self.fc_h_dim = hparams['fc_h_dim']
        self.fc_nonlin = hparams['fc_nonlin']
        self.dropout_rate = hparams['dropout_rate']

        # self-evident
        self.output_dim = hparams['output_dim']



    def __init__(self,
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 hparams: Dict,
                 normalize_input_at_runtime: bool = False,
                 verbose: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.load_hparams(hparams)
        self.normalize_input_at_runtime = normalize_input_at_runtime

        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.lmax_list is None or self.n_cg_blocks == len(self.lmax_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)

        if self.do_initial_linear_projection:
            if verbose: print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(self.irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.SO3_linearity(self.irreps_in, initial_irreps)
            if verbose: print(initial_irreps.dim, initial_irreps)
        else:
            if verbose: print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = self.irreps_in


        # equivariant, cg blocks
        prev_irreps = initial_irreps
        cg_blocks = []
        for i in range(self.n_cg_blocks):
            irreps_hidden = (self.ch_size_list[i]*o3.Irreps.spherical_harmonics(self.lmax_list[i], 1)).sort().irreps.simplify()
            cg_blocks.append(nn.CGBlock(prev_irreps,
                                                irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=self.linearity_first,
                                                filter_symmetric=self.filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=self.ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=self.ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=self.norm_type, # None, layer, signal
                                                normalization=self.normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=self.norm_balanced,
                                                norm_affine=self.norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=self.norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=self.norm_location, # first, between, last
                                                weights_initializer=self.weights_initializer,
                                                init_scale=1.0))

            prev_irreps = cg_blocks[-1].irreps_out
            if verbose: print(prev_irreps.dim, prev_irreps)

        self.cg_blocks = torch.nn.ModuleList(cg_blocks)

        invariants_dim = [mul for (mul, _) in prev_irreps][0] # number of channels for l = 0
        self.invariants_dim = invariants_dim


        # invariant, fully connected blocks
        prev_dim = invariants_dim
        fc_blocks = []
        for _ in range(self.n_fc_blocks):
            block = []
            block.append(torch.nn.Linear(prev_dim, self.fc_h_dim))
            block.append(eval(nn.NONLIN_TO_ACTIVATION_MODULES[self.fc_nonlin]))
            if self.dropout_rate > 0.0:
                block.append(torch.nn.Dropout(self.dropout_rate))

            fc_blocks.append(torch.nn.Sequential(*block))
            prev_dim = self.fc_h_dim

        if len(fc_blocks) > 0:
            self.fc_blocks = torch.nn.ModuleList(fc_blocks)
        else:
            self.fc_blocks = None


        # output head
        self.output_head = torch.nn.Linear(prev_dim, self.output_dim)

    
    def forward(self, x: Dict[int, Tensor]) -> Tensor:

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants = h[0].squeeze(-1)


        # invariant, fully connected blocks
        h = invariants
        if self.fc_blocks is not None:
            for block in self.fc_blocks:
                h = block(h)
                # h += block(h) # skip connections

        # output head
        out = self.output_head(h)

        return out

    def predict(self,
                dataloader: torch.utils.data.DataLoader,
                emb_i: int = -1,
                device: str = 'cpu',
                verbose: bool = False,
                loading_bar: bool = False) -> Dict:

        if loading_bar:
            loading_bar = tqdm
        else:
            loading_bar = lambda x: x

        if verbose: print('Making predictions on %s.' % device)

        self.eval()
        
        # inference loop!
        embeddings_all = []
        y_hat_all_logits = []
        y_hat_all_index = []
        y_all = []
        res_ids_all = []
        for i, (X, X_vec, y, (rot, res_ids)) in loading_bar(enumerate(dataloader)):
            X = put_dict_on_device(X, device)
            y = y.to(device)
            self.eval()
            
            if emb_i is not None:
                X_copy = deepcopy(X)
                embeddings = self.get_inv_embedding(X_copy, emb_i=emb_i)
            else:
                embeddings = torch.zeros(10)
            
            y_hat = self(X)

            if emb_i == -1:
                # little sanity check
                assert np.allclose(self.output_head(embeddings).detach().cpu().numpy(), y_hat.detach().cpu().numpy())

            embeddings_all.append(embeddings.detach().cpu().numpy())
            y_hat_all_logits.append(y_hat.detach().cpu().numpy())
            y_hat_all_index.append(np.argmax(y_hat.detach().cpu().numpy(), axis=1))
            y_all.append(y.detach().cpu().numpy())
            res_ids_all.append(res_ids)

        embeddings_all = np.vstack(embeddings_all)
        y_hat_all_logits = np.vstack(y_hat_all_logits)
        y_hat_all_index = np.hstack(y_hat_all_index)
        y_all = np.hstack(y_all)
        res_ids_all = np.hstack(res_ids_all)
    
        return {
            'embeddings': embeddings_all,
            'logits': y_hat_all_logits,
            'probabilities': softmax(y_hat_all_logits.astype(np.float64), axis=-1),
            'best_indices': y_hat_all_index,
            'targets': y_all,
            'res_ids': res_ids_all
        }

    def get_inv_embedding(self, x: Dict[int, Tensor], emb_i: Union[int, str] = -1) -> Tensor:
        '''
        Gets invariant embedding from the FC blocks (backwards, must be negative), or from the input to the FC blocks
        '''
        assert emb_i in ['cg_output', 'all'] or emb_i in [-i for i in range(1, self.n_fc_blocks + 1)]
        self.eval()

        all_output = []

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants = h[0].squeeze(-1)


        if emb_i == 'cg_output':
            return invariants
        elif emb_i == 'all':
            all_output.append(invariants)
        
        h = invariants
        if self.fc_blocks is not None:
            for n, block in enumerate(self.fc_blocks):
                h = block(h)
                if emb_i == 'all':
                    all_output.append(h)
                elif n == len(self.fc_blocks) + emb_i:
                    return h
        
        # NB: only runs if emb_i == 'all'
        return all_output
    


class SO3_ConvNetPlusEmbeddings(SO3_ConvNet):

    def load_embeddings_hparams(self, embedding_hparams: Dict):
        self.embedding_dim = embedding_hparams['embedding_dim']
        self.attn_latent_dim = embedding_hparams['attn_latent_dim']
        self.num_hidden_layers_for_embeddings = embedding_hparams['num_hidden_layers_for_embeddings']
        self.embedding_hidden_dim = embedding_hparams['embedding_hidden_dim']
        self.alphabet_size = 20
    

    def __init__(self,
                 embedding_hparams: Dict,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.load_embeddings_hparams(embedding_hparams)

        self.proj_dim = self.fc_h_dim

        layers = []
        prev_dim = self.embedding_dim
        for _ in range(self.num_hidden_layers_for_embeddings):
            layers.append(torch.nn.Linear(prev_dim, self.embedding_hidden_dim))
            layers.append(torch.nn.LayerNorm(self.embedding_hidden_dim))
            layers.append(torch.nn.LeakyReLU())
            prev_dim = self.embedding_hidden_dim
        layers.append(torch.nn.Linear(prev_dim, self.proj_dim))
        self.embedding_projector = torch.nn.Sequential(*layers)

        self.embedding_output_head = torch.nn.Linear(self.proj_dim, self.alphabet_size)

        self.invariants_attn_proj_list = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(self.proj_dim, self.attn_latent_dim*2),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(self.attn_latent_dim*2, self.attn_latent_dim))
                                for _ in range(self.alphabet_size)])
        
        self.embedding_attn_proj_list = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(self.proj_dim, self.attn_latent_dim*2),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(self.attn_latent_dim*2, self.attn_latent_dim))
                                for _ in range(self.alphabet_size)])

        # self.invariants_attn_proj = torch.nn.Sequential(torch.nn.Linear(self.proj_dim, self.attn_latent_dim*2),
        #                                                 torch.nn.LeakyReLU(),
        #                                                 torch.nn.Linear(self.attn_latent_dim*2, self.attn_latent_dim))
        
        # self.embedding_attn_proj = torch.nn.Sequential(torch.nn.Linear(self.proj_dim, self.attn_latent_dim*2),
        #                                                 torch.nn.LeakyReLU(),
        #                                                 torch.nn.Linear(self.attn_latent_dim*2, self.attn_latent_dim))

    
    def forward(self, zgram, emb, return_individual_predictions=False):
        ## override forward method

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in zgram:
                zgram[l] = zgram[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(zgram)
        else:
            h = zgram
        
        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants_BF = h[0].squeeze(-1)
        emb_projected_BF = self.embedding_projector(emb)


        # invariant, fully connected blocks
        h = invariants_BF
        if self.fc_blocks is not None:
            for block in self.fc_blocks:
                h = block(h)
                # h += block(h) # skip connections

        # output head
        hcnn_out_BA = self.output_head(h)

        emb_out_BA = self.embedding_output_head(emb_projected_BF)

        alpha_BA = []
        for i in range(self.alphabet_size):
            invariants_attn_BL = self.invariants_attn_proj_list[i](invariants_BF)
            emb_attn_BL = self.embedding_attn_proj_list[i](emb_projected_BF)
            alpha_BA.append(torch.sigmoid(torch.einsum('bl,bl->b', invariants_attn_BL, emb_attn_BL) / torch.sqrt(torch.tensor(self.attn_latent_dim).float())))
        alpha_BA = torch.stack(alpha_BA, dim=1)

        # invariants_attn_BL = self.invariants_attn_proj(invariants_BF)
        # emb_attn_BL = self.embedding_attn_proj(emb_projected_BF)
        # alpha_B = torch.sigmoid(torch.einsum('bl,bl->b', invariants_attn_BL, emb_attn_BL) / torch.sqrt(torch.tensor(self.attn_latent_dim).float()))
        # alpha_BA = alpha_B[:, None]

        out_BA = alpha_BA * emb_out_BA + (1 - alpha_BA) * hcnn_out_BA

        if return_individual_predictions:
            return out_BA, hcnn_out_BA, emb_out_BA
        else:
            return out_BA
    
    
    def predict(self,
                dataloader: torch.utils.data.DataLoader, # its batches must be of the form (zgram, zgram_vec, input_emb, y, (rot, res_ids))
                emb_i: int = -1,
                device: str = 'cpu',
                verbose: bool = False,
                loading_bar: bool = False) -> Dict:
        ## override predict method

        if loading_bar:
            loading_bar = tqdm
        else:
            loading_bar = lambda x: x

        if verbose: print('Making predictions on %s.' % device)

        self.eval()
        
        # inference loop!
        embeddings_all = [] # NOTE: due to legacy reasons, these "embeddings" are different, as they ae those of HCNN, not of the input
        y_hat_all_logits = []
        y_hat_all_logits_hcnn = []
        y_hat_all_logits_emb = []
        y_hat_all_index = []
        y_all = []
        res_ids_all = []
        for i, (zgram, zgram_vec, input_emb, y, (rot, res_ids)) in loading_bar(enumerate(dataloader)):
            zgram = put_dict_on_device(zgram, device)
            input_emb = input_emb.to(device)
            y = y.to(device)
            self.eval()
            
            if emb_i is not None:
                zgram_copy = deepcopy(zgram)
                embeddings = self.get_inv_embedding(zgram_copy, emb_i=emb_i)
            else:
                embeddings = torch.zeros(10)
            
            y_hat, y_hat_hcnn, y_hat_emb = self(zgram, input_emb, return_individual_predictions=True)

            ## this is now false
            # if emb_i == -1:
            #     # little sanity check
            #     assert np.allclose(self.output_head(embeddings).detach().cpu().numpy(), y_hat.detach().cpu().numpy())

            embeddings_all.append(embeddings.detach().cpu().numpy())
            y_hat_all_logits.append(y_hat.detach().cpu().numpy())
            y_hat_all_logits_hcnn.append(y_hat_hcnn.detach().cpu().numpy())
            y_hat_all_logits_emb.append(y_hat_emb.detach().cpu().numpy())
            y_hat_all_index.append(np.argmax(y_hat.detach().cpu().numpy(), axis=1))
            y_all.append(y.detach().cpu().numpy())
            res_ids_all.append(res_ids)

        embeddings_all = np.vstack(embeddings_all)
        y_hat_all_logits = np.vstack(y_hat_all_logits)
        y_hat_all_logits_hcnn = np.vstack(y_hat_all_logits_hcnn)
        y_hat_all_logits_emb = np.vstack(y_hat_all_logits_emb)
        y_hat_all_index = np.hstack(y_hat_all_index)
        y_all = np.hstack(y_all)
        res_ids_all = np.hstack(res_ids_all)
    
        return {
            'embeddings': embeddings_all, # NOTE: due to legacy reasons, these "embeddings" are different, as they ae those of HCNN, not of the input
            'logits': y_hat_all_logits,
            'logits_hcnn': y_hat_all_logits_hcnn,
            'logits_emb': y_hat_all_logits_emb,
            'probabilities': softmax(y_hat_all_logits.astype(np.float64), axis=-1),
            'best_indices': y_hat_all_index,
            'targets': y_all,
            'res_ids': res_ids_all
        }


# class SO3_ConvNetPlusEmbeddings(torch.nn.Module):
#     def __init__(self,
#                  hcnn_model: SO3_ConvNet,
#                  emb_dim: int = -1, # placeholder value to make it kwarg
#                  attn_latent_dim: int = -1, # placeholder value to make it kwarg
#                  num_hidden_layers_for_embeddings: int = -1, # placeholder value to make it kwarg
#                  **kwargs):
#         super().__init__()
#         self.hcnn_model = hcnn_model
#         self.emb_dim = emb_dim
#         self.attn_latent_dim = attn_latent_dim


#         # self.proj_dim = self.hcnn_model.invariants_dim

#         # self.embedding_projector = torch.nn.Sequential(
#         #     torch.nn.Linear(self.emb_dim, self.proj_dim),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(self.proj_dim, self.proj_dim),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(self.proj_dim, self.proj_dim)
#         # )

#         # self.invariants_attn_proj = torch.nn.Linear(self.proj_dim, self.attn_latent_dim)
#         # self.embedding_attn_proj = torch.nn.Linear(self.proj_dim, self.attn_latent_dim)


#         self.proj_dim = self.hcnn_model.fc_h_dim

#         layers = [torch.nn.Linear(self.emb_dim, self.proj_dim)]
#         for _ in range(num_hidden_layers_for_embeddings):
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Linear(self.proj_dim, self.proj_dim))
#         self.embedding_projector = torch.nn.Sequential(*layers)

#         self.embedding_output_head = torch.nn.Linear(self.proj_dim, 20)

#         self.invariants_attn_proj = torch.nn.Linear(self.proj_dim, self.attn_latent_dim)
#         self.embedding_attn_proj = torch.nn.Linear(self.proj_dim, self.attn_latent_dim)
        

    
#     def forward(self, zgram, emb, return_individual_predictions=False):
#         ## copying the forward method from SO3_ConvNet, adding the embeddings bit
#         ## easier for me to do it this way than to subclass SO3_ConvNet and override the forward method,
#         ## just because i am not sure how to load partial weights 

#         # normalize input data if desired
#         if self.hcnn_model.normalize_input_at_runtime and self.hcnn_model.input_normalizing_constant is not None:
#             for l in zgram:
#                 zgram[l] = zgram[l] / self.hcnn_model.input_normalizing_constant

#         if self.hcnn_model.do_initial_linear_projection:
#             h = self.hcnn_model.initial_linear_projection(zgram)
#         else:
#             h = zgram
        
#         # equivariant, cg blocks
#         for i, block in enumerate(self.hcnn_model.cg_blocks):
#             h_temp = block(h)
#             if self.hcnn_model.use_additive_skip_connections:
#                 for l in h:
#                     if l in h_temp:
#                         if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
#                             h_temp[l] += h[l]
#                         elif h[l].shape[1] > h_temp[l].shape[1]:
#                             h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
#                         else: # h[l].shape[1] < h_temp[l].shape[1]
#                             h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
#             h = h_temp
        
#         invariants_BF = h[0].squeeze(-1)
#         emb_projected_BF = self.embedding_projector(emb)


#         # # invariants_attn_BL = self.invariants_attn_proj(invariants_BF)
#         # # emb_attn_BL = self.embedding_attn_proj(emb_projected_BF)
#         # # alpha_B = torch.sigmoid(torch.einsum('bl,bl->b', invariants_attn_BL, emb_attn_BL) / torch.sqrt(torch.tensor(self.attn_latent_dim).float()))
#         # # print(alpha_B[:5])
#         # alpha_B = torch.full((emb_projected_BF.shape[0],), 0.5, device=emb_projected_BF.device)

#         # # print('esm: %.3f, hcnn: %.3f' % (torch.mean(torch.abs(emb_projected_BF)).item(), torch.mean(torch.abs(invariants_BF)).item()))

#         # merged_representation_BF = alpha_B[:, None] * emb_projected_BF + (1 - alpha_B[:, None]) * invariants_BF

#         # # invariant, fully connected blocks
#         # h = merged_representation_BF
#         # if self.hcnn_model.fc_blocks is not None:
#         #     for block in self.hcnn_model.fc_blocks:
#         #         h = block(h)
#         #         # h += block(h) # skip connections

#         # # output head
#         # out = self.hcnn_model.output_head(h)

#         # if return_individual_predictions:
#         #     return out, None, None
#         # else:
#         #     return out


#         # invariant, fully connected blocks
#         h = invariants_BF
#         if self.hcnn_model.fc_blocks is not None:
#             for block in self.hcnn_model.fc_blocks:
#                 h = block(h)
#                 # h += block(h) # skip connections

#         # output head
#         hcnn_out = self.hcnn_model.output_head(h)

#         emb_out = self.embedding_output_head(emb_projected_BF)

#         invariants_attn_BL = self.invariants_attn_proj(invariants_BF)
#         emb_attn_BL = self.embedding_attn_proj(emb_projected_BF)
#         alpha_B = torch.sigmoid(torch.einsum('bl,bl->b', invariants_attn_BL, emb_attn_BL) / torch.sqrt(torch.tensor(self.attn_latent_dim).float()))
#         # print(alpha_B[:5])
#         # alpha_B = torch.full((emb_projected_BF.shape[0],), 0.5, device=emb_projected_BF.device)

#         out = alpha_B[:, None] * emb_out + (1 - alpha_B[:, None]) * hcnn_out

#         if return_individual_predictions:
#             return out, hcnn_out, emb_out
#         else:
#             return out
    
#     def predict(self,
#                 dataloader: torch.utils.data.DataLoader,
#                 emb_i: int = -1,
#                 device: str = 'cpu',
#                 verbose: bool = False,
#                 loading_bar: bool = False) -> Dict:

#         if loading_bar:
#             loading_bar = tqdm
#         else:
#             loading_bar = lambda x: x

#         if verbose: print('Making predictions on %s.' % device)

#         self.eval()
        
#         # inference loop!
#         embeddings_all = [] # NOTE: due to legacy reasons, these "embeddings" are different, as they ae those of HCNN, not of the input
#         y_hat_all_logits = []
#         y_hat_all_index = []
#         y_all = []
#         res_ids_all = []
#         for i, (zgram, zgram_vec, input_emb, y, (rot, res_ids)) in loading_bar(enumerate(dataloader)):
#             zgram = put_dict_on_device(zgram, device)
#             input_emb = input_emb.to(device)
#             y = y.to(device)
#             self.eval()
            
#             if emb_i is not None:
#                 zgram_copy = deepcopy(zgram)
#                 embeddings = self.get_inv_embedding(zgram_copy, emb_i=emb_i)
#             else:
#                 embeddings = torch.zeros(10)
            
#             y_hat = self(zgram, input_emb, return_individual_predictions=False)

#             if emb_i == -1:
#                 # little sanity check
#                 assert np.allclose(self.output_head(embeddings).detach().cpu().numpy(), y_hat.detach().cpu().numpy())

#             embeddings_all.append(embeddings.detach().cpu().numpy())
#             y_hat_all_logits.append(y_hat.detach().cpu().numpy())
#             y_hat_all_index.append(np.argmax(y_hat.detach().cpu().numpy(), axis=1))
#             y_all.append(y.detach().cpu().numpy())
#             res_ids_all.append(res_ids)

#         embeddings_all = np.vstack(embeddings_all)
#         y_hat_all_logits = np.vstack(y_hat_all_logits)
#         y_hat_all_index = np.hstack(y_hat_all_index)
#         y_all = np.hstack(y_all)
#         res_ids_all = np.hstack(res_ids_all)
    
#         return {
#             'embeddings': embeddings_all, # NOTE: due to legacy reasons, these "embeddings" are different, as they ae those of HCNN, not of the input
#             'logits': y_hat_all_logits,
#             'probabilities': softmax(y_hat_all_logits.astype(np.float64), axis=-1),
#             'best_indices': y_hat_all_index,
#             'targets': y_all,
#             'res_ids': res_ids_all
#         }
