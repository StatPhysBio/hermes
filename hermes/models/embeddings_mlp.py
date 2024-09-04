

import torch
from typing import *


class EmbeddingsMLP(torch.nn.Module):

    def load_embeddings_hparams(self, embedding_hparams):
        self.embedding_dim = embedding_hparams['embedding_dim']
        self.embedding_hidden_dim = embedding_hparams['embedding_hidden_dim']
        self.num_hidden_layers_for_embeddings = embedding_hparams['num_hidden_layers_for_embeddings']
        self.proj_dim = 256 # from the HCNN model
        self.alphabet_size = 20

    def __init__(self,
                 embedding_hparams: Dict):
        super().__init__()
        self.load_embeddings_hparams(embedding_hparams)

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
    
    def forward(self, emb):
        emb = self.embedding_projector(emb)
        emb = self.embedding_output_head(emb)
        return emb
    
