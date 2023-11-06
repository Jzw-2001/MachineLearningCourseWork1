import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, nfeat, nhid, nlayers, dropout, alpha, training):
        super(NeuralNetwork, self).__init__()
        self.node_feature_dim = nfeat
        self.hidden_dim = nhid
        self.n_layers = nlayers
        self.drop = dropout
        self.batch_size=256
        self.training = training

        self.first_linear = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.linears = nn.ModuleList()
        for i in range(self.n_layers):
            self.linears.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.last_linear = nn.Linear(self.hidden_dim, 1)
        
        self.norm_features = nn.BatchNorm1d(self.node_feature_dim)
        self.norm = nn.BatchNorm1d(self.hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            print('reset_parameters', p)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.norm_features(x)
        x = F.relu(self.first_linear(x))
        x = F.dropout(x, p=self.drop, training=self.training)

        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                x = self.last_linear(x)
                continue

            x = self.linears[i](x)
            x = self.norm(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop, training=self.training)
        
        x = torch.sigmoid(x)
        return x
    