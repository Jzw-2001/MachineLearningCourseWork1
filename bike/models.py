# Date,Rented Bike Count,Hour,Temperature(�C),Humidity(%),Wind speed (m/s),Visibility (10m),Dew point temperature(�C),Solar Radiation (MJ/m2),Rainfall(mm),Snowfall (cm),Seasons,Holiday,Functioning Day
# 01/12/2017,254,0,-5.2,37,2.2,2000,-17.6,0,0,0,Winter,No Holiday,Yes
# 01/12/2017,204,1,-5.5,38,0.8,2000,-17.6,0,0,0,Winter,No Holiday,Yes
# 01/12/2017,173,2,-6,39,1,2000,-17.7,0,0,0,Winter,No Holiday,Yes
# 01/12/2017,107,3,-6.2,40,0.9,2000,-17.6,0,0,0,Winter,No Holiday,Yes
# ...

# a model for predicting the number of bikes rented in a given hour

# import libraries
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
        # Date,Rented Bike Count,Hour,Temperature(°C),Humidity(%),Wind speed (m/s),Visibility (10m),Dew point temperature(°C),Solar Radiation (MJ/m2),Rainfall(mm),Snowfall (cm),Seasons,Holiday,Functioning Day
        # 01/12/2017,254,0,-5.2,37,2.2,2000,-17.6,0,0,0,Winter,No Holiday,Yes
        # 01/12/2017,204,1,-5.5,38,0.8,2000,-17.6,0,0,0,Winter,No Holiday,Yes
        # 01/12/2017,173,2,-6,39,1,2000,-17.7,0,0,0,Winter,No Holiday,Yes
        # predict the number of bikes rented in a given hour
        x = self.norm_features(x)
        x = F.relu(self.first_linear(x))
        x = F.dropout(x, p=self.drop, training=self.training)

        for i in range(self.n_layers):
            x = self.linears[i](x)
            x = self.norm(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
        
        # output 1 integer as the number of bikes rented in a given hour
        x = F.relu(self.linears[-1](x))
        x = torch.sum(x, dim=1)
        return x