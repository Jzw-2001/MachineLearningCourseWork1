import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from torch import nn

from dataset import ShillBiddingDataset
from models import NeuralNetwork
from utils import debug_log


def accuracy_of_class(predict, target):
    accuracy = 0
    for i in range(len(predict)):
        if round(float(predict[i])) == int(target[i]):
            accuracy += 1
    accuracy /= len(predict)
    return accuracy
        


def train(train_params, batch_size):
    model.train()
    train_loss = 0
    accuracy_train = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
 
        output = model(data)
        output = output.cuda()

        target = target.unsqueeze(1)

        loss_fn = nn.BCELoss()
        loss = loss_fn(output.float(), target.float())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        accuracy_train += accuracy_of_class(output, target)

    train_loss /= len(train_loader)
    accuracy_train /= len(train_loader)

    print("Train accuracy:", accuracy_train)
    debug_log("Train accuracy: {}".format(accuracy_train), train_params['log_file'])
    print('Train set: Average loss: {:.4f}'.format(
        train_loss))
    debug_log('Train set: Average loss: {:.4f}'.format(
        train_loss), train_params['log_file'])
    


def test():
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    targets = torch.Tensor()
    outputs = torch.Tensor()
    targets = targets.cuda()
    outputs = outputs.cuda()

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        output = output.cuda()
        target = target.unsqueeze(1)
        outputs = torch.cat((outputs, output), 0)
        targets = torch.cat((targets, target), 0)

    loss_fn = nn.BCELoss()
    test_loss = loss_fn(outputs.float(), targets.float())
    accuracy_test = accuracy_of_class(outputs, targets)

    print("Test accuracy:", accuracy_test)
    debug_log("Test accuracy: {}".format(accuracy_test), train_params['log_file'])
    print('Test set: Average loss: {:.4f}'.format(
        test_loss))
    debug_log('Test set: Average loss: {:.4f}'.format(
        test_loss), train_params['log_file'])



train_params = yaml.load(open('trainparams.yaml'), Loader=yaml.FullLoader)

dataset = ShillBiddingDataset(train_params['dataset_dir'])
train_size = int(train_params['train_ratio'] * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

model = NeuralNetwork(
    nfeat = dataset[0][0].shape[0],
    nhid = train_params['hidden_units'],
    nlayers = train_params['n_layers'],
    dropout = train_params['dropout'],
    alpha = train_params['alpha'],
    training = True
)

optimizer = optim.SGD(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

model.cuda()

for epoch in range(train_params['epochs']):
    debug_log('Epoch {}'.format(epoch), train_params['log_file'])
    # train the model
    train(train_params, batch_size=train_params['batch_size'])
    # test the model
    test()