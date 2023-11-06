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

from dataset import SeoulBikeDataset
from models import NeuralNetwork
from utils import debug_log


def train(train_params, batch_size):
    model.train()
    train_loss = 0
    train_mae = 0
    train_rmse = 0
    accuracy_train = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
 
        output = model(data)
        # print("data", data)
        # print("output", output)
        # print("target", target)

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += F.l1_loss(output, target).item()
        train_rmse += torch.sqrt(F.mse_loss(output, target)).item()
        # use r2 to calculate accuracy
        accuracy_train += r2_score(target.cpu().detach().numpy(), output.cpu().detach().numpy())

    # print("len:", len(train_loader))
    # print("train_loss:", train_loss)
    # print("train_mae:", train_mae)
    # print("train_rmse:", train_rmse)
    train_loss /= len(train_loader)
    train_mae /= len(train_loader)
    train_rmse /= len(train_loader)
    accuracy_train /= len(train_loader)
    print("Train accuracy:", accuracy_train)
    print('Train set: Average loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(
        train_loss, train_mae, train_rmse))
    debug_log('Train set: Average loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(
        train_loss, train_mae, train_rmse), train_params['log_file'])

def test():
    model.eval()
    test_loss = 0
    test_mae = 0
    test_rmse = 0
    accuracy_test = 0

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    targets = torch.Tensor()
    outputs = torch.Tensor()
    targets = targets.cuda()
    outputs = outputs.cuda()

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        # print("target",target.shape)
        # print("output",output.shape)
        targets = torch.cat((targets, target), 0)
        outputs = torch.cat((outputs, output), 0)
        #print target value and output value
        # print("target",target)
        # print("output",output)
    # print("outputs",outputs)
    # print("targets",targets)


    test_loss = F.mse_loss(outputs, targets)
    test_mae = F.l1_loss(outputs, targets)
    test_rmse = torch.sqrt(F.mse_loss(outputs, targets))
    accuracy_test = r2_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())


    print("Test accuracy:", accuracy_test)
    print('Test set: Average loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(
        test_loss, test_mae, test_rmse))
    
    debug_log('Test set: Average loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(
        test_loss, test_mae, test_rmse), train_params['log_file'])


train_params = yaml.load(open('trainparams.yaml', 'r'), Loader=yaml.FullLoader)

# load the dataset
dataset = SeoulBikeDataset(train_params['dataset_dir'])
train_size = int(len(dataset) *
    (train_params['train_ratio'] + train_params['val_ratio']))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

# load the model
model = NeuralNetwork(
    nfeat = dataset[0][0].shape[0],
    nhid = train_params['hidden_units'],
    nlayers = train_params['n_layers'],
    dropout = train_params['dropout'],
    alpha = train_params['alpha'],
    training = True
)

optimizer = optim.SGD(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
model = model.cuda()

for epoch in range(train_params['epochs']):
    debug_log('Epoch {}'.format(epoch), train_params['log_file'])
    # train the model
    train(train_params, batch_size=train_params['batch_size'])
    # test the model
    test()