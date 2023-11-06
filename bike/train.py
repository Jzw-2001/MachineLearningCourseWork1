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


def train(train_params, batch_size, epoch):
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
    
    # os.makedirs(os.path.dirname(train_params['pkl_file_path']), exist_ok=True)
    # torch.save(model.state_dict(), train_params['pkl_file_path'])

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
    
    return accuracy_test, test_loss, test_mae, test_rmse


train_params = yaml.load(open('trainparams.yaml', 'r'), Loader=yaml.FullLoader)

# load the dataset
# dataset = SeoulBikeDataset(train_params['dataset_dir'])
# train_size = int(len(dataset) *
#     (train_params['train_ratio'] + train_params['val_ratio']))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

# # load the model
# model = NeuralNetwork(
#     nfeat = dataset[0][0].shape[0],
#     nhid = train_params['hidden_units'],
#     nlayers = train_params['n_layers'],
#     dropout = train_params['dropout'],
#     alpha = train_params['alpha'],
#     training = True
# )

# optimizer = optim.SGD(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
# model = model.cuda()

# for epoch in range(train_params['epochs']):
#     debug_log('Epoch {}'.format(epoch), train_params['log_file'])
#     # train the model
#     train(train_params, batch_size=train_params['batch_size'])
#     # test the model
#     test()


# load dataset
dataset = SeoulBikeDataset(train_params['dataset_dir'])

# load model
model = NeuralNetwork(
    nfeat = dataset[0][0].shape[0],
    nhid = train_params['hidden_units'],
    nlayers = train_params['n_layers'],
    dropout = train_params['dropout'],
    alpha = train_params['alpha'],
    training = True
)
model = model.cuda()

# random suffle the dataset
random.shuffle(dataset)

# k-fold cross validation
k = train_params['k_fold']

best_accuracy = []
best_test_loss = []
best_test_mae = []
best_test_rmse = []

for i in range(k):
    print("k-fold cross validation:", i)
    test_size = int(len(dataset) / k)
    train_size = len(dataset) - test_size

    # train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    test_dataset = dataset[i*test_size:(i+1)*test_size]
    train_dataset = dataset[0:i*test_size] + dataset[(i+1)*test_size:len(dataset)]

    optimizer = optim.SGD(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

    for epoch in range(train_params['epochs']):
        debug_log('Epoch {}'.format(epoch), train_params['log_file'])
        # train the model
        train(train_params, batch_size=train_params['batch_size'], epoch=epoch)
        # test the model
        result = test()
        accuracy_test, test_loss, test_mae, test_rmse = result
        best_accuracy.append(accuracy_test)
        best_test_loss.append(test_loss)
        best_test_mae.append(test_mae)
        best_test_rmse.append(test_rmse)

    print("k-fold cross validation:", i, "finished")
    debug_log("k-fold cross validation: " + str(i) + " finished", train_params['log_file'])

print("all k-fold cross validation finished")
debug_log("all k-fold cross validation finished", train_params['log_file'])
print("best accuracy:")
for i in range(len(best_accuracy)):
    print(best_accuracy[i])
print("best test loss:")
for i in range(len(best_test_loss)):
    print(best_test_loss[i])
print("best test mae:")
for i in range(len(best_test_mae)):
    print(best_test_mae[i])
print("best test rmse:")
for i in range(len(best_test_rmse)):
    print(best_test_rmse[i])
print("average best accuracy:")
print(np.mean(best_accuracy))
print("average best test loss:")
print(np.mean(best_test_loss))
print("average best test mae:")
print(np.mean(best_test_mae))
print("average best test rmse:")
print(np.mean(best_test_rmse))
debug_log("best accuracy:", train_params['log_file'])
for i in range(len(best_accuracy)):
    debug_log(str(best_accuracy[i]), train_params['log_file'])
debug_log("best test loss:", train_params['log_file'])
for i in range(len(best_test_loss)):
    debug_log(str(best_test_loss[i]), train_params['log_file'])
debug_log("best test mae:", train_params['log_file'])
for i in range(len(best_test_mae)):
    debug_log(str(best_test_mae[i]), train_params['log_file'])
debug_log("best test rmse:", train_params['log_file'])
for i in range(len(best_test_rmse)):
    debug_log(str(best_test_rmse[i]), train_params['log_file'])
debug_log("average best accuracy:", train_params['log_file'])
debug_log(str(np.mean(best_accuracy)), train_params['log_file'])
debug_log("average best test loss:", train_params['log_file'])
debug_log(str(np.mean(best_test_loss)), train_params['log_file'])
debug_log("average best test mae:", train_params['log_file'])
debug_log(str(np.mean(best_test_mae)), train_params['log_file'])
debug_log("average best test rmse:", train_params['log_file'])
debug_log(str(np.mean(best_test_rmse)), train_params['log_file'])
