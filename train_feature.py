import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import models
import sys
from datetime import datetime
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nn', type=str, help='model')

parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                                        help='how many batches to wait before logging training status')
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                                        help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout between layers')
parser.add_argument('--save', default='./data/model', help='the path to save model files')


opt = parser.parse_args()
print(opt)


def train(model, train_loader, epoch, optimizer, criterion, tb_train=None):
    model.train()
    criterion.size_average = True
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.float()
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))            


def val(model, val_loader, epoch, criterion, tb_valid=None):
    model.eval()
    criterion.size_average = False
    loss = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        target = target.float()
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model(data)
        loss += criterion(output, target)
        loss /= len(val_loader.dataset)
        print('Eval: \tLoss: {:.6f}'.format(loss.data[0]))
    return loss


def dataLoad():
    # Load old version and new version raw data
    feature_train, feature_val, feature_test = (np.loadtxt('all_data/feature_train'),
                                                np.loadtxt('all_data/feature_val'),
                                                np.loadtxt('all_data/feature_test'))
    y_train, y_val, y_test = (np.loadtxt('all_data/y_train'),
                              np.loadtxt('all_data/y_val'),
                              np.loadtxt('all_data/y_test'))
    day_train, day_val, day_test = (np.loadtxt('all_data/day_train'),
                                    np.loadtxt('all_data/day_val'),
                                    np.loadtxt('all_data/day_test'))

    feature2_train, feature2_val, feature2_test = (np.loadtxt('all_data/feature2_train'),
                                                   np.loadtxt('all_data/feature2_val'),
                                                   np.loadtxt('all_data/feature2_test'))
    y2_train, y2_val, y2_test = (np.loadtxt('all_data/y2_train'),
                                 np.loadtxt('all_data/y2_val'),
                                 np.loadtxt('all_data/y2_test'))
    day2_train, day2_val, day2_test = (np.loadtxt('all_data/day2_train'),
                                       np.loadtxt('all_data/day2_val'),
                                       np.loadtxt('all_data/day2_test'))
    #  Append datasets and day feature
    def appendDay(day, feature):
        if len(day.shape) == 1:
            day = np.expand_dims(day, -1)
        return np.append(day, feature, axis=1)
    
    feature_train, feature_val, feature_test = (appendDay(day_train, feature_train),
                                                appendDay(day_val, feature_val),
                                                appendDay(day_test, feature_test))
    feature2_train, feature2_val, feature2_test = (appendDay(day2_train, feature2_train),
                                                   appendDay(day2_val, feature2_val),
                                                   appendDay(day2_test, feature2_test))
    def appendData(data1, data2):
        return np.append(data1, data2, axis=0)

    feature_train, feature_val, feature_test = (appendData(feature_train, feature2_train),
                                                appendData(feature_val, feature2_val),
                                                appendData(feature_test, feature2_test))
    y_train, y_val, y_test = (appendData(y_train, y2_train),
                              appendData(y_val, y2_val),
                              appendData(y_test, y2_test))

    # Only take the first two features: days, views
    feature_train, feature_val, feature_test = (feature_train[:,:2],
                                                feature_val[:,:2],
                                                feature_test[:,:2])
    
    # add log view feature
    feature_train = \
        np.append(feature_train, np.log(feature_train[:, 1]).reshape(-1, 1), 1)
    feature_val = np.append(feature_val, np.log(feature_val[:, 1]).reshape(-1, 1), 1)
    feature_test = np.append(feature_test, np.log(feature_test[:, 1]).reshape(-1, 1), 1)

    # standardize features
    train_mean = np.mean(feature_train, 0)
    train_std = np.std(feature_train, 0)
    feature_train = (feature_train - train_mean) / train_std
    feature_val = (feature_val - train_mean) / train_std
    feature_test = (feature_test - train_mean) / train_std

    # log upvotes
    y_train, y_val, y_test = np.log(y_train + 1), np.log(y_val + 1), np.log(y_test + 1)

    
    return feature_train, feature_val, feature_test, y_train, y_val, y_test


def main():
    feature_train, feature_val, feature_test, y_train, y_val, y_test = dataLoad()
    print(feature_train.shape, feature_val.shape, feature_test.shape)
    if opt.model != 'nn':
        model = RandomForestRegressor(n_estimators = 100)
        print(model)
        model.fit(feature_train[:,:], y_train)
        pred = model.predict(feature_test[:,:])
        print(np.mean(np.power(pred - y_test, 2)))
        return
        
    feature_train, feature_val, feature_test = (torch.from_numpy(feature_train).float(),
                                                torch.from_numpy(feature_val).float(),
                                                torch.from_numpy(feature_test).float())
    y_train, y_val, y_test = (torch.from_numpy(y_train).float(),
                              torch.from_numpy(y_val).float(),
                              torch.from_numpy(y_val).float())
    print(feature_train.size(), feature_val.size(), y_train.size(), y_val.size())
    

if __name__ == "__main__":
    main()
