import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import models.FeatureModels as Models
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
import os
import torch.optim.lr_scheduler as lr_scheduler
import dill


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nn', type=str, help='model')
parser.add_argument('--batchnorm', action='store_true')


parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')

parser.add_argument('--seed', type=int, default=1234, help='seed')
parser.add_argument('--gpus', default=[], nargs='+', type=int,
                    help='Use CUDA on the listed devices')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate')
parser.add_argument('--decay_factor', type=float, default=0.5,
                    help='factor by which the learning rate will be reduced')
parser.add_argument('--patience', type=int, default=2,
                    help='''number of epochs with no improvement after which
                    learning rate with be reduced. ''')
parser.add_argument('--epoch_fix_lr', type=int, default=10,
                    help='number of epochs to train for')

parser.add_argument('--optim', default='adam',
                    help="""Optimization method.
                    [sgd|adam]""")

parser.add_argument('--param_init', type=float, default=0.1,
                    help="Parameters are initialized over uniform distribution")

parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.3, help='dropout between layers')
parser.add_argument('--save', default='./feature_model/',
                    help='the path to save model files')


opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save):
    os.makedirs(opt.save)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    torch.cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


def train(model, train_loader, epoch, optimizer, criterion):
    model.train()
    criterion.size_average = True
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.float()
        if len(opt.gpus) > 0:
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


def val(model, val_loader, epoch, criterion):
    model.eval()
    criterion.size_average = False
    loss = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        target = target.float()
        if len(opt.gpus) > 0:
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
    feature_train, feature_val, feature_test = (feature_train[:, :2],
                                                feature_val[:, :2],
                                                feature_test[:, :2])

    # add log view feature
    feature_train = np.append(feature_train,
                              np.log(feature_train[:, 1]).reshape(-1, 1), 1)
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

    return feature_train[:, 1:], feature_val[:, 1:], \
        feature_test[:, 1:], y_train, y_val, y_test


def main():
    feature_train, feature_val, feature_test, y_train, y_val, y_test = dataLoad()
    print(feature_train.shape, feature_val.shape, feature_test.shape)
    if opt.model != 'nn':
        model = RandomForestRegressor(n_estimators=100)
        print(model)
        model.fit(feature_train, y_train)
        pred = model.predict(feature_test)
        print(np.mean(np.power(pred - y_test, 2)))
        return

    feature_train, feature_val, feature_test = (torch.from_numpy(feature_train).float(),
                                                torch.from_numpy(feature_val).float(),
                                                torch.from_numpy(feature_test).float())
    y_train, y_val, y_test = (torch.from_numpy(y_train).float(),
                              torch.from_numpy(y_val).float(),
                              torch.from_numpy(y_test).float())
    print(feature_train.size(), feature_val.size(), y_train.size(), y_val.size())
    dataset_train, dataset_val, dataset_test = (TensorDataset(feature_train, y_train),
                                                TensorDataset(feature_val, y_val),
                                                TensorDataset(feature_test, y_test))
    train_loader = DataLoader(dataset_train, batch_size=opt.batchSize,
                              shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset_val, batch_size=opt.batchSize,
                            shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset_test, batch_size=opt.batchSize,
                             shuffle=True, num_workers=1)

    num_features = feature_train.size(1)
    model = Models.Base(num_features, opt)
    print(model)

    print('Intializing params')
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

    criterion = nn.MSELoss()

    if len(opt.gpus) > 0:
        model.cuda()
        criterion.cuda()

    if opt.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt.lr)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                               patience=opt.patience,
                                               factor=opt.decay_factor)
    loss_old, loss, loss_best = float("inf"), 0, float("inf")
    best_model = None
    for e in range(1, opt.epoch + 1):
        train(model, train_loader, e, optimizer, criterion)
        loss = val(model, val_loader, e, criterion)
        if e > opt.epoch_fix_lr:
            scheduler.step(loss.data[0])
        print('LR: \t: {:.10f}'.format(optimizer.param_groups[0]['lr']))
        if loss.data[0] < loss_old:
            if loss.data[0] < loss_best:
                loss_best = loss.data[0]
                best_model = model
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': opt,
                    'epoch': e,
                    'optim': optimizer
                }
                filename = 'e%d_%.5f' % (e, loss_best)
                torch.save(checkpoint, os.path.join(opt.save, filename),
                           pickle_module=dill)
        loss_old = loss.data[0]

    # test loss:
    print('test loss: ')
    val(best_model, test_loader, 0, criterion)


if __name__ == "__main__":
    main()
