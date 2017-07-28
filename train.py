import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import utils
import models
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')

opt = parser.parse_args()
print(opt)


def train(model, train_loader, epoch, optimizer, criterion):
    model.train()
    criterion.size_average = True
    for batch_idx, (data, target) in enumerate(train_loader):
        if opt.cuda:
            data, target = data.cuda(), target.float().cuda()
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


def val(model, val_loader, optimizer, criterion):
    model.eval()
    criterion.size_average = False
    loss = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if opt.cuda:
            data, target = data.cuda(), target.float().cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model(data)
        loss += criterion(output, target)
    loss /= len(val_loader.dataset)
    print('Eval: \tLoss: {:.6f}'.format(loss.data[0]))
    return loss


def main():
    feature_train, feature_val = utils.loadFeatures()
    feature_train, feature_val = utils.preprocessFeatures(feature_train, feature_val)
    y_train, y_val = utils.loadUpvote()
    feature_train = torch.from_numpy(feature_train).float()
    feature_val = torch.from_numpy(feature_val).float()
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()
    dataset_train = TensorDataset(feature_train, y_train)
    dataset_val = TensorDataset(feature_val, y_val)
    train_loader = DataLoader(dataset_train, batch_size=opt.batchSize,
                              shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset_val, batch_size=opt.batchSize,
                            shuffle=True, num_workers=1)
    print('train data: ', len(train_loader.dataset))
    print('val_data', len(val_loader.dataset))
    print('number of features: ', feature_train.size()[1])
    model = models.Fc(feature_train.size(1), opt.hidden1, opt.hidden2)
    criterion = nn.MSELoss()
    if opt.cuda:
        model.cuda()
        criterion.cuda()
    lr = opt.lr
    loss_old, loss = sys.maxint, 0
    for e in range(opt.epoch):
        optimizer = optim.SGD(model.parameters(), lr=lr)
        train(model, train_loader, e, optimizer, criterion)
        loss = val(model, val_loader, optimizer, criterion)
        if loss.data[0] > loss_old:
            lr = lr * 0.5
        loss_old = loss.data[0]
        print('LR: \t: {:.6f}'.format(lr))
        print('')


if __name__ == "__main__":
    main()
