import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import utils
import models

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')

opt = parser.parse_args()
print(opt)


def train(model, train_loader, epoch, optimizer, criterion):
    model.train()
    for e in range(epoch):
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
                        e, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))


def main():
    feature_train, feature_val = utils.loadFeatures()
    y_train, y_val = utils.loadUpvote()
    feature_train = torch.from_numpy(feature_train).float()
    feature_val = torch.from_numpy(feature_val)
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val)
    dataset_train = TensorDataset(feature_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=32,
                              shuffle=True, num_workers=1)
    print(len(train_loader))
    print(feature_train.size())
    model = models.Fc(feature_train.size(1), opt.hidden1, opt.hidden2)
    criterion = nn.MSELoss()
    if opt.cuda:
        model.cuda()
        criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    train(model, train_loader, opt.epoch, optimizer, criterion)


if __name__ == "__main__":
    main()
