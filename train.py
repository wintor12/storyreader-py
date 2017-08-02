import argparse
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torchtext.data import BucketIterator
import models
import dill
from datetime import datetime
from pycrayon import CrayonClient


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=[], nargs='+', type=int,
                    help='Use CUDA on the listed devices')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout between layers')

parser.add_argument('--data', default='./data/', help='the path to load data')
parser.add_argument('--save', default='./data/model', help='the path to save model files')
parser.add_argument('--crayon', action='store_true', help='visualization')

opt = parser.parse_args()
print(opt)


def train(model, trainData, epoch, optimizer, criterion, tb_train=None):
    model.train()
    train = BucketIterator(
        dataset=trainData, batch_size=opt.batch_size,
        device=opt.gpus[0] if opt.gpus else -1,
        repeat=False)

    criterion.size_average = True
    for batch_idx, batch in enumerate(train):
        optimizer.zero_grad()
        output = model(batch.feature)
        loss = criterion(output, batch.tgt)
        loss.backward()
        optimizer.step()
        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train.batch_size, len(trainData),
                100. * batch_idx / len(train), loss.data[0]))
        if tb_train:
            tb_train.add_scalar_dict(
                data={'loss': loss.data[0]},
                step=epoch)


def val(model, validData, epoch, criterion, tb_valid=None):
    model.eval()
    valid = BucketIterator(
        dataset=validData, batch_size=opt.batch_size,
        device=opt.gpus[0] if opt.gpus else -1,
        repeat=False, train=False, sort=True)

    criterion.size_average = False
    loss = 0
    for batch_idx, batch in enumerate(valid):
        output = model(batch.feature)
        loss += criterion(output, batch.tgt)
    loss /= len(validData)
    print('Eval: \tLoss: {:.6f}'.format(loss.data[0]))
    if tb_valid:
        tb_valid.add_scalar_dict(
            data={'loss': loss.data[0]},
            step=epoch)
    return loss


def main():
    # load dataset
    trainData = torch.load(opt.data + 'train.pt', pickle_module=dill)
    fields = torch.load(opt.data + 'fields.pt', pickle_module=dill)
    validData = torch.load(opt.data + 'valid.pt', pickle_module=dill)
    fields = dict(fields)
    print(list(fields.keys()))
    print(' * vocabulary size. source = %d *' % len(fields['src'].vocab))

    tb_train, tb_valid = None, None
    if opt.crayon:
        tb_client = CrayonClient()
        tb_name = '{}-{}'.format(datetime.now().strftime("%y%m%d-%H%M%S"),
                                 opt.save)
        tb_train = tb_client.create_experiment('{}/train'.format(tb_name))
        tb_valid = tb_client.create_experiment('{}/valid'.format(tb_name))

    num_features = len(trainData[0].feature)
    print('Num of features: ' + str(num_features))

    model = models.Fc(num_features, opt.hidden1, opt.hidden2, opt.dropout)
    criterion = nn.MSELoss()

    if opt.gpus:
        model.cuda()
        criterion.cuda()

    lr = opt.lr
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_old, loss, loss_best = float("inf"), 0, float("inf")
    for e in range(1, opt.epoch + 1):
        optimizer = optim.SGD(model.parameters(), lr=lr)
        train(model, trainData, e, optimizer, criterion, tb_train)
        loss = val(model, validData, e, criterion, tb_valid)
        print('LR: \t: {:.6f}'.format(lr))
        if loss.data[0] < loss_old:
            if loss.data[0] < loss_best:
                loss_best = loss.data[0]
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': opt,
                    'epoch': e,
                    'optim': optimizer
                }
                torch.save(checkpoint,
                           '%s_loss_%.5f_e%d.pt' % (opt.save, loss_best, e))
                loss_old = loss.data[0]
            else:
                lr = lr * 0.5
                print('')


if __name__ == "__main__":
    main()
