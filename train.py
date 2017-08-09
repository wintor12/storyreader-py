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
import torch.optim.lr_scheduler as lr_scheduler
import os


parser = argparse.ArgumentParser()
parser.add_argument('--reader', type=str, default='r',
                    help='r(Regional Reader)|s(Sequential Reader)|h(Holistic Reader)')
parser.add_argument('--gpus', default=[], nargs='+', type=int,
                    help='Use CUDA on the listed devices')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--seed', type=int, default=1234, help='seed')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=128)

# optimizer
parser.add_argument('--lr', type=float, default=0.00015, metavar='LR',
                    help='learning rate')
parser.add_argument('--decay_factor', type=float, default=0.1,
                    help='factor by which the learning rate will be reduced')
parser.add_argument('--patience', type=int, default=2,
                    help='''number of epochs with no improvement after which
                    learning rate with be reduced. ''')
parser.add_argument('--optim', default='adam',
                    help="""Optimization method.
                    [sgd|adam]""")

parser.add_argument('--dropout', type=float, default=0.5, help='dropout between layers')
parser.add_argument('--param_init', type=float, default=0.1,
                    help="Parameters are initialized over uniform distribution")

parser.add_argument('--word_vec_size', type=int, default=300,
                    help='Word embedding sizes')
parser.add_argument('--pre_word_vec', action='store_true',
                    help='Use pre-trained word embeddings')
parser.add_argument('--fix_word_vec', action='store_true',
                    help='if true, word embeddings are fixed during training')


parser.add_argument('--region_nums', type=int, default=10,
                    help="Number of regions in each text")
parser.add_argument('--region_words', type=int, default=36,
                    help="Number of words in each region")

parser.add_argument('--data', default='./data/', help='the path to load data')
parser.add_argument('--save', default='./data/model/',
                    help='the path to save model files')
parser.add_argument('--crayon', action='store_true', help='visualization')
parser.add_argument('--debug', action='store_true', help='debug the model')


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


def train(model, trainData, epoch, optimizer, criterion, tb_train=None):
    model.train()
    train = BucketIterator(
        dataset=trainData, batch_size=opt.batch_size,
        device=opt.gpus[0] if opt.gpus else -1,
        repeat=False)

    criterion.size_average = True
    for batch_idx, batch in enumerate(train):
        optimizer.zero_grad()
        output = model(batch)
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
        output = model(batch)
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
    print('Loading data ... ')
    trainData = torch.load(opt.data + 'train.pt', pickle_module=dill)
    fields = torch.load(opt.data + 'fields.pt', pickle_module=dill)
    validData = torch.load(opt.data + 'valid.pt', pickle_module=dill)
    if opt.debug:
        testData = torch.load(opt.data + 'test.pt', pickle_module=dill)
    fields = dict(fields)
    print(list(fields.keys()))
    vocab = fields['src'].vocab
    print(' * vocabulary size. source = %d *' % len(vocab))

    tb_train, tb_valid, tb_test = None, None, None
    if opt.crayon:
        tb_client = CrayonClient()
        tb_name = '{}-{}'.format(datetime.now().strftime("%y%m%d-%H%M%S"),
                                 opt.save)
        tb_train = tb_client.create_experiment('{}/train'.format(tb_name))
        tb_valid = tb_client.create_experiment('{}/valid'.format(tb_name))
        tb_test = tb_client.create_experiment('{}/test'.format(tb_name))

    num_features = len(trainData[0].feature)
    print('Num of features: ' + str(num_features))

    s_rcnn = models.RegionalCNN(opt, opt.region_nums)
    q_rcnn = models.RegionalCNN(opt, 1)
    fc = models.Fc(num_features + 110, opt)

    if opt.reader == 'r':
        model = models.RegionalReader(vocab, opt.word_vec_size,
                                      s_rcnn, q_rcnn, fc)
    elif opt.reader == 's':
        model = models.SequentialReader(vocab, opt.word_vec_size,
                                        s_rcnn, q_rcnn, fc)
    elif opt.reader == 'h':
        model = models.HolisticReader(vocab, opt.word_vec_size,
                                      s_rcnn, q_rcnn, fc)
    else:
        raise Exception('reader has to be "r" or "s" or "h"')
    print(model)

    print('Intializing params')
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

    # load pre_trained word vectors
    wv = None
    if opt.pre_word_vec:
        wv = torch.load(opt.data + 'wv.pt', pickle_module=dill)
        model.load_pretrained_vectors(wv)

    # fix word embeddings
    if opt.fix_word_vec:
        model.embed.weight.requires_grad = False

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
    for e in range(1, opt.epoch + 1):
        train(model, trainData, e, optimizer, criterion, tb_train)
        loss = val(model, validData, e, criterion, tb_valid)
        scheduler.step(loss.data[0])
        print('LR: \t: {:.10f}'.format(optimizer.param_groups[0]['lr']))
        if loss.data[0] < loss_old:
            if loss.data[0] < loss_best:
                loss_best = loss.data[0]
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


if __name__ == "__main__":
    main()
