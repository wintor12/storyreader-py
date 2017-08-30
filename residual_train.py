import argparse
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torchtext.data import BucketIterator
import models.Models as Models
import models.FeatureModels as ModelsBase
import models.ModelsFixLen as ModelsFixLen
import models.ResidualModels as RModel
import dill
from datetime import datetime
from pycrayon import CrayonClient
import torch.optim.lr_scheduler as lr_scheduler
import os
import utils
from utils import Statistics


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=[], nargs='+', type=int,
                    help='Use CUDA on the listed devices')
parser.add_argument('--seed', type=int, default=1234, help='seed')
parser.add_argument('--f_model', required=True,
                    help='the path to load feature model')
parser.add_argument('--t_model', required=True,
                    help='the path to load text model')
parser.add_argument('--data', default='./residual_model/', help='the path to load data')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=40, help='number of epochs to train for')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./residual_model/',
                    help='the path to save model files')
parser.add_argument('--fix_word_vec', action='store_true',
                    help='if true, word embeddings are fixed during training')


# optimizer
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate')
parser.add_argument('--decay_factor', type=float, default=0.5,
                    help='factor by which the learning rate will be reduced')
parser.add_argument('--patience', type=int, default=2,
                    help='''number of epochs with no improvement after which
                    learning rate with be reduced. ''')
parser.add_argument('--optim', default='adam',
                    help="""Optimization method.
                    [sgd|adam]""")
parser.add_argument('--epoch_fix_lr', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('--grad_clipping', type=float, default=0,
                    help='clip the gradients of region cnn')

parser.add_argument('--crayon', action='store_true', help='visualization')
parser.add_argument('--debug', action='store_true', help='debug the model')


opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save):
    os.makedirs(opt.save)


if opt.gpus:
    torch.cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")


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
        if opt.grad_clipping:
            nn.utils.clip_grad_norm(model.q_rcnn.parameters(),
                                    opt.grad_clipping)
            nn.utils.clip_grad_norm(model.s_rcnn.parameters(),
                                    opt.grad_clipping)
        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train.batch_size, len(trainData),
                100. * batch_idx / len(train), loss.data[0]))
        if tb_train:
            stat = Statistics(loss=loss.data[0])
            if opt.debug:
                feature_model = model.feature_model
                text_model = model.text_model
                stat = Statistics(loss=loss.data[0],
                                  model_grad=utils.weight_grad_norm(
                                      model.parameters()),
                                  f_model_grad=utils.weight_grad_norm(
                                      feature_model.parameters()),
                                  t_model_grad=utils.weight_grad_norm(
                                      text_model.parameters()),
                                  embed_grad=utils.weight_grad_norm(
                                      text_model.embed.parameters()),
                                  q_grad=utils.weight_grad_norm(
                                      text_model.q_rcnn.parameters()),
                                  s_grad=utils.weight_grad_norm(
                                      text_model.s_rcnn.parameters()),
                                  fc_grad=utils.weight_grad_norm(
                                      text_model.fc.parameters()),
                                  # rnn_cell=utils.weight_grad_norm(
                                  #     text_model.rnn_cell.parameters())
                                  # if opt.reader == 's' else None,
                                  # rnn=utils.weight_grad_norm(
                                  #     text_model.rnn.parameters())
                                  # if opt.reader == 'h' else None
                                  )
            tb_train.add_scalar_dict(
                data=stat.__dict__,
                step=epoch
            )

        optimizer.step()


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
    print("Loading feture model ... ")
    checkpoint = torch.load(opt.f_model,
                            map_location=lambda storage, loc: storage,
                            pickle_module=dill)
    f_model_opt = checkpoint['opt']
    print(f_model_opt)
    f_num_features = 2
    feature_model = ModelsBase.Base(f_num_features, f_model_opt)
    print('feature_model')
    print(feature_model)
    feature_model.load_state_dict(checkpoint['model'])

    print("Loading text model ... ")
    trainData = torch.load(opt.data + 'train.pt', pickle_module=dill)
    fields = torch.load(opt.data + 'fields.pt', pickle_module=dill)
    validData = torch.load(opt.data + 'valid.pt', pickle_module=dill)
    testData = torch.load(opt.data + 'test.pt', pickle_module=dill)
    fields = dict(fields)
    print(list(fields.keys()))
    vocab = fields['src'].vocab
    print(' * vocabulary size. source = %d *' % len(vocab))

    checkpoint = torch.load(opt.t_model,
                            map_location=lambda storage, loc: storage,
                            pickle_module=dill)

    model_opt = checkpoint['opt']
    print(model_opt)

    s_rcnn = Models.RegionalCNN(model_opt)
    q_rcnn = Models.RegionalCNN(model_opt)

    num_features = 0
    fc_input_dim = num_features + model_opt.r_emb * (
        model_opt.region_nums + 1 if model_opt.region_nums > 0 else 1)
    fc = Models.Fc(fc_input_dim, model_opt)
    if model_opt.reader == 'r':
        model = Models.RegionalReader(
            fields['src'].vocab, model_opt.word_vec_size,
            s_rcnn, q_rcnn, fc, model_opt)
    elif model_opt.reader == 's':
        if model_opt.region_nums:
            model = ModelsFixLen.SequentialReader(
                fields['src'].vocab, model_opt.word_vec_size,
                s_rcnn, q_rcnn, fc, model_opt)
        else:
            model = Models.SequentialReader(
                fields['src'].vocab, model_opt.word_vec_size,
                s_rcnn, q_rcnn, fc, model_opt)
    elif model_opt.reader == 'h':
        if model_opt.region_nums:
            model = ModelsFixLen.HolisticReader(
                fields['src'].vocab, model_opt.word_vec_size,
                s_rcnn, q_rcnn, fc, model_opt)
        else:
            model = Models.HolisticReader(
                fields['src'].vocab, model_opt.word_vec_size,
                s_rcnn, q_rcnn, fc, model_opt)
    else:
        raise Exception('reader has to be "r" or "s" or "h"')
    print('text_model')
    print(model)
    model.load_state_dict(checkpoint['model'])

    # fix word embeddings
    if opt.fix_word_vec:
        model.embed.weight.requires_grad = False

    residual_model = RModel.ResidualModel(feature_model, model)

    nParams = sum([p.nelement() for p in residual_model.parameters()])
    print('* number of parameters: %d' % nParams)
    enc, dec = 0, 0
    for name, param in residual_model.named_parameters():
        if param.requires_grad:
            if 'embed' in name:
                print(name, param.nelement())
            elif 'feature_model' in name:
                enc += param.nelement()
            elif 'text_model' in name:
                dec += param.nelement()
            else:
                print(name, param.nelement())
    print('feature_model: ', enc)
    print('text_model: ', dec)

    criterion = nn.MSELoss()
    if len(opt.gpus) > 0:
        residual_model.cuda()
        criterion.cuda()

    if opt.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     residual_model.parameters()),
                              lr=opt.lr)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      residual_model.parameters()),
                               lr=opt.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                               patience=opt.patience,
                                               factor=opt.decay_factor)
    loss_old, loss, loss_best = float("inf"), 0, float("inf")
    bestModel = None

    tb_train, tb_valid = None, None
    if opt.crayon:
        tb_client = CrayonClient()
        tb_name = '{}-{}'.format(datetime.now().strftime("%y%m%d-%H%M%S"),
                                 opt.save)
        tb_train = tb_client.create_experiment('{}/train'.format(tb_name))
        tb_valid = tb_client.create_experiment('{}/valid'.format(tb_name))

    for e in range(1, opt.epoch + 1):
        train(residual_model, trainData, e, optimizer, criterion, tb_train)
        loss = val(residual_model, validData, e, criterion, tb_valid)
        if e > opt.epoch_fix_lr:
            scheduler.step(loss.data[0])
        print('LR: \t: {:.10f}'.format(optimizer.param_groups[0]['lr']))
        if loss.data[0] < loss_old:
            if loss.data[0] < loss_best:
                bestModel = residual_model
                loss_best = loss.data[0]
                checkpoint = {
                    'model': residual_model.state_dict(),
                    'opt': opt,
                    'epoch': e,
                    'optim': optimizer
                }
                filename = 'e%d_%.5f' % (e, loss_best)
                torch.save(checkpoint, os.path.join(opt.save, filename),
                           pickle_module=dill)
        loss_old = loss.data[0]

    print("Computing test loss ... ")
    loss = val(bestModel, testData, 0, criterion)
    print(loss)
    # pred = pred.data.squeeze(1).cpu()
    # indice = indice.data.cpu()
    # _, order = torch.sort(indice)
    # pred = torch.index_select(pred, 0, order)
    # print(pred)
    # y = torch.FloatTensor([ex.tgt for ex in testData])
    # mse = torch.mean((pred - y) ** 2)
    # print('Manualy compute: ', mse)


if __name__ == "__main__":
    main()
