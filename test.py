import argparse
import torch
import torch.nn as nn
import models
import dill
from torchtext.data import BucketIterator


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/', help='the path to load data')
parser.add_argument('--model', required=True, help='the path to the saved model')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--gpu', type=int, default=-1, help='gpu device to run on')


opt = parser.parse_args()
print(opt)

if opt.gpu >= 0:
    torch.cuda.set_device(opt.gpu)


def val(model, validData, criterion, tb_valid=None):
    model.eval()
    valid = BucketIterator(
        dataset=validData, batch_size=opt.batch_size,
        device=opt.gpu,
        repeat=False, train=False,
        sort=True)

    criterion.size_average = False
    loss = 0
    for batch_idx, batch in enumerate(valid):
        output = model(batch)
        loss += criterion(output, batch.tgt)
    loss /= len(validData)
    print('Eval: \tLoss: {:.6f}'.format(loss.data[0]))
    if tb_valid:
        tb_valid.add_scalar_dict(
            data={'loss': loss.data[0]})
    return loss


def main():
    print("Loading data ... ")
    fields = torch.load(opt.data + 'fields.pt', pickle_module=dill)
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage,
                            pickle_module=dill)

    model_opt = checkpoint['opt']
    print(model_opt)
    testData = torch.load(opt.data + 'valid.pt', pickle_module=dill)

    criterion = nn.MSELoss()
    num_features = len(testData[0].feature)

    s_rcnn = models.RegionalCNN(model_opt, model_opt.region_nums)
    q_rcnn = models.RegionalCNN(model_opt, 1)
    fc = models.Fc(num_features + 110, model_opt)
    if model_opt.reader == 'r':
        model = models.RegionalReader(fields['src'].vocab, model_opt.word_vec_size,
                                      s_rcnn, q_rcnn, fc, model_opt)
    elif model_opt.reader == 's':
        model = models.SequentialReader(fields['src'].vocab, model_opt.word_vec_size,
                                        s_rcnn, q_rcnn, fc, model_opt)
    elif model_opt.reader == 'h':
        model = models.HolisticReader(fields['src'].vocab, model_opt.word_vec_size,
                                      s_rcnn, q_rcnn, fc, model_opt)
    else:
        raise Exception('reader has to be "r" or "s" or "h"')
    print(model)

    model.load_state_dict(checkpoint['model'])
    if opt.gpu >= 0:
        model.cuda()
        criterion.cuda()

    tb_valid = None
    print("Computing test loss ... ")
    val(model, testData, criterion, tb_valid)


if __name__ == "__main__":
    main()
