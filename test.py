import argparse
import torch
import utils
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import models
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--model', required=True, help='the path to the saved model')

opt = parser.parse_args()
print(opt)


def val(model, val_loader, criterion, tb_valid=None):
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
    print('Test: \tLoss: {:.6f}'.format(loss.data[0]))
    if tb_valid:
        tb_valid.add_scalar_dict(
            data={'loss': loss.data[0]})
    return loss


def dataLoad():
    feature_train, feature_test = utils.loadFeatures(train=False)
    _, feature_test = utils.preprocessFeatures(feature_train, feature_test)
    _, y_test = utils.loadUpvote(train=False)
    feature_test = torch.from_numpy(feature_test).float()
    y_test = torch.from_numpy(y_test).float()
    dataset_test = TensorDataset(feature_test, y_test)
    test_loader = DataLoader(dataset_test, batch_size=opt.batchSize,
                             shuffle=True, num_workers=1)
    print('test_data', len(test_loader.dataset))
    return test_loader, feature_test.size(1)


def main():
    test_loader, num_features = dataLoad()
    criterion = nn.MSELoss()
    checkpoint = torch.load(opt.model)
    model_opt = checkpoint['opt']
    print(model_opt)
    model = models.Fc(num_features, model_opt.hidden1, model_opt.hidden2,
                      model_opt.dropout)
    model.load_state_dict(checkpoint['model'])
    if opt.cuda:
        model.cuda()
    tb_valid = None
    val(model, test_loader, criterion, tb_valid)


if __name__ == "__main__":
    main()
