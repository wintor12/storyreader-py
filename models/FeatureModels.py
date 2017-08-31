import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    def __init__(self, feature, opt):
        super(Base, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(feature, opt.hidden1)
        self.fc2 = nn.Linear(opt.hidden1, opt.hidden2)
        self.fc3 = nn.Linear(opt.hidden2, 1)
        self.dropout = nn.Dropout(opt.dropout)
        if opt.batchnorm:
            self.batch_norm1 = nn.BatchNorm1d(opt.hidden1)
            self.batch_norm2 = nn.BatchNorm2d(opt.hidden2)

    def forward(self, input):
        if self.opt.batchnorm:
            x = F.relu(self.batch_norm1(self.fc1(input)))
            x = self.dropout(x)
            x = F.relu(self.batch_norm2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(input))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x
