import torch.nn as nn
import torch.nn.functional as F


class Fc(nn.Module):
    def __init__(self, feature, hidden1, hidden2, dropout):
        super(Fc, self).__init__()
        self.fc1 = nn.Linear(feature, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x
