import torch.nn as nn
import torch.nn.functional as F


class ResidualModel(nn.Module):
    def __init__(self, feature_model, text_model):
        super(ResidualModel, self).__init__()
        self.feature_model = feature_model
        self.text_model = text_model

    def forward(self, batch):
        y1 = self.feature_model(batch.feature[:, :2])
        y2 = self.text_model(batch)
        y = F.relu(y1 + y2)
        return y
