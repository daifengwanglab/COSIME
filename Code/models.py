# models.py

import torch
import torch.nn as nn

# Binary early fusion model
class ModelBinaryEarly(nn.Module):
    def __init__(self, dim, dropout):
        super(ModelBinaryEarly, self).__init__()
        self.dropout = dropout
        self.layer1 = nn.Linear(dim, dim // 2)
        self.layer2 = nn.Linear(dim // 2, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout_layer(self.layer1(x))
        x = self.layer2(x)
        return x

# Binary late fusion model
class ModelBinaryLate(nn.Module):
    def __init__(self, dim, dropout):
        super(ModelBinaryLate, self).__init__()
        self.layer1 = nn.Linear(dim, dim // 2)
        self.layer2 = nn.Linear(dim // 2, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.dropout_layer(self.layer1(x1))
        x2 = self.dropout_layer(self.layer1(x2))
        x = torch.cat((x1, x2), dim=1)
        x = self.layer2(x)
        return x

# Continuous early fusion model
class ModelContinuousEarly(nn.Module):
    def __init__(self, dim, dropout):
        super(ModelContinuousEarly, self).__init__()
        self.layer1 = nn.Linear(dim, dim // 2)
        self.layer2 = nn.Linear(dim // 2, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout_layer(self.layer1(x))
        x = self.layer2(x)
        return x

# Continuous late fusion model
class ModelContinuousLate(nn.Module):
    def __init__(self, dim, dropout):
        super(ModelContinuousLate, self).__init__()
        self.layer1 = nn.Linear(dim, dim // 2)
        self.layer2 = nn.Linear(dim // 2, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.dropout_layer(self.layer1(x1))
        x2 = self.dropout_layer(self.layer1(x2))
        x = torch.cat((x1, x2), dim=1)
        x = self.layer2(x)
        return x
