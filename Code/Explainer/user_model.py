# user_model.py

import torch
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super(Model, self).__init__()

        assert len(input_dims) > 0, 'Must provide at least one input dim.'

        self.num_modalities = len(input_dims)
        self.dropout = dropout

        self.layers = nn.ModuleList([
            nn.Sequential(

                nn.Linear(int(input_dim), int(input_dim*0.5)),
                nn.BatchNorm1d(int(input_dim*0.5)),
                nn.LeakyReLU(),
                nn.Dropout(dropout),

                nn.Linear(int(input_dim*0.5), dim)
            )
            for input_dim in input_dims
        ])
        
        self.mu = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in input_dims
        ])

        self.logsigma = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in input_dims
        ])

        self.Net = nn.Sequential(
            torch.nn.Linear(dim, 1)
            )
    
    def net_forward(self, *X):
        def process(x, layer):
            x = self.layers[layer](x)
            x = x.view(x.size(0), -1)
            mu = self.mu[layer](x)
            logsigma = self.logsigma[layer](x)
            z = self.gaussian_sampler(mu, logsigma)

            return z, mu, logsigma

        return [process(x, layer) for layer, x in enumerate(X)]

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(logsigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def logistic_forward(self, X):
        logits = self.Net(X)
        return logits
    
    def forward(self, *X):
        net_logits = self.net_forward(*X)
        logistic_logits = [self.logistic_forward(nl[0]) for nl in net_logits]

        return net_logits, logistic_logits
