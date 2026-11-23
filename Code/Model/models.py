"""COSIME architectures: single view (continuous, binary) and multiple views
(binary, continuous with a regression or NN head; early or late fusion)."""
import torch
from torch import nn

class SingleContinuous(nn.Module):
    def __init__(self, input_dim, dim, dropout):
        super().__init__()

        self.dropout = dropout

        self.layers = nn.Sequential(
                # nn.Linear(input_dim, int(input_dim*0.75)),
                # nn.BatchNorm1d(int(input_dim*0.75)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                nn.Linear(int(input_dim), int(input_dim*0.5)),
                nn.BatchNorm1d(int(input_dim*0.5)),
                nn.LeakyReLU(),
                nn.Dropout(dropout),

                # nn.Linear(int(input_dim*0.5), int(input_dim*0.25)),
                # nn.BatchNorm1d(int(input_dim*0.25)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim*4, int(input_dim*5)),
                # nn.BatchNorm1d(int(input_dim*5)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim*5, int(input_dim*4)),
                # nn.BatchNorm1d(int(input_dim*4)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim*1.4), int(input_dim*1.3)),
                # nn.BatchNorm1d(int(input_dim*1.3)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim*1.3), int(input_dim*1.2)),
                # nn.BatchNorm1d(int(input_dim*1.2)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim*2), input_dim),
                # nn.BatchNorm1d(input_dim),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                nn.Linear(int(input_dim*0.5), dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout) 
                )

        self.mu = nn.Linear(dim, dim)
        self.logsigma = nn.Linear(dim, dim)

        self.Net = nn.Sequential(
            nn.Linear(dim, 1)
        )

    
    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(logsigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def net_forward(self, X):
            x = self.layers(X)
            x = x.view(x.size(0), -1)
            # mu, logsigma = self.mu(x), self.logsigma(x)
            mu = self.mu(x)
            logsigma = self.logsigma(x)
            z = self.gaussian_sampler(mu, logsigma)

            return z, mu, logsigma
    
    def regression_forward(self, X):
        return self.Net(X)
    
    def forward(self, X):

        # Obtain outputs from net_forward
        # net_logits = self.net_forward(X)
        z, mu, logsigma = self.net_forward(X)

        # Pass through logistic forward
        # NN_logits = self.logistic_forward(net_logits)
        probs = self.regression_forward(z)

        # Return converted tensors
        # return net_logits, NN_logits
        return z, mu, logsigma, probs

class SingleBinary(nn.Module):
    def __init__(self, input_dim, dim, dropout):
        super().__init__()

        self.dropout = dropout

        self.layers = nn.Sequential(
                # nn.Linear(input_dim, int(input_dim*0.75)),
                # nn.BatchNorm1d(int(input_dim*0.75)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                nn.Linear(int(input_dim), int(input_dim*0.5)),
                nn.BatchNorm1d(int(input_dim*0.5)),
                nn.Dropout(dropout),
                nn.LeakyReLU(),

                # nn.Linear(int(input_dim*0.5), int(input_dim*0.25)),
                # nn.BatchNorm1d(int(input_dim*0.25)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim*4, int(input_dim*5)),
                # nn.BatchNorm1d(int(input_dim*5)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim*5, int(input_dim*4)),
                # nn.BatchNorm1d(int(input_dim*4)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim*1.4), int(input_dim*1.3)),
                # nn.BatchNorm1d(int(input_dim*1.3)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim*1.3), int(input_dim*1.2)),
                # nn.BatchNorm1d(int(input_dim*1.2)),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim*2), input_dim),
                # nn.BatchNorm1d(input_dim),
                # nn.Dropout(dropout),
                # nn.LeakyReLU(),

                nn.Linear(int(input_dim*0.5), dim)
                )

        self.mu = nn.Linear(dim, dim)
        self.logsigma = nn.Linear(dim, dim)

        self.Net = nn.Sequential(
            nn.Linear(dim, 1)
        )

    
    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(logsigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def net_forward(self, X):
            x = self.layers(X)
            x = x.view(x.size(0), -1)
            # mu, logsigma = self.mu(x), self.logsigma(x)
            mu = self.mu(x)
            logsigma = self.logsigma(x)
            z = self.gaussian_sampler(mu, logsigma)

            return z, mu, logsigma
    
    def logistic_forward(self, X):
        logits = self.Net(X)
        probabilities = torch.sigmoid(logits)
        return probabilities
    
    def forward(self, X):

        # Obtain outputs from net_forward
        # net_logits = self.net_forward(X)
        z, mu, logsigma = self.net_forward(X)

        # Pass through logistic forward
        # NN_logits = self.logistic_forward(net_logits)
        probs = self.logistic_forward(z)

        # Return converted tensors
        # return net_logits, NN_logits
        return z, mu, logsigma, probs



# Binary early fusion model
class Binary_Early(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super().__init__()

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
        X = [x.float() for x in X]
        net_logits = self.net_forward(*X)

        if isinstance(net_logits, list) and all(isinstance(item, tuple) for item in net_logits):
            tensors = [item[0] for item in net_logits]
            stacked_logits = torch.stack(tensors, dim=0) 
            averaged_logits = stacked_logits.mean(dim=0) 
        else:
            raise TypeError("net_logits must be a list of tuples containing tensors")

        NN_logits = self.logistic_forward(averaged_logits)

        return net_logits, NN_logits

# Binary late fusion model
class Binary_Late(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super().__init__()

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

# Continuous early fusion model - regression
class Continuous_Early_Regression(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super().__init__()

        assert len(input_dims) > 0, 'Must provide at least one input dim.'

        self.num_modalities = len(input_dims)
        self.dropout = dropout

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(input_dim * 1), int(input_dim * 0.5)),
                nn.BatchNorm1d(int(input_dim * 0.5)),
                nn.Dropout(dropout),
                nn.LeakyReLU(),

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
             torch.nn.Linear(int(dim), 1),
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
    
    def NN_forward(self, X):
        return self.Net(X)
    
    def forward(self, *X):
        X = [x.float() for x in X]
        net_outputs = self.net_forward(*X)
        if isinstance(net_outputs, list) and all(isinstance(item, tuple) for item in net_outputs):
            tensors = [item[0] for item in net_outputs]
            stacked_outputs = torch.stack(tensors, dim=0)
            averaged_outputs = stacked_outputs.mean(dim=0)
        else:
            raise TypeError("net_outputs must be a list of tuples containing tensors")

        outputs = self.NN_forward(averaged_outputs)

        return net_outputs, outputs

# Continuous late fusion model - regression
class Continuous_Late_Regression(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super().__init__()

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
             torch.nn.Linear(int(dim), 1),
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
    
    def regression_forward(self, X):
        outputs = self.Net(X)
        return outputs
    
    def forward(self, *X):
        net_outputs = self.net_forward(*X)
        outputs = [self.regression_forward(nl[0]) for nl in net_outputs]

        return net_outputs, outputs

# Continuous early fusion model - Neural network
class Continuous_Early_NN(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super().__init__()

        # Check for input dims
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
             torch.nn.Linear(dim, int(dim*0.5)),
             torch.nn.BatchNorm1d(int(dim*0.5)),
             nn.LeakyReLU(),

             torch.nn.Linear(int(dim*0.5), 1),
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
    
    def NN_forward(self, X):
        outputs = self.Net(X)
        return outputs
    
    def forward(self, *X):
        X = [x.float() for x in X]
        net_outputs = self.net_forward(*X)

        if isinstance(net_outputs, list) and all(isinstance(item, tuple) for item in net_outputs):
            tensors = [item[0] for item in net_outputs]
            stacked_outputs = torch.stack(tensors, dim=0) 
            averaged_outputs = stacked_outputs.mean(dim=0)
        else:
            raise TypeError("net_outputs must be a list of tuples containing tensors")

        outputs = self.NN_forward(averaged_outputs)

        return net_outputs, outputs

# Continuous late fusion model - Neural network
class Continuous_Late_NN(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super().__init__()

        # Check for input dims
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
             torch.nn.Linear(dim, int(dim*0.5)),
             torch.nn.BatchNorm1d(int(dim*0.5)),
             nn.LeakyReLU(),

             torch.nn.Linear(int(dim*0.5), 1),
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
    
    def NN_forward(self, X):
        outputs = self.Net(X)
        return outputs
    
    def forward(self, *X):
        net_outputs = self.net_forward(*X)
        outputs = [self.NN_forward(nl[0]) for nl in net_outputs]

        return net_outputs, outputs

