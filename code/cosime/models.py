import torch
import torch.nn as nn

# Binary early fusion model
class Model(nn.Module):
    def __init__(self, *input_dims, dim, dropout, m_type, fusion, **kwargs):
        super().__init__()

        # Check for input dims
        assert len(input_dims) > 0, 'Must provide at least one input dim.'

        self.num_modalities = len(input_dims)
        self.dropout = dropout
        self.fusion = fusion

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(input_dim), int(input_dim*0.5)),
                nn.BatchNorm1d(int(input_dim*0.5)),
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
            torch.nn.Linear(dim, 1)
            )
    
    def net_forward(self, *X):
        def process(x, layer):
            x = self.layers[layer](x)
            x = x.view(x.size(0), -1)
            # mu, logsigma = self.mu(x), self.logsigma(x)
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
        probabilities = torch.sigmoid(logits)
        return probabilities
    
    def forward(self, *X):
        # [[0,1], [1,3], ]
        # Convert input tensors to Float if they are not already
        X = [x.float() for x in X]

        # Obtain outputs from net_forward
        net_logits = self.net_forward(*X)

        if self.fusion == 'early':
            # Ensure net_logits is a list of tuples
            if isinstance(net_logits, list) and all(isinstance(item, tuple) for item in net_logits):
                # Extract tensors from each tuple in the list
                tensors = [item[0] for item in net_logits]

                # Stack the tensors along a new dimension
                stacked_logits = torch.stack(tensors, dim=0)  # Shape [num_models, batch_size, num_features]

                # Average the logits along the new dimension
                averaged_logits = stacked_logits.mean(dim=0)  # Shape [batch_size, num_features]
            else:
                raise TypeError("net_logits must be a list of tuples containing tensors")
            
            # Pass through logistic forward
            NN_logits = self.logistic_forward(averaged_logits)

        elif self.fusion == 'late':
            NN_logits = [self.logistic_forward(nl[0]) for nl in net_logits]
            
        else: raise ValueError(f'Fusion {self.fusion} unsupported.')

        # Return converted tensors
        return net_logits, NN_logits
