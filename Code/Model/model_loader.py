"""Build the COSIME model for a YAML configuration."""
from models import (SingleContinuous, SingleBinary, Binary_Early, Binary_Late,
                    Continuous_Early_Regression, Continuous_Late_Regression,
                    Continuous_Early_NN, Continuous_Late_NN)
import torch
from torch import nn

class ModelAdapter(nn.Module):
    def __init__(self, architecture, mode, outcome_type, fusion):
        super().__init__()
        self.architecture=architecture
        self.mode=mode
        self.outcome_type=outcome_type
        self.fusion=fusion
    def forward(self, xs):
        if self.mode == 'single':
            z,mu,lv,pred=self.architecture(xs[0])
            pred=pred.reshape(-1)
            if self.outcome_type=='binary':
                # SingleBinary returns probabilities; the loss uses logits.
                pred=torch.logit(pred.clamp(1e-6,1-1e-6))
            return [(z,mu,lv)],pred
        enc, raw=self.architecture(*xs)
        if isinstance(raw,list):
            pred=torch.stack([p.reshape(-1) for p in raw]).mean(dim=0)
        else:
            pred=raw.reshape(-1)
        return enc,pred
    def view_forward(self, j, x):
        """Late fusion: encoder and prediction head of view j for that view's own samples."""
        a=self.architecture
        h=a.layers[j](x.float())
        h=h.view(h.size(0),-1)
        mu,lv=a.mu[j](h),a.logsigma[j](h)
        z=a.gaussian_sampler(mu,lv)
        return (z,mu,lv),a.Net(z).reshape(-1)

def load_model(dims,cfg):
    model=cfg['model']; mode=cfg['data']['mode']; typ=model.get('type','continuous')
    predictor=model.get('predictor','regression'); fusion=model.get('fusion','early')
    dim=int(model.get('dim',100)); dropout=float(model.get('dropout',.5))
    if mode=='single':
        cls=SingleBinary if typ=='binary' else SingleContinuous
        arch=cls(dims[0],dim=dim,dropout=dropout)
        if predictor=='NN':
            # NN predictor: one hidden layer instead of the linear head.
            hidden=max(2,dim//2)
            arch.Net=nn.Sequential(nn.Linear(dim,hidden),nn.LeakyReLU(),nn.Linear(hidden,1))
    else:
        if typ=='binary':
            cls=Binary_Early if fusion=='early' else Binary_Late
        else:
            cls={('early','regression'):Continuous_Early_Regression,
                 ('late','regression'):Continuous_Late_Regression,
                 ('early','NN'):Continuous_Early_NN,
                 ('late','NN'):Continuous_Late_NN}[(fusion,predictor)]
        arch=cls(*dims,dim=dim,dropout=dropout)
        if typ=='binary' and predictor=='NN':
            # NN predictor: one hidden layer instead of the linear head.
            hidden=max(2,dim//2)
            arch.Net=nn.Sequential(nn.Linear(dim,hidden),nn.LeakyReLU(),nn.Linear(hidden,1))
    return ModelAdapter(arch,mode,typ,fusion)
