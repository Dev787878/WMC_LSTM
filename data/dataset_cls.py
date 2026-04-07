import torch

def make_data_cls(n=1000,seq=10):
    x=torch.randn(n,seq,1)
    s=x.sum(dim=1)
    y=(s>0).float()
    return x,y