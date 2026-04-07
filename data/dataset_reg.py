import torch

def make_data_reg(n=1000,seq=10):
    x=torch.randn(n,seq,1)
    y=x.sum(dim=1)
    return x,y