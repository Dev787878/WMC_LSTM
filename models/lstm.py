import torch
import torch.nn as nn

class BaseLSTM(nn.Module):
    def __init__(self,in_s,hidden_s):
        super().__init__()
        self.l=nn.LSTM(in_s,hidden_s,batch_first=True)
        self.o=nn.Linear(hidden_s,1)

    def forward(self,x):
        y,_=self.l(x)
        y=y[:,-1,:]
        return self.o(y)