import torch
import torch.nn as nn

class WMC_Cell(nn.Module):
    def __init__(self,in_s,hidden_s):
        super().__init__()
        self.h=hidden_s

        self.wx=nn.Linear(in_s,4*hidden_s)
        self.wh=nn.Linear(hidden_s,4*hidden_s)

        self.ci=nn.Linear(hidden_s,hidden_s)
        self.cf=nn.Linear(hidden_s,hidden_s)
        self.co=nn.Linear(hidden_s,hidden_s)

    def forward(self,x,h_prev,c_prev):

        temp=self.wx(x)+self.wh(h_prev)
        i,f,g,o=temp.chunk(4,dim=1)

        i=torch.sigmoid(i+torch.tanh(self.ci(c_prev)))
        f=torch.sigmoid(f+torch.tanh(self.cf(c_prev)))

        g=torch.tanh(g)
        c=f*c_prev+i*g

        o=torch.sigmoid(o+torch.tanh(self.co(c)))

        h=o*torch.tanh(c)

        return h,c


class WMC_Net(nn.Module):
    def __init__(self,in_s,hidden_s):
        super().__init__()
        self.cell=WMC_Cell(in_s,hidden_s)
        self.out=nn.Linear(hidden_s,1)

    def forward(self,x):

        b,s,_=x.size()

        h=torch.zeros(b,self.cell.h)
        c=torch.zeros(b,self.cell.h)

        for t in range(s):
            h,c=self.cell(x[:,t,:],h,c)

        return self.out(h)