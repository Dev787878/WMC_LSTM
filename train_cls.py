import torch
import torch.nn as nn
import torch.optim as optim

from models.wmc_lstm import WMC_Net
from data.dataset_cls import make_data_cls

x,y=make_data_cls()

model=WMC_Net(1,32)

loss_fn=nn.BCELoss()
opt=optim.Adam(model.parameters(),lr=0.001)

for i in range(300):
    opt.zero_grad()

    pred=model(x)
    pred=torch.sigmoid(pred)

    loss=loss_fn(pred,y)

    loss.backward()
    opt.step()

    acc=((pred>0.5).float()==y).float().mean()

    print("epoch",i+1,"loss",loss.item(),"acc",acc.item())