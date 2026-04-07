import torch
import torch.nn as nn
import torch.optim as optim

from models.wmc_lstm import WMC_Net
from data.dataset_reg import make_data_reg

x,y=make_data_reg()

model=WMC_Net(1,32)

loss_fn=nn.MSELoss()
opt=optim.Adam(model.parameters(),lr=0.001)

for i in range(300):
    opt.zero_grad()

    pred=model(x)
    loss=loss_fn(pred,y)

    loss.backward()
    opt.step()

    print("epoch",i+1,"loss",loss.item())