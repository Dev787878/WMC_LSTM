import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.wmc_lstm import WMC_Net
from models.lstm import BaseLSTM
from data.dataset_cls import make_data_cls

x,y=make_data_cls()

def train_model(model):

    loss_fn=nn.BCELoss()
    opt=optim.Adam(model.parameters(),lr=0.001)

    ep=300
    losses=[]
    accs=[]

    for i in range(ep):
        opt.zero_grad()

        pred=model(x)
        pred=torch.sigmoid(pred)

        loss=loss_fn(pred,y)

        loss.backward()
        opt.step()

        losses.append(loss.item())

        p=(pred>0.5).float()
        acc=(p==y).float().mean().item()
        accs.append(acc)

    return losses,accs


wmc_model=WMC_Net(1,32)
base_model=BaseLSTM(1,32)

loss_wmc,acc_wmc=train_model(wmc_model)
loss_base,acc_base=train_model(base_model)


plt.plot(loss_wmc,label="WMC")
plt.plot(loss_base,label="LSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Comparison")
plt.savefig("loss.png")
plt.clf()


plt.plot(acc_wmc,label="WMC")
plt.plot(acc_base,label="LSTM")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Comparison")
plt.savefig("accuracy.png")