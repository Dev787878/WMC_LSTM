from train_compare import loss_wmc,loss_base,acc_base,acc_wmc

n=len(loss_wmc)

for i in range(n-5,n):
    print("epoch",i+1, "wmc_loss",loss_wmc[i], "base_loss",loss_base[i], "wmc_acc",acc_wmc[i], "base_acc",acc_base[i])