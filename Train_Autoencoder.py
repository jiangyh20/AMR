import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from AutomaticWeightedLoss import AutomaticWeightedLoss

from RDNet import rdnet,rdnetd

device = torch.device("cuda")
import scipy.io as sio
import numpy as np
import torch
import h5py

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

residual_num = 2
encoded_dim = 2048  

mat = h5py.File('/home/jyh/anaconda3/Pos1M2.mat')
data = np.transpose(mat['Hdata1']) 
data = data.astype('float32')
data = data+0.5
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(70%) and validation(30%)
np.random.shuffle(data)
start = floor(len(data)*0.7)
train, test = data[start:,:,:,:], data[:start,:,:,:]


BATCH_SIZE = 1024 // 4
from torch.utils.data import Dataset

import torch.utils.data as Data

train = torch.tensor(train)
test = torch.tensor(test)
torch_dataset = Data.TensorDataset(train, train)
trainloader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  
    num_workers=0,  
    drop_last=True
)
torch_dataset = Data.TensorDataset(test, test)
testloader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  
    num_workers=0, 
    drop_last=True
)

pretrained =1
lr = 1e-5


path1 = '/home/jyh/anaconda3/crnet.pt'
path2 = '/home/jyh/anaconda3/save.pt'

model = rdnetd(4).to(device)
if pretrained == 0:
    with open(path1,"w") as f:
        f.close()
if pretrained == 1:
    try:
        checkpoint = torch.load(path1)
    except:
        checkpoint = torch.load(path2)
    save_model = checkpoint['net']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
# location2


classifier_net = model
optimizer = torch.optim.Adam(classifier_net.parameters(),lr=lr)
if pretrained == 2:
    optimizer.load_state_dict(checkpoint['optimizer'])
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
last_testloss = 5e-3
criterion = nn.MSELoss(reduction='mean').to(device)


if pretrained == 1:
    epoch_start = checkpoint['epoch']
else:
    epoch_start = 0

temp_loss = 0.0
iteration = 0
temp = 0
iter = 0
TRAIN = 1
nmse4=np.zeros((0,))
nmse8=np.zeros((0,))
nmse16=np.zeros((0,))
nmse32=np.zeros((0,))

for epoch in range((epoch_start + 1), (epoch_start + 1000)):
    state_dict = {"net": classifier_net.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": epoch}
    torch.save(state_dict, path1)
    classifier_net.train()
    for i, data in enumerate(trainloader):
        pointk, label = data[0], data[1]
        torch.cuda.empty_cache()
        pointk, label = pointk.to(device), label.to(device)
        # init_data=init_data.to(device)
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        outputs = classifier_net(pointk)
        loss1 = torch.nn.MSELoss()(outputs, pointk)
        torch.cuda.empty_cache()
        if TRAIN == 1:
            classifier_net = classifier_net.to(device)
            classifier_net.zero_grad()

            loss =loss1
            loss = loss.to(device)

            loss.requires_grad_()
            loss.backward()
            # update paraeters in optimizer(update weigtht)
            optimizer.step()
        temp_loss += loss.cpu().item()
        iteration += 1
    scheduler.step(temp_loss)

    if epoch % 1 == 0 and epoch != 0:
        classifier_net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                outputs = classifier_net(images)
                testloss = criterion(outputs, images)
                x_ori = images
                x_hat = outputs
                x_real = torch.reshape(x_ori[:, 0, :, :], (len(x_ori), -1))-0.5
                x_imag = torch.reshape(x_ori[:, 1, :, :], (len(x_ori), -1))-0.5
                x_hat_real = torch.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))-0.5
                x_hat_imag = torch.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))-0.5
                power = torch.sum(x_real ** 2, 1) + torch.sum(x_imag ** 2, 1)
                mse = torch.sum(abs(x_real - x_hat_real) ** 2, 1) + torch.sum(abs(x_imag - x_hat_imag) ** 2, 1)
                nmse = 10 * torch.log10(torch.mean(torch.div(mse, power)))
                temp += nmse
                nmse4 = np.append(nmse4, nmse.cpu().detach().numpy())

                print('testloss1=%e' % testloss)
                print('NMSE1=%e' % nmse)

                np.savetxt('traindenseindense2.csv', nmse4, fmt='%.4f', delimiter=',')
                break
        # state_dict = {"net": classifier_net.state_dict(), "net2": awl.state_dict(), "optimizer": optimizer.state_dict(),
        #               "epoch": epoch}
    print('[epoch={0} lr={1}] mean_loss={2:e} '.format(epoch,
                                                                                     optimizer.state_dict()[
                                                                                         'param_groups'][0]['lr'],
                                                                                     temp_loss / iteration))
    temp_loss = 0
    iteration = 0
    temp = 0
    iter = 0
