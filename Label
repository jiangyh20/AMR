import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
device = torch.device("cuda")
import scipy.io as sio
import numpy as np
import h5py
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from AutomaticWeightedLoss import AutomaticWeightedLoss
from RDNet import rdnetd
m1=rdnetd(4)
m2=rdnetd(8)
m3=rdnetd(16)
m4=rdnetd(32)
path1 = '/home/jyh/anaconda3/save1.1.pt'
path2 = '/home/jyh/anaconda3/save1.2.pt'
path3 = '/home/jyh/anaconda3/save1.3.pt'
path4 = '/home/jyh/anaconda3/save1.4.pt'

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

mat = h5py.File('/home/jyh/anaconda3/Pos1M2.mat')
data = np.transpose(mat['Hdata1'])
data = data.astype('float32')


data = data+0.5
l=len(data)
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
data = torch.tensor(data)
data=data.to(device)

from torch.utils.data import Dataset

import torch.utils.data as Data
BATCH_SIZE =1024//8
torch_dataset = Data.TensorDataset(data, data)

trainloader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # shuffle的英文含义是洗牌，=true意味着每回顺序是打乱的
    num_workers=0,  # 2个“工人”
    drop_last=False
)

def nmse(images,outputs):
    x_ori = images
    x_hat = outputs
    x_real = torch.reshape(x_ori[:, 0, :, :], (len(x_ori), -1))-0.5
    x_imag = torch.reshape(x_ori[:, 1, :, :], (len(x_ori), -1))-0.5
    x_hat_real = torch.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))-0.5
    x_hat_imag = torch.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))-0.5
    power = torch.sum(x_real ** 2, 1) + torch.sum(x_imag ** 2, 1)
    mse = torch.sum(abs(x_real - x_hat_real) ** 2, 1) + torch.sum(abs(x_imag - x_hat_imag) ** 2, 1)
    # nmse = 10 * torch.log10(torch.mean(torch.div(mse, power)))
    nmse = 10 * torch.log10(torch.div(mse, power))
    nmse = nmse.cpu().detach()
    return nmse

print(data.shape)
nmse4=np.zeros((0,))
nmse8=np.zeros((0,))
nmse16=np.zeros((0,))
nmse32=np.zeros((0,))


model1 = m1.to(device)
checkpoint = torch.load(path1)
model_dict = checkpoint['net']
model1.load_state_dict(model_dict)

model2 = m2.to(device)
checkpoint = torch.load(path2)
model_dict = checkpoint['net']
model2.load_state_dict(model_dict)

model3 = m3.to(device)
checkpoint = torch.load(path3)
model_dict = checkpoint['net']
model3.load_state_dict(model_dict)

model4 = m4.to(device)
checkpoint = torch.load(path4)
model_dict = checkpoint['net']
model4.load_state_dict(model_dict)



label=1
stan=-10
iter=0
gtruth=np.zeros(l,dtype=int)
cri =np.zeros(l,dtype=float)
for i, data1 in enumerate(trainloader):
    pointk, images = data1[0], data1[1]
    outputs = model1(pointk)
    temp4 = nmse(images,outputs).cpu().detach().numpy()
    nmse4 = np.append(nmse4, nmse(images,outputs).cpu().detach().numpy())
    torch.cuda.empty_cache()

    outputs = model2(pointk)
    temp8 = nmse(images, outputs).cpu().detach().numpy()
    nmse8 = np.append(nmse8, nmse(images,outputs).cpu().detach().numpy())
    torch.cuda.empty_cache()

    outputs = model3(pointk)
    temp16 = nmse(images, outputs).cpu().detach().numpy()
    nmse16 = np.append(nmse16, nmse(images, outputs).cpu().detach().numpy())
    torch.cuda.empty_cache()

    outputs = model4(pointk)
    temp32 = nmse(images, outputs).cpu().detach().numpy()
    nmse32 = np.append(nmse32, nmse(images,outputs).cpu().detach().numpy())
    torch.cuda.empty_cache()

if label == 1:
    for i in range(l):
        if nmse32[i] < stan:
            gtruth[i] = 3
            nmse8[i]=5
            nmse16[i] = 5
            nmse4[i] = 5
            # cri[i]=1

        elif nmse16[i]< stan:
            gtruth[i] = 2
            nmse8[i] = 5
            nmse4[i] = 5
            nmse32[i] = 5
            # cri[i] = 2
        elif nmse8[i] < stan:
            gtruth[i] = 1
            nmse4[i] = 5
            nmse16[i] = 5
            nmse32[i] = 5
        else :
            gtruth[i] = 0
            nmse8[i] = 5
            nmse16[i] = 5
            nmse32[i] = 5
    cri[i]=nmse4[i]+nmse8[i]+nmse16[i]+nmse32[i]


if label == 2:
    cri = np.loadtxt('cri.csv',delimiter=',')
    cri = cri.astype('int')
    print (len(cri))
    for i in range(l):
        if cri[i]==3:
            nmse8[i]=5
            nmse16[i] = 5
            nmse4[i] = 5

        elif cri[i]==2:

            nmse8[i] = 5
            nmse4[i] = 5
            nmse32[i] = 5

        elif cri[i]==1:

            nmse4[i] = 5
            nmse16[i] = 5
            nmse32[i] = 5
        else :
            nmse8[i] = 5
            nmse16[i] = 5
            nmse32[i] = 5

if label ==1:
    np.savetxt('sparsity.csv', save, delimiter=',')
    np.savetxt('gtruth.csv', gtruth, delimiter=',')

np.savetxt('nmse4.csv',nmse4,delimiter=',')
np.savetxt('nmse8.csv',nmse8,delimiter=',')
np.savetxt('nmse16.csv',nmse16,delimiter=',')
np.savetxt('nmse32.csv',nmse32,delimiter=',')
