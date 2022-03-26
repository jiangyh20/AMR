import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch
import h5py
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
class Swish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)
        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)

class Enhanced_CE_loss(torch.nn.Module):
    def __init__(self,weight):
        super(Enhanced_CE_loss, self).__init__()
        self.weight=weight
    def forward(self, input, target):
        input =nn.Softmax()(input)
        loss = 0.0
        for i in range(input.shape[0]):
            gamma=self.weight
            x = torch.max(torch.log(input[i,target[i]]), torch.tensor([-1000.0]).to(device))
            y = torch.max(torch.log(torch.tensor([1,1,1,1]).to(device)-input[i,:]), torch.tensor([-1000.0]).to(device))
            l = -x +gamma[target[i]]*y[target[i]]- torch.dot(gamma,y)
            loss += l
        return loss



class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32*32*2,32*32),
            Swish(),
            nn.Linear(1024, 512),
            Swish(),
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, 128),
            Swish(),
            nn.Linear(128, 4),
        )
    def forward(self, x):
        output = self.layers(x)
        return output

device = torch.device("cuda")

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

residual_num = 2
encoded_dim = 2048  


# BATCH_SIZE=1024*4
BATCH_SIZE = 1024 // 4
a = np.loadtxt('gtruth.csv',delimiter=',')
print(a.shape)
mat = h5py.File('/home/jyh/anaconda3/Pos1M2.mat')
data = np.transpose(mat['Hdata1'])  # shape=(320000, 1024)
data = data.astype('float32')
data=data-0.5
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(70%) and validation(30%)
np.random.shuffle(data)
start = floor(len(data)*0.7)
train, test = data[start:,:,:,:], data[:start,:,:,:]

l=len(train)
b=torch.tensor(a)
torch_dataset = Data.TensorDataset(train, b)
trainloader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=0, 
    drop_last=False
)
torch_dataset = Data.TensorDataset(test, b)
testloader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  
    num_workers=0,  
    drop_last=False
)

pretrained = 1
lr = 7e-4
r1 = 2
r2 = 2
test_r1_1 = 2
test_r2_1 = 4
test_r1_2 = 4
test_r2_2 = 4
test_r1_3 = 4
test_r2_3 = 8

path1 = '/home/jyh/anaconda3/classify.pt'
path2 = '/home/jyh/anaconda3/save.pt'

model = myNet().to(device)

# awl = AutomaticWeightedLoss(4).to(device)
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



# count loss in an epoch
temp_loss = 0.0
if pretrained == 1:
    epoch_start = checkpoint['epoch']
else:
    epoch_start = 0
iteration = 0
TRAIN = 1
cri=np.zeros((0,))
real=np.zeros((0,))
anti=np.zeros((0,))
accu=np.zeros((0,))
weight_CE= torch.FloatTensor([1.0,1.05,1.1,1.15]).to(device)
for epoch in range((epoch_start + 1), (epoch_start + 2)):
    state_dict = {"net": classifier_net.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": epoch}
    torch.save(state_dict, path1)
    np.savetxt('cri.csv', cri, delimiter=',')
    print(np.shape(cri))
    cri = np.zeros((0,))
    # torch.save(state_dict, path2)
    classifier_net.train()
    num_correct=0
    real = np.zeros((0,))
    anti = np.zeros((0,))
    for i, data in enumerate(trainloader):
        pointk, label = data[0], data[1]
        torch.cuda.empty_cache()
        pointk, label = pointk.to(device), label.to(device)
        # init_data=init_data.to(device)
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        outputs = classifier_net(pointk)
        label =label.long().squeeze()
        # loss1 = F.cross_entropy(outputs, label)
        loss1 = Enhanced_CE_loss(weight=weight_CE)(outputs, label)
        torch.cuda.empty_cache()
        if TRAIN == 1:
            # back propagation
            classifier_net = classifier_net.to(device)
            classifier_net.zero_grad()
            loss =loss1
            loss = loss.to(device)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
        temp_loss += loss.cpu().item()
        iteration += 1
        prednp =outputs.argmax(dim=1)
        pred=prednp.cpu().detach().numpy()
        anti = np.append(anti,pred)
        real=np.append(real,label.cpu().detach().numpy())
        num_correct += torch.eq(prednp, label).sum().float().item()
        cri = np.append(cri,pred)
    # running_loss.append(temp_loss / iteration)
    scheduler.step(temp_loss / iteration)
    print('[epoch={0} lr={1}] accuracy={2} mean_loss={3:e} '.format(epoch,
                                                                                     optimizer.state_dict()[
                                                                                         'param_groups'][0]['lr'],
                                                                    (num_correct/l),
                                                                                     temp_loss / iteration))
    temp_loss = 0
    iteration = 0
