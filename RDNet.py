import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import os
from AutomaticWeightedLoss import AutomaticWeightedLoss
device = torch.device("cuda:0")

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out.mul(x)
        return out

class SE(nn.Module):
    def __init__(self, ch):
        super(SE, self).__init__()
        self.l1 = nn.AdaptiveAvgPool2d((1))
        if (ch > 4):
            self.l2 = ConvBN(ch, 4, 1)
            self.l3 = ConvBN(4, ch, 1)
        else:
            self.l2 = ConvBN(ch, 2, 1)
            self.l3 = ConvBN(2, ch, 1)
        self.SpatialAttentionModule=SpatialAttentionModule()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        x = self.l1(x)
        x = self.l2(x)
        x = Mish()(x)
        x = self.l3(x)
        x = nn.Sigmoid()(x)
        x = identity.mul(x)
        x=self.SpatialAttentionModule(x)
        return x

class RDBlock(nn.Module):
    def __init__(self):
        super(RDBlock, self).__init__()
        self.l0 = nn.Sequential(OrderedDict([
            ('conv3x3_1', ConvBN(2, 2, 5)),
            ('relu1', Mish()),

        ]))
        self.l1 = nn.Sequential(OrderedDict([
            ('conv3x3_2', ConvBN(2, 4, 1)),
            ('relu1', Mish()),
            ('conv3x3_3', ConvBN(4, 4, [1, 3])),
            ('relu1', Mish()),
            ('conv3x3_4', ConvBN(4, 4, [3, 1])),
            ('relu1', Mish()),
            ('conv3x3_5', ConvBN(4, 4, [1, 3])),
            ('relu1', Mish()),
            ('conv3x3_6', ConvBN(4, 4, [3, 1])),
            ('relu1', Mish()),
        ]))
        self.l2 = nn.Sequential(OrderedDict([
            ('conv3x3_7', ConvBN(6, 4, 5)),
            ('relu2', Mish()),
        ]))
        self.l3 = SE(10)
        self.l4 = nn.Sequential(OrderedDict([
            ('conv3x3_3', ConvBN(10, 2, [3, 3])),
            ('relu3', Mish()),
        ]))
        # self.autograd = torch.tensor(1.0, requires_grad=True)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut0 = x
        shortcut1 = self.l0(x)
        x = self.l1(x)
        shortcut2 = x
        x = torch.cat((shortcut1, shortcut2), dim=1)
        x = self.l2(x)
        shortcut3 = x
        x = torch.cat((shortcut1, shortcut2, shortcut3), dim=1)
        x = self.l3(x)

        # x=  self.l4(x)+self.autograd*shortcut1
        # x = self.l4(x) + self.autograd * shortcut0
        x= self.l4(x)
        # x= Mish()(x)
        return x


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if isinstance(kernel_size, int):
            padding = kernel_size//2
        else:
            padding = (kernel_size[0]//2,kernel_size[1]//2)
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Mish(torch.nn.Module):
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



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder0 = nn.Sequential(OrderedDict([
            ("decoder0_0", ConvBN(2, 2, 5))
        ]))
        self.decoder12 = nn.Sequential(OrderedDict([
            ("conv5x5_bn1", ConvBN(2, 32, [1, 5])),
            ("mish", Mish()),
            # ("mish", SE(32)),
            ("conv5x5_bn2", ConvBN(32, 2, [5, 1])),
            ("mish", Mish()),
            # ("mish", SE(2)),
            ("CRBlock2", RDBlock())
        ]))
        self.decoder22 = nn.Sequential(OrderedDict([
            ("conv5x5_bn1", ConvBN(2, 4, [1, 5])),
            ("mish", Mish()),
            # ("mish56", SE(4)),
            ("conv5x5_bn", ConvBN(4, 2, [5, 1])),
            ("mish", Mish()),
            # ("mish", SE(2)),
            ("CRBlock2", RDBlock())
        ]))
        self.decoder32 = nn.Sequential(OrderedDict([
            ("conv5x5_bn31", ConvBN(2, 32, 5)),
            ("mish", SE(32)),
            ("mish", Mish()),
            ("conv5x5_bn4", ConvBN(32, 2, 5)),
            ("mish", SE(2)),
            ("mish", Mish()),
            ("CRBlock2", RDBlock())
        ]))
        self.decoder42 = nn.Sequential(OrderedDict([
            ("conv5x5_bn31", ConvBN(2, 32, 5)),
            ("mish", SE(32)),
            ("mish", Mish()),
            ("conv5x5_bn4", ConvBN(32, 2, 5)),
            ("mish", SE(2)),
            ("mish", Mish()),
            ("CRBlock2", RDBlock())
        ]))
        self.con_final = SE(10)
        self.final1 = nn.Sequential(OrderedDict([
            ("conv1x1_bn_6", ConvBN(10, 32, 3)),
            ("mish", Mish()),
            ("mish_1", SE(32)),
            ("conv5x5_bn5", ConvBN(32, 2, 3)),
            ("mish", Mish()),
        ]))
        self.final2 = nn.Sequential(OrderedDict([
            ("conv1x1_bn_6", ConvBN(2, 32, 5)),
            ("mish", Mish()),
            ("mish_2", SE(32)),
            ("conv5x5_bn5", ConvBN(32, 2, 5)),
            ("mish", Mish()),
        ]))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, out):
        out = self.decoder0(out)
        # dense in dense
        shortcut0 = out

        out = self.decoder12(out)
        shortcut1 = out
        # out = torch.cat((shortcut0, shortcut1), dim=1)
        out = self.decoder22(out)
        shortcut2 = out
        # out = torch.cat((shortcut0, shortcut1, shortcut2), dim=1)
        out = self.decoder32(out)
        shortcut3 = out
        # out = torch.cat((out, shortcut0, shortcut1, shortcut2), dim=1)
        out = self.decoder42(out)
        out = torch.cat((out, shortcut0, shortcut1, shortcut2, shortcut3), dim=1)
        out = self.con_final(out)
        out = self.final1(out)
        out = out + shortcut0
        # out = self.final2(out)
        return out
class Decoderd(nn.Module):
    def __init__(self):
        super(Decoderd, self).__init__()
        self.decoder0 = nn.Sequential(OrderedDict([
            ("decoder0_0", ConvBN(2, 2, 5))
        ]))
        self.decoder12 = nn.Sequential(OrderedDict([
            ("conv5x5_bn1", ConvBN(2, 32, [1, 5])),
            ("mish", Mish()),
            # ("mish", SE(32)),
            ("conv5x5_bn2", ConvBN(32, 2, [5, 1])),
            ("mish", Mish()),
            # ("mish", SE(2)),
            ("CRBlock2", RDBlock())
        ]))
        self.decoder22 = nn.Sequential(OrderedDict([
            ("conv5x5_bn1", ConvBN(4, 4, [1, 5])),
            ("mish", Mish()),
            # ("mish56", SE(4)),
            ("conv5x5_bn", ConvBN(4, 2, [5, 1])),
            ("mish", Mish()),
            # ("mish", SE(2)),
            ("CRBlock2", RDBlock())
        ]))
        self.decoder32 = nn.Sequential(OrderedDict([
            ("conv5x5_bn31", ConvBN(6, 32, 5)),
            ("mish", SE(32)),
            ("mish", Mish()),
            ("conv5x5_bn4", ConvBN(32, 2, 5)),
            ("mish", SE(2)),
            ("mish", Mish()),
            ("CRBlock2", RDBlock())
        ]))
        self.decoder42 = nn.Sequential(OrderedDict([
            ("conv5x5_bn31", ConvBN(8, 32, 5)),
            ("mish", SE(32)),
            ("mish", Mish()),
            ("conv5x5_bn4", ConvBN(32, 2, 5)),
            ("mish", SE(2)),
            ("mish", Mish()),
            ("CRBlock2", RDBlock())
        ]))
        self.con_final = SE(10)
        self.final1 = nn.Sequential(OrderedDict([
            ("conv1x1_bn_6", ConvBN(10, 32, 3)),
            ("mish", Mish()),
            ("mish_1", SE(32)),
            ("conv5x5_bn5", ConvBN(32, 2, 3)),
            ("mish", Mish()),
        ]))
        self.final2 = nn.Sequential(OrderedDict([
            ("conv1x1_bn_6", ConvBN(2, 32, 5)),
            ("mish", Mish()),
            ("mish_2", SE(32)),
            ("conv5x5_bn5", ConvBN(32, 2, 5)),
            ("mish", Mish()),
        ]))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, out):
        out = self.decoder0(out)
        # dense in dense
        shortcut0 = out

        out = self.decoder12(out)
        shortcut1 = out
        out = torch.cat((shortcut0, shortcut1), dim=1)
        out = self.decoder22(out)
        shortcut2 = out
        out = torch.cat((shortcut0, shortcut1, shortcut2), dim=1)
        out = self.decoder32(out)
        shortcut3 = out
        out = torch.cat((out, shortcut0, shortcut1, shortcut2), dim=1)
        out = self.decoder42(out)
        out = torch.cat((out, shortcut0, shortcut1, shortcut2, shortcut3), dim=1)
        out = self.con_final(out)
        out = self.final1(out)
        out = out + shortcut0
        return out

class RDNet(nn.Module):
    def __init__(self,reduction,ifd):
        super(RDNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.ifd=ifd
        self.SpatialAttentionModule=SpatialAttentionModule()
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn_1", ConvBN(in_channel, 32, 3)),
            ("relu1", Mish()),
            ("conv1x9_bn_2", ConvBN(32, 32, [5, 5])),
            ("relu2", Mish()),
            ("conv9x1_bn_3", ConvBN(32, 2, 1)),

        ]))
        self.encoder2 = nn.Sequential(
            ConvBN(in_channel, 2, 3),
            Mish(),
            SE(2)
        )
        self.encoder3 = nn.Sequential(
            ConvBN(2, 32, 1),
            Mish(),
            ConvBN(32, 32, [1, 7]),
            Mish(),
            ConvBN(32, 32, [7, 1]),
            Mish(),
            ConvBN(32, 2, 5),

        )
        self.concat = ConvBN(6, 2, 5)
        self.switch= nn.Linear(total_size, total_size // reduction)

        self.switch2 = nn.Linear(total_size// reduction, total_size )
        if ifd==0:
            self.decoder = Decoder()
        if ifd==1:
            self.decoder = Decoderd()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # for p in self.parameters():
        #   if p!=
        #     p.requires_grad=False
    def forward(self,x):
        n, c, h, w = x.detach().size()
        x=self.SpatialAttentionModule(x)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        encode3 = self.encoder3(x)
        out = torch.cat((encode1, encode2, encode3), dim=1)
        out = self.concat(out)
        out = out.view(n,-1)
        out = self.switch(out)
        out = self.switch2(out).view(n,c,h,w)
        out = self.decoder(out)
        out = self.sigmoid(out)
        return out
def rdnet(r):
    model = RDNet(r,0)
    return model
def rdnetd(r):
    model = RDNet(r, 1)
    return model
