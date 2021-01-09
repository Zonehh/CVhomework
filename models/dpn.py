'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels,dense_channels,stride,is_shortcut):
        super(Block,self).__init__()
        self.is_shortcut = is_shortcut
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=stride,padding=1,groups=32,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels,out_channels+dense_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels+dense_channels)
        )
        if self.is_shortcut:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels,out_channels+dense_channels,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(out_channels+dense_channels)
            )

    def forward(self, x):
        a = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_shortcut:
            a = self.shortcut(a)
        d = self.out_channels
        x = torch.cat([a[:,:d,:,:]+x[:,:d,:,:], a[:,d:,:,:], x[:,d:,:,:]], dim=1)
        x = self.relu(x)
        return x

class DPN(nn.Module):
    def __init__(self,cfg):
        super(DPN,self).__init__()
        mid_channels = cfg['mid_channels']
        out_channels = cfg['out_channels']
        num = cfg['num']
        dense_channels = cfg['dense_channels']
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(mid_channels[0], out_channels[0], dense_channels[0], num[0], stride=1)
        self.conv3 = self._make_layer(mid_channels[1], out_channels[1], dense_channels[1], num[1], stride=2)
        self.conv4 = self._make_layer(mid_channels[2], out_channels[2], dense_channels[2], num[2], stride=2)
        self.conv5 = self._make_layer(mid_channels[3], out_channels[3], dense_channels[3], num[3], stride=2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels[3]+(num[3]+1)*dense_channels[3], cfg['classes'])
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self,mid_channels,out_channels,dense_channels,num,stride):
        layers = []
        block_1 = Block(self.in_channels,mid_channels,out_channels,dense_channels,stride,is_shortcut=True)
        self.in_channels = out_channels + 2*dense_channels
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(self.in_channels,mid_channels,out_channels,dense_channels,stride=1,is_shortcut=False))
            self.in_channels = out_channels + (i+2)*dense_channels
        return nn.Sequential(*layers)

def DPN92():
    cfg = {
        'mid_channels': (96,192,384,768),
        'out_channels': (256,512,1024,2048),
        'num': (3,4,20,3),
        'dense_channels': (16,32,24,128),
        'classes': (10)
    }
    return DPN(cfg)