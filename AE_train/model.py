# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Nerfusion-EG3D\AE_train\model.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-29 21:15:29
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/AE_train/model.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-28 19:52:12
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class Cov(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=2,bias=False,activation=nn.SELU()):
        super(Cov, self).__init__()
        padding_size=1
        self.conv=nn.Sequential(
            nn.ReflectionPad2d((padding_size)),
            nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride, bias=bias),
            activation
        )
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if bias:
                    nn.init.zeros_(layer.bias)

    def forward(self,x):
        y=self.conv(x)
        return y

class DeCov(nn.Module):
    def __init__(self,in_channels,out_channels,factor=2,kernel_size=3,stride=1,bias=False,activation=nn.SELU()):
        super(DeCov, self).__init__()
        padding_size=(kernel_size-1)//2
        self.dconv=nn.Sequential(
            nn.Upsample(scale_factor=factor), 
            nn.ReflectionPad2d((padding_size)),
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=bias),
            # nn.InstanceNorm2d(out_channels),
            activation
        )
        for layer in self.dconv:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if bias:
                    nn.init.zeros_(layer.bias)

    def forward(self,x):
        y=self.dconv(x)
        return y


class GenResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, activation=F.selu):
        super(GenResBlock, self).__init__()

        self.activation = activation
        if h_ch is None:
            h_ch = out_ch
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        # self.b1 = nn.InstanceNorm2d(in_ch,affine=True)
        # self.b2 = nn.InstanceNorm2d(h_ch,affine=True)

    def forward(self, x):
        return x + self.residual(x)

    def residual(self, x):
        h = self.c1(x)
        # h = self.b1(h)
        h = self.activation(h)
        h = self.c2(h)
        # h = self.b2(h)
        h = self.activation(h)
        return h


act_dict = {
    'relu':[F.relu, nn.ReLU()],
    'selu':[F.selu, nn.SELU()]
}
class AE_triplane(nn.Module):
    def __init__(self, activation='selu'):
        super().__init__()

        self.ec1 = Cov(96, 128, activation=act_dict[activation][1])
        self.ec2 = Cov(128, 256, activation=act_dict[activation][1])
        self.ec3 = Cov(256, 384, activation=act_dict[activation][1])
        self.rb1 = GenResBlock(384, 384, activation=act_dict[activation][0])
        self.rb2 = GenResBlock(384, 384, activation=act_dict[activation][0])
        self.rb3 = GenResBlock(384, 384, activation=act_dict[activation][0])
        self.rb4 = GenResBlock(384, 384, activation=act_dict[activation][0])
        self.rb5 = GenResBlock(384, 384, activation=act_dict[activation][0])
        self.dc1 = DeCov(384, 256, 2, activation=act_dict[activation][1])
        self.dc2 = DeCov(256, 128, 2, activation=act_dict[activation][1])
        self.dc3 = DeCov(128, 96, 2, activation=act_dict[activation][1])
        self.cout = nn.Conv2d(96, 96, (1, 1), 1, bias=True)

    def forward(self, features):
        h = self.ec1(features)
        h = self.ec2(h)
        h = self.ec3(h)
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        h = self.rb4(h)
        h = self.rb5(h)
        h = self.dc1(h)
        h = self.dc2(h)
        h = self.dc3(h)
        h = self.cout(h)
        return h