# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/Autoencoder/FusedOperators.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-29 17:25:35
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
import torch
import torch.nn as nn
import math
from torch_utils.ops import bias_act

class BiasedActivationReference(nn.Module):
    Gain = math.sqrt(2)
    Function = nn.functional.silu
    
    def __init__(self, InputUnits):
        super(BiasedActivationReference, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
    def forward(self, x):
        y = x + self.Bias.view(1, -1, 1, 1) if len(x.shape) > 2 else x + self.Bias.view(1, -1)
        return BiasedActivationReference.Function(y)

class BiasedActivationCUDA(nn.Module):
    Gain = math.sqrt(2)
    Function = 'swish'
    
    def __init__(self, InputUnits):
        super(BiasedActivationCUDA, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
    def forward(self, x):
        return bias_act.bias_act(x, self.Bias, act=BiasedActivationCUDA.Function, gain=1)

BiasedActivation = BiasedActivationReference
