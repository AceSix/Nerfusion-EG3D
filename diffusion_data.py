# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/home/zliu177/Desktop/diffusion-EG3D/diffusion_data.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-12-07 16:34:27
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
import torch
import torch.utils.data as data


class BottleDataset(data.Dataset):
    def __init__(self, path):
        super(BottleDataset, self).__init__()
        triplanes = torch.load(path, map_location='cpu')
        print(triplanes.min(), triplanes.max())
        self.low = triplanes.min()
        self.range = triplanes.max() - triplanes.min()
        self.data = (triplanes - triplanes.min()) / (triplanes.max() - triplanes.min())
        print(self.data.shape)

    def denormalize(self, data):
        return data*self.range + self.low

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def name(self):
        return 'TriplainDataset'