# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/AE_train/triplane_dataset.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-12-02 16:05:35
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################

import torch
import torch.utils.data as data


class TriplainDataset(data.Dataset):
    def __init__(self, path):
        super(TriplainDataset, self).__init__()

        self.data = torch.load(path, map_location='cpu')
        self.data = self.data .view(len(self.data ), 96, self.data .shape[-2], self.data .shape[-1])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def name(self):
        return 'TriplainDataset'