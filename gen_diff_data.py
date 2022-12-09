# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/gen_diff_data.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-12-07 16:23:50
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/gen_features.py
###   @Author: AceSix
###   @Date: 2022-11-13 12:36:11
###   @LastEditors: AceSix
###   @LastEditTime: 2022-12-05 13:34:01
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

# import click
import dnnlib
import numpy as np
# import PIL.Image
import torch
# from tqdm import tqdm
# import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator, TriPlaneFeatureGenerator
from Autoencoder.Networks import Autoencoder


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------

def generate_features(
    network_pkl: str
):
    ### load network
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    model = Autoencoder(96, 8, [192, 512, 1024], [6,6,6]).to(device)
    model.load_state_dict(torch.load(network_pkl))

    features = torch.load("features-1024.pth")
    features = features.view(len(features), 96, features.shape[-2], features.shape[-1])

    bottlenecks = []
    with torch.no_grad():
        for i in range(int(1024/4)):
            bottleneck = model.EncoderLayer(features[i*4:(i+1)*4].to(device))
            bottlenecks.append(bottleneck)
    
    return torch.cat(bottlenecks, 0)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    with torch.no_grad():
        features = generate_features("logs/convnext20c6b/model_state/120000_iter.pth") # pylint: disable=no-value-for-parameter
        print(features.shape)
        torch.save(features, "bottlenecks-1024.pth")


#----------------------------------------------------------------------------
