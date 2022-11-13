# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Nerfusion-EG3D\gen_features.py
###   @Author: AceSix
###   @Date: 2022-11-13 12:36:11
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-13 13:45:29
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


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------

def generate_features(
    network_pkl: str,
    seed: int,
    batch_size: int,
    truncation_psi: float,
    truncation_cutoff: int,
    fov_deg: float
):
    ### load network
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    G_new = TriPlaneFeatureGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate features.
    np.random.RandomState(seed)
    zs = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    features = []
    for i in range(batch_size):
        ws = G.mapping(zs[i:i+1], conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        features.append(G.gen_planes(ws)) 
    features = torch.cat(features, 0)
    return features



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_features("afhqcats512-128.pkl", 0, 5, 1, 14, 18.837) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
