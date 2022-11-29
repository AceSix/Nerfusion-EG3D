# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/AE_train/utils.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-28 20:07:38
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
import numpy as np
import math

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100, mse
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), mse