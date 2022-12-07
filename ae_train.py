# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /AE_test/home/zliu177/Desktop/Nerfusion-EG3D/ae_train.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-12-05 16:57:35
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################

import os
import torch
import argparse
import numpy as np
from torchvision.utils import save_image
import PIL.Image

from AE_train.model import AE_triplane
from Autoencoder.Networks import Autoencoder
from AE_train.triplane_dataset import TriplainDataset
from AE_train.utils import PSNR


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import dnnlib
from torch_utils import misc
from training.triplane import TriPlaneFeatureGenerator
from Autoencoder.Networks import Autoencoder

class Trainer(object):
    def __init__(self, config):

        self.train_loader = torch.utils.data.DataLoader(
            TriplainDataset(config.train_data_dir),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        self.test_samples = torch.load("features-16-test.pth")

        self.version_dir = f'{config.save_dir}/{config.version}'
        self.model_state_dir = f'{self.version_dir}/model_state'
        self.image_dir = f'{self.version_dir}/image'

        if not os.path.exists(self.version_dir):
            os.mkdir(self.version_dir)
            os.mkdir(self.model_state_dir)
            os.mkdir(self.image_dir)

        if config.model=="convnext":
            self.model = Autoencoder(96, 384, [192, 256, 384])
        elif config.model=="convnext8x":
            self.model = Autoencoder(96, 384, [192, 256, 384, 384], [2, 2, 2, 2])
        elif config.model=="convnext8x12c":
            self.model = Autoencoder(96, 32, [512, 1024, 2048, 2048], [2, 2, 2, 2])
        elif config.model=="convnext4c":
            self.model = Autoencoder(96, 96, [192, 384, 512])
        elif config.model=="convnext8c4b":
            self.model = Autoencoder(96, 64, [192, 512, 1024], [4,4,4])
        elif config.model=="convnext12c4b":
            self.model = Autoencoder(96, 32, [192, 512, 1024], [4,4,4])
        elif config.model=="convnext16c6b":
            self.model = Autoencoder(96, 16, [192, 512, 1024], [6,6,6])
        elif config.model=="convnext20c6b":
            self.model = Autoencoder(96, 8, [192, 512, 1024], [6,6,6])
        else:
            self.model = AE_triplane()

        if torch.cuda.device_count() > 1 and config.parallel:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.config = config

        self.load_test_model()

    def load_test_model(self):
        print('Loading networks from "%s"...' % self.config.eg3d_dir)
        device = torch.device('cuda')
        with dnnlib.util.open_url(self.config.eg3d_dir) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

        print("Reloading Modules!")
        G_new = TriPlaneFeatureGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        self.G = G_new

    def generate_demo(self, content, recon):
        content = content.view(len(content), 3, 32, content.shape[-2], content.shape[-1])
        recon = recon.view(len(recon), 3, 32, recon.shape[-2], recon.shape[-1])
        device = torch.device('cuda')
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        intrinsics = FOV_to_intrinsics(18.837, device=device)

        num_image = content.shape[0]
        angle_p = -0.2

        cols = self.config.batch_size//4
        imgs = [[] for c in range(cols)]
        for i in range(num_image):
            for angle_y, angle_p in [(.4, angle_p)]:
                cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
                img = self.G.synthesis(content[i:i+1], camera_params)['image_raw']
                rec = self.G.synthesis(recon[i:i+1], camera_params)['image_raw']
    
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                rec = (rec.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs[i%cols].append(torch.cat([img, rec], dim=2))
        col_imgs = []
        for i in range(cols):
            col_imgs.append(torch.cat(imgs[i], dim=1))
        return torch.cat(col_imgs, dim=2)
    

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.SmoothL1Loss()

        contents = iter(self.train_loader)
        for i in range(0, self.config.iter_size+1):
            try:
                content = next(contents)
            except:
                contents = iter(self.train_loader)
                content = next(contents)

            content = content.cuda()
            recon = self.model(content)

            loss = criterion(content, recon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%self.config.log_interval == 0:
                with torch.no_grad():
                    psnr, mse = PSNR(content.cpu().numpy(),recon.cpu().numpy())
                if i%(self.config.log_interval*10) == 0:
                    torch.save(self.model.state_dict(), f'{self.model_state_dir}/{i}_iter.pth')
                    print(f"[iter]-[{i}]-[psnr]-[{round(psnr,2)}]-[mse]-[{round(mse,2)}]-[checkpoint]")
                    with torch.no_grad():
                        self.model.eval()
                        content = self.test_samples.cuda()
                        content = content.view(len(content), 96, content.shape[-2], content.shape[-1])
                        recon = self.model(content)
                        out = self.generate_demo(content, recon)
                        self.model.train()
                        PIL.Image.fromarray(out[0].cpu().numpy(), 'RGB').save(f'{self.image_dir}/{i}_iter.png')
                else:
                    print(f"[iter]-[{i}]-[psnr]-[{round(psnr,2)}]-[mse]-[{round(mse,2)}]")



def getParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_dir', type=str, default="./features-128.pth")
    parser.add_argument('--eg3d_dir', type=str, default="./afhqcats512-128.pkl")
    parser.add_argument('--version', type=str, default="simple AE")
    parser.add_argument('--model', type=str, default="simple")
    parser.add_argument('--parallel', type=int, default=0)

    # AE training setting
    parser.add_argument('--iter_size', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # Path 
    parser.add_argument('--save_dir', type=str, default='./logs')


    return parser.parse_args()


if __name__ == "__main__":
    config = getParameters()
    trainer = Trainer(config)
    trainer.train()
