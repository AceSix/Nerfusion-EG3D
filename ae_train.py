# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Nerfusion-EG3D\ae_train.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-29 17:09:57
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################

import os
import torch
import  argparse
from torchvision.utils import save_image

from AE_train.model import AE_triplane
from Autoencoder.Networks import Autoencoder
from AE_train.triplane_dataset import TriplainDataset
from AE_train.utils import PSNR

class Trainer(object):
    def __init__(self, config):

        self.train_loader = torch.utils.data.DataLoader(
            TriplainDataset(config.train_data_dir),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        self.version_dir = f'{config.save_dir}/{config.version}'
        self.model_state_dir = f'{self.version_dir}/model_state'
        self.image_dir = f'{self.version_dir}/image'

        if not os.path.exists(self.version_dir):
            os.mkdir(self.version_dir)
            os.mkdir(self.model_state_dir)
            os.mkdir(self.image_dir)

        if config.model=="convnext":
            self.model = Autoencoder(96, 384, [192, 256, 384]).cuda()
        else:
            self.model = AE_triplane().cuda()

        self.config = config

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
                    torch.save(recon, os.path.join(self.image_dir, f"iter-{i}.pth"))
                else:
                    print(f"[iter]-[{i}]-[psnr]-[{round(psnr,2)}]-[mse]-[{round(mse,2)}]")



def getParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_dir', type=str, default="./features.pth")
    parser.add_argument('--version', type=str, default="simple AE")
    parser.add_argument('--model', type=str, default="simple")

    # AE training setting
    parser.add_argument('--iter_size', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=8)
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
