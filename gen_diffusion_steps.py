# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: /Nerfusion-EG3D/gen_diffusion_steps.py
###   @Author: AceSix
###   @Date: 1969-12-31 19:00:00
###   @LastEditors: AceSix
###   @LastEditTime: 2022-12-15 17:54:42
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from multiprocessing import cpu_count
from Autoencoder.Networks import Autoencoder
from pathlib import Path
from torch import nn, einsum
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import dnnlib
from torch_utils import misc
from training.triplane import TriPlaneFeatureGenerator
from Autoencoder.Networks import Autoencoder
import numpy as np
import argparse
from diffusion_data import BottleDataset
import PIL
from torchvision import transforms as T, utils
import math
import os

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class DiffTrainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        config,
        *,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 16,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = config.batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.ds = BottleDataset(config.train_data_dir)
        dl = torch.utils.data.DataLoader(self.ds, batch_size = config.batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.config = config
        self.load_test_model()

        # optimizer
        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


    def load_test_model(self):
        print('Loading networks from "%s"...' % self.config.AE_dir)

        if config.model=="convnext20c6b":
            AE = Autoencoder(96, 8, [192, 512, 1024], [6,6,6])
        elif config.model=="convnext20c2b":
            AE = Autoencoder(96, 8, [512, 1024, 2048], [2,2,2])
        AE.load_state_dict(torch.load(self.config.AE_dir))
        self.decoder = AE.DecoderLayer.cuda()
        self.decoder.eval()

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

        angle_y = 0.4
        angle_p = -0.2

        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.5], device=device), radius=2.7, device=device)
        intrinsics = FOV_to_intrinsics(35.837, device=device)
        # intrinsics = FOV_to_intrinsics(18.837, device=device)

        cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        self.camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(self.G.z_dim) for seed in [1]])).to(device)
        self.ws = self.G.mapping(z=zs, c=conditioning_params, truncation_psi=1, truncation_cutoff=14)

    def generate_demo(self, content):
        content = content.view(len(content), 3, 32, content.shape[-2], content.shape[-1])
        num_image = content.shape[0]

        cols = int(math.sqrt(self.num_samples))
        imgs = [[] for c in range(cols)]
        for i in range(num_image):
            sr, raw = self.triplane_decode(content[i:i+1])
            imgs[i%cols].append(raw)
        img_cols = [torch.cat(c, dim=1) for c in imgs]
        return torch.cat(img_cols, dim=2)

    def triplane_decode(self, triplane):
        device = torch.device('cuda')

        decode_out = self.G.synthesis(triplane, self.camera_params, ws=self.ws)
        img = decode_out['image']
        img2 = decode_out['image_raw']
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img2 = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img, img2

    def gen_step_samples(self):
        batch, device = 1, torch.device('cuda')

        shape = (batch, self.model.channels, self.image_size, self.image_size)

        img = torch.randn(shape, device=device)

        x_start = None
        self_condition = self.model.self_condition

        for t in tqdm(reversed(range(0, self.model.num_timesteps)), desc = 'sampling loop time step', total = self.model.num_timesteps):
            self_cond = x_start if self_condition else None
            img, x_start = self.ema.ema_model.p_sample(img, t, self_cond)

            sr, raw = self.diff2img(img)
            PIL.Image.fromarray(sr[0].cpu().numpy(), 'RGB').save(f'{self.config.out_dir}/sample-sr-{t}.png')
            PIL.Image.fromarray(raw[0].cpu().numpy(), 'RGB').save(f'{self.config.out_dir}/sample-raw-{t}.png')

        triplane = self.diff2triplane(img)
        print(triplane.shape)
        torch.save(triplane, f'{self.config.out_dir}/final_triplane.pth')

    def diff2triplane(self, img):
        content = self.decoder(self.ds.denormalize(unnormalize_to_zero_to_one(img)))
        return content

    def diff2img(self, img):
        content = self.decoder(self.ds.denormalize(unnormalize_to_zero_to_one(img)))
        # print(content.shape)
        sr, raw = self.triplane_decode(content)
        # print(out.shape)
        return sr, raw
        # return img
    
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                            all_images = torch.cat(all_images_list, dim = 0)

                            content = self.decoder(self.ds.denormalize(all_images))
                            out = self.generate_demo(content)
                            print(out.shape)
                        PIL.Image.fromarray(out[0].cpu().numpy(), 'RGB').save(f'{self.results_folder}/sample-{milestone}.png')

                        # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)
                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')




def getParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_dir', type=str, default="/home/zliu177/Desktop/diffusion-EG3D/bottlenecks-1024.pth")
    parser.add_argument('--eg3d_dir', type=str, default="/home/zliu177/Desktop/Nerfusion-EG3D/afhqcats512-128.pkl")
    parser.add_argument('--AE_dir', type=str, default="/home/zliu177/Desktop/Nerfusion-EG3D/logs/convnext20c6b/model_state/120000_iter.pth")
    parser.add_argument('--version', type=str, default="Diffusion-c8")
    parser.add_argument('--model', type=str, default="convnext20c6b")
    parser.add_argument('--parallel', type=int, default=0)

    # AE training setting
    parser.add_argument('--iter_size', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=8e-5)

    # Path 
    parser.add_argument('--save_dir', type=str, default='/home/zliu177/Desktop/diffusion-EG3D/results')
    parser.add_argument('--out_dir', type=str, default='./cat_steps')
    parser.add_argument('--checkpoint', type=int, default=100)


    return parser.parse_args()


if __name__ == "__main__":
    config = getParameters()

    version_folder = os.path.join(config.save_dir, config.version)
    os.makedirs(version_folder, exist_ok=True)
    

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=8
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2'          # L1 or L2
    ).cuda()
    

    trainer = DiffTrainer(diffusion, config, results_folder=version_folder, train_num_steps=config.iter_size)
    trainer.load(config.checkpoint)
    os.makedirs(config.out_dir, exist_ok=True)
    trainer.gen_step_samples()