import torch
import argparse
import os
from data.default_dataset import get_eval_loader
from pytorch_msssim import ssim, ms_ssim
import pickle
import utils.hand_utils as handutils
import numpy as np


def crop_image(img0, img1, pts):
    uvis = pts[:, 0]
    vvis = pts[:, 1]

    umin = min(uvis)
    vmin = min(vvis)
    umax = max(uvis)
    vmax = max(vvis)

    B = round(1.5 * max([umax - umin, vmax - vmin]))

    us = max((umin + umax - B) / 2, 0)
    ue = min((umin + umax + B) / 2, 255)
    vs = max((vmin + vmax - B) / 2, 0)
    ve = min((vmin + vmax + B) / 2, 255)

    us = int(us)
    vs = int(vs)
    ue = int(ue)
    ve = int(ve)

    img0 = img0[:, :, vs:ve + 1, us:ue + 1]
    img1 = img1[:, :, vs:ve + 1, us:ue + 1]
    return img0, img1


@torch.no_grad()
def calculate_ssim_given_paths(paths, img_size=256, batch_size=1):
    print('Calculating SSIM given paths %s and %s...' % (paths[0], paths[1]))
    loaders = [get_eval_loader(path, img_size, batch_size, shuffle=False, drop_last=False, num_workers=0) for path in paths]
    loaders_iter = [iter(loader) for loader in loaders]
    num_clips = (len(os.listdir(paths[0])) - 1) // batch_size + 1
    ssim_values = []
    msssim_values = []

    for i in range(num_clips):
        img0 = loaders_iter[0].next().cuda()
        img1 = loaders_iter[1].next().cuda()

        ssim_val = ssim(img0, img1, data_range=255, size_average=False)  # return (N,)
        ms_ssim_val = ms_ssim(img0, img1, data_range=255, size_average=False)  # (N,)
        ssim_values.append(ssim_val)
        msssim_values.append(ms_ssim_val)
    ssim_values = torch.cat(ssim_values, dim=0)
    msssim_values = torch.cat(msssim_values, dim=0)
    ssim_mean = torch.mean(ssim_values)
    msssim_mean = torch.mean(msssim_values)

    return ssim_mean, msssim_mean

if __name__ == '__main__':
    path1 = 'imitators/'
    path2 = 'gt/'

    ssim = calculate_ssim_given_paths([path1, path2])
    print(ssim)