#!/usr/bin/env python

import shutil
from scipy.io.matlab.mio import savemat, loadmat

import argparse
import os
from importlib import import_module
import torch
import numpy as np
import torch.nn as nn
import time

parser = argparse.ArgumentParser(description="Test Script for Real Image Denoising (sRGB)")
parser.add_argument("--model", type=str, default='ferm',
                    help="name of model for this training")
parser.add_argument("--checkpoint", type=str, default='checkpoints/srgb_fused.pth',
                    help="path to load model checkpoint")
parser.add_argument("--data_root", type=str, default='data/test/srgb',
                    help="root of the test data")
opt = parser.parse_args()
print(opt)

# load model
model = import_module('models.' + opt.model.lower()).make_model(opt)
model.load_state_dict(torch.load(opt.checkpoint))
model = nn.DataParallel(model)
model = model.cuda()
model = model.eval()


def np2Tensor(*args, rgb_range=1.):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(a) for a in args]


def denoiser(img_ori, index):
    img_ori = img_ori[:, :, ::-1]  # RGB to BGR
    img_hf = img_ori[:, ::-1, :]
    img_vf = img_ori[::-1, :, :]
    img_hvf = img_ori[::-1, ::-1, :]
    img_t = img_ori.transpose(1, 0, 2)
    img_hf_t = img_hf.transpose(1, 0, 2)
    img_vf_t = img_vf.transpose(1, 0, 2)
    img_hvf_t = img_hvf.transpose(1, 0, 2)

    img_ori, img_hf, img_vf, img_hvf, img_t, img_hf_t, img_vf_t, img_hvf_t = np2Tensor(img_ori, img_hf, img_vf, img_hvf,
                                                                                       img_t, img_hf_t, img_vf_t,
                                                                                       img_hvf_t)
    img = torch.stack((img_ori, img_hf, img_vf, img_hvf, img_t, img_hf_t, img_vf_t, img_hvf_t), dim=0)

    img = img.cuda()

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        output = model(img)
        torch.cuda.synchronize()
        end = time.time()

    print(index, end - start)

    output = output.detach().cpu().numpy().transpose((0, 2, 3, 1))
    out = output[0, :, :, :] + output[1, :, ::-1, :] + \
          output[2, ::-1, :, :] + output[3, ::-1, ::-1, :] + \
          output[4].transpose((1, 0, 2)) + output[5].transpose((1, 0, 2))[:, ::-1, :] + \
          output[6].transpose((1, 0, 2))[::-1, :, :] + output[7].transpose((1, 0, 2))[::-1, ::-1, :]

    out = out / 8.

    out = out * 255.
    out[out >= 255] = 255
    out[out <= 0] = 0
    out = out[:, :, ::-1]  # BGR to RGB
    return out


data_root = opt.data_root

# load noisy images
noisy_fn = 'siddplus_test_noisy_srgb.mat'
noisy_key = 'siddplus_test_noisy_srgb'
noisy_mat = loadmat(os.path.join(data_root, noisy_fn))[noisy_key]

# denoise
n_im, h, w, c = noisy_mat.shape
results = noisy_mat.copy()
for i in range(n_im):
    noisy = np.reshape(noisy_mat[i, :, :, :], (h, w, c))
    denoised = denoiser(noisy, i)
    results[i, :, :, :] = denoised

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join("results", "srgb", "results.mat")
res_key = 'results'
savemat(res_fn, {res_key: results})

