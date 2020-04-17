import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv_19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_19, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(get_mask_19(in_channels, out_channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3), requires_grad=True)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, groups=1)
        return x

def get_mask_19(in_channels, out_channels, kernel_size=3):
    mask = np.zeros((out_channels, in_channels, 3, 3))
    for c in range(kernel_size):
        mask[:, :, c, c] = 1.
    return mask


class Conv_37(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_37, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(get_mask_37(in_channels, out_channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3), requires_grad=True)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, groups=1)
        return x

def get_mask_37(in_channels, out_channels, kernel_size=3):
    mask = np.zeros((out_channels, in_channels, 3, 3))
    for c in range(kernel_size):
        mask[:, :, 2-c, c] = 1.
    return mask


class SEACB(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(SEACB, self).__init__()

        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=1,
                                     padding=1, bias=False, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(3, 1), stride=1,
                                  padding=(1, 0), bias=False, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(1, 3), stride=1, 
                                  padding=(0, 1), bias=False, padding_mode=padding_mode)

    def forward(self, input):
        square_outputs = self.square_conv(input)
        vertical_outputs = self.ver_conv(input)
        horizontal_outputs = self.hor_conv(input)

        return square_outputs + vertical_outputs + horizontal_outputs


class EACB(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(EACB, self).__init__()


        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=1,
                                     padding=1, bias=False, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(3, 1), stride=1,
                                  padding=(1, 0), bias=False, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(1, 3), stride=1, 
                                  padding=(0, 1), bias=False, padding_mode=padding_mode)

        self.diag19_conv = Conv_19(in_channels=in_channels, out_channels=out_channels)

        self.diag37_conv = Conv_37(in_channels=in_channels, out_channels=out_channels)


    def forward(self, input):
        square_outputs = self.square_conv(input)
        vertical_outputs = self.ver_conv(input)
        horizontal_outputs = self.hor_conv(input)
        diag19_outputs = self.diag19_conv(input)
        diag37_outputs = self.diag37_conv(input)

        return square_outputs + vertical_outputs + horizontal_outputs + diag19_outputs + diag37_outputs


def default_conv(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=bias)


def seacb_conv(in_channels, out_channels, bias=False):
    return SEACB(in_channels, out_channels, padding_mode='zeros')

def eacb_conv(in_channels, out_channels, bias=False):
    return EACB(in_channels, out_channels, padding_mode='zeros')


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=[0.5, 0.5, 0.5], rgb_std=1.0, sign=-1, in_frames=1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor([rgb_std])
        rgb_means = []
        for i in range(in_frames):
            rgb_means.extend(rgb_mean)
        self.weight.data = torch.eye(in_frames*3).view(in_frames*3, in_frames*3, 1, 1) / std.view(1, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_means) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=False, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=False):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

