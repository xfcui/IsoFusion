#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: xfcui<xfcui@email.sdu.edu.cn>
# @Date: Mon 04 Apr 2022 09:06:01 PM HKT
# @Desc: models

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf

from IsoFusion.config import KERNEL_RT, KERNEL_MZ, MAX_CH, MAX_MZ, MAX_RT


class EmbedLayer(nn.Module):
    def __init__(self, width, width_scale=4):
        super().__init__()
        self.width = width

        self.block = nn.Sequential(nn.Conv3d(1, width*width_scale, [1, KERNEL_RT*2+1, KERNEL_MZ*2+1],
                                       stride=[1, 1, KERNEL_MZ*2+1], padding=0),
                         nn.GELU(),
                         nn.Conv3d(width*width_scale, width, 1))

    def forward(self, x):
        return self.block(x)


class ReshapeLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        x = x.reshape(x.shape[0], self.width, -1, *x.shape[2:])
        if x.shape[-3] == 1:
            x = pt.einsum('bhxiyz->bhxyz', x)
        elif x.shape[-2] == 1:
            x = pt.einsum('bhyxiz->bhxyz', x)
        elif x.shape[-1] == 1:
            x = pt.einsum('bhzxyi->bhxyz', x)
        else:
            raise Exception('Unknown shape:', x.shape)
        return x


class GateLayer(nn.Module):
    def __init__(self, num_group=4):
        super().__init__()
        self.num_group = num_group

        self.init0 = nn.parameter.Parameter(pt.zeros(num_group))

    def forward(self, x):
        xx = x.reshape(x.shape[0], self.num_group, -1, *x.shape[2:])
        xx = pt.einsum('bghxyz,g->bghxyz', xx, pt.exp(self.init0))
        xx = xx.reshape(x.shape)
        return xx


class NormLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

        self.norm_ch  = nn.Sequential(nn.Conv3d(width, width, 1), nn.GroupNorm(1, width, affine=False))
        self.norm_rt  = nn.Sequential(nn.Conv3d(width, width, 1), nn.GroupNorm(1, width, affine=False))
        self.norm_mz  = nn.Sequential(nn.Conv3d(width, width, 1), nn.GroupNorm(1, width, affine=False))

    def forward(self, x, l_ch=None):
        return self.norm_ch(x), self.norm_rt(x), self.norm_mz(x)


class ConvBlockRtMz(nn.Module):
    def __init__(self, width, width_scale=4, num_group=4):
        super().__init__()
        self.width = width

        self.norm = nn.Sequential(nn.Conv3d(width, width, 1),
                        nn.GroupNorm(num_group, width, affine=False))
        self.eye  = nn.Sequential(nn.Conv3d(width, width*width_scale, 1, stride=1, padding=0, groups=num_group),
                        nn.GELU())
        self.msg  = nn.Sequential(nn.Conv3d(width, width*width_scale, [1, KERNEL_RT*2-1, KERNEL_MZ*2-1],
                                      stride=1, padding=[0, KERNEL_RT-1, KERNEL_MZ-1], groups=num_group),
                        GateLayer(num_group),
                        nn.GELU())
        self.mix  = nn.Conv3d(width*width_scale, width, 1)

    def forward(self, x, wres=1.0):
        xx = self.norm(x)
        xx = self.eye(xx) + self.msg(xx)
        xx = self.mix(xx)
        return x + xx * wres


class FuseBlockCh(nn.Module):
    def __init__(self, width, num_group=4):
        super().__init__()
        self.width = width

        self.norm = nn.Sequential(nn.Conv3d(width, width, 1),
                        nn.GroupNorm(num_group, width, affine=False))
        self.eye  = nn.Sequential(nn.Conv3d(width, width, 1, stride=1, padding=0, groups=num_group),
                        nn.GELU())
        self.msg  = nn.Sequential(nn.Conv3d(width, width*MAX_CH, [MAX_CH, 1, 1], stride=1, padding=0, groups=num_group),
                        ReshapeLayer(width),
                        GateLayer(num_group),
                        nn.GELU())
        self.mix  = nn.Conv3d(width, width, 1)

    def forward(self, x, wres=1.0):
        xx = self.norm(x)
        xx = self.eye(xx) + self.msg(xx)
        xx = self.mix(xx)
        return x + xx * wres


class FuseBlockChMz(nn.Module):
    def __init__(self, width, num_group=4):
        super().__init__()
        self.width = width

        self.norm = nn.Sequential(nn.Conv3d(width, width, 1),
                        nn.GroupNorm(num_group, width, affine=False))
        self.eye  = nn.Sequential(nn.Conv3d(width, width, 1, stride=1, padding=0, groups=num_group),
                        nn.GELU())
        self.msg  = nn.Sequential(nn.Conv3d(width, width*MAX_MZ, [1, 1, MAX_MZ], stride=1, padding=0, groups=num_group),
                        ReshapeLayer(width),
                        nn.Conv3d(width, width*MAX_CH, [MAX_CH, 1, 1], stride=1, padding=0, groups=num_group),
                        ReshapeLayer(width),
                        GateLayer(num_group),
                        nn.GELU())
        self.mix  = nn.Conv3d(width, width, 1)

    def forward(self, x, wres=1.0):
        xx = self.norm(x)
        xx = self.eye(xx) + self.msg(xx)
        xx = self.mix(xx)
        return x + xx * wres


class DenseBlock(nn.Module):
    def __init__(self, width, width_scale=4, num_group=4):
        super().__init__()
        self.width = width

        self.dense = nn.Sequential(nn.Conv3d(width, width, 1),
                         nn.GroupNorm(num_group, width, affine=False),
                         nn.Conv3d(width, width*width_scale, 1, stride=1, padding=0, groups=num_group),
                         nn.GELU(),
                         nn.Conv3d(width*width_scale, width, 1))

    def forward(self, x, wres=1.0):
        return x + self.dense(x) * wres


class BaseHeadBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

        self.head_ch = nn.Linear(width*MAX_CH, MAX_CH)
        self.head_rt = nn.Linear(width*MAX_RT, MAX_RT)
        self.head_mz = nn.Linear(width*MAX_MZ, MAX_MZ)

    def forward(self, x_ch, x_rt, x_mz, l_ch=None):
        x_ch = pt.einsum('bhxyz->bhx', x_ch) / np.sqrt(x_ch.shape[-1] * x_ch.shape[-2])
        x_ch = self.head_ch(x_ch.reshape(x_ch.shape[0], -1))

        if self.training:
            if l_ch is None:  # training without label
                weight = nnf.softmax(x_ch, -1)
            else:             # training with label
                weight = pt.zeros_like(x_ch)
                weight.scatter_(-1, l_ch[:, None], 1)
        else:
            if l_ch is None:  # prediction with soft label
                weight = nnf.softmax(x_ch, -1)
            else:             # prediction with hard label
                l_ch = pt.argmax(x_ch, -1)
                weight = pt.zeros_like(x_ch)
                weight.scatter_(-1, l_ch[:, None], 1)

        x_rt = pt.einsum('bhxyz,bx->bhy', x_rt, weight) / np.sqrt(x_rt.shape[-1])
        x_rt = self.head_rt(x_rt.reshape(x_rt.shape[0], -1))
        x_mz = pt.einsum('bhxyz,bx->bhz', x_mz, weight) / np.sqrt(x_mz.shape[-2])
        x_mz = self.head_mz(x_mz.reshape(x_mz.shape[0], -1))

        return x_ch, x_rt, x_mz


class IsoFusion(nn.Module):
    def __init__(self, width, depth, width_scale=4, num_group=4, depth_scale=2, level_fuse=2):
        super().__init__()
        self.width = width
        self.depth = depth

        self.embed = EmbedLayer(width, width_scale)
        self.block = nn.ModuleList()
        for i in range(depth):
            for _ in range(depth_scale):
                self.block.append(ConvBlockRtMz(width, width_scale, num_group))
            if i == 0: continue
            if level_fuse == 0:
                pass
            elif level_fuse == 1:
                self.block.append(FuseBlockCh(width, num_group))
                self.block.append(DenseBlock(width, width_scale, num_group))
            elif level_fuse == 2:
                self.block.append(FuseBlockChMz(width, num_group))
                self.block.append(DenseBlock(width, width_scale, num_group))
            else:
                raise Exception('Unknown fusion level:', level_fuse)
        self.norm = NormLayer(width)
        self.head = BaseHeadBlock(width)

    def forward(self, x, l_ch=None):
        xx = self.embed(x)
        for b in self.block:
            xx = b(xx, 1 / self.depth)
        x_ch, x_rt, x_mz = self.head(*self.norm(xx), l_ch)
        return x_ch, x_rt, x_mz
