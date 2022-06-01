'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .impl.ses_conv import SESMaxProjection
from .impl.ses_conv import SESConv_Z2_H, SESConv_H_H


class MNIST_SES_Scalar(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], rotations=[0.0], basis_type='A', **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales, rotations=rotations,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            SESConv_Z2_H(C1, C2, kernel_size, 7, scales=scales, rotations=rotations,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            SESConv_Z2_H(C2, C3, kernel_size, 7, scales=scales, rotations=rotations,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(kwargs.get('dropout', 0.7)),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MNIST_SES_V(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], rotations=[0.0], scale_size=1, rotation_size=1, basis_type='A', dropout=0.7, **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales, rotations=rotations,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            SESConv_H_H(C1, C2, scale_size, rotation_size, kernel_size, 7,
                        scales=scales, rotations=rotations,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            SESConv_H_H(C2, C3, scale_size, rotation_size, kernel_size, 7,
                        scales=scales, rotations=rotations,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def mnist_res_scalar_96_rot_1(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [1.0]
    rotations = [0.0]
    spt = 3.5
    basis = kwargs.get('basis', "D1")
    model = MNIST_SES_Scalar(pool_size=12, kernel_size=size, scales=scales, rotations=rotations,
                             basis_type=basis, mult=mult, max_order=4, dropout=dropout, spt=spt)
    return model

def mnist_res_scalar_96_rot_4(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [1.0]
    rotations = [x * 2*np.pi/4 for x in range(4)]
    spt = 3.5    
    basis = kwargs.get('basis', "D1")
    model = MNIST_SES_Scalar(pool_size=12, kernel_size=size, scales=scales, rotations=rotations,
                             basis_type=basis, mult=mult, max_order=4, dropout=dropout)
    return model

    
def mnist_res_scalar_96_rot_8(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [1.0]
    rotations = [x * 2*np.pi/8 for x in range(8)]
    spt = 3.5
    basis = kwargs.get('basis', "D1")
    model = MNIST_SES_Scalar(pool_size=12, kernel_size=size, scales=scales, rotations=rotations,
                             basis_type=basis, mult=mult, max_order=4, dropout=dropout)
    return model

def mnist_res_vector_96_rot_8_interrot_1(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [1.0]
    rotations = [x * 2*np.pi/8 for x in range(8)]
    spt = 3.5
    basis = kwargs.get('basis', "D1")
    model = MNIST_SES_V(pool_size=12, kernel_size=size, scales=scales, rotations=rotations,
                        scale_size=1, rotation_size=1,
                        basis_type=basis, mult=mult, max_order=4, dropout=dropout)
    return model

def mnist_res_vector_96_rot_8_interrot_4(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [1.0]
    rotations = [x * 2*np.pi/8 for x in range(8)]
    spt = 3.5
    basis = kwargs.get('basis', "D1")
    model = MNIST_SES_V(pool_size=12, kernel_size=size, scales=scales, rotations=rotations,
                        scale_size=1, rotation_size=4,
                        basis_type=basis, mult=mult, max_order=4, dropout=dropout)
    return model

def mnist_res_vector_96_rot_8_interrot_8(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [1.0]
    rotations = [x * 2*np.pi/8 for x in range(8)]
    spt = 3.5
    basis = kwargs.get('basis', "D1")
    model = MNIST_SES_V(pool_size=12, kernel_size=size, scales=scales, rotations=rotations,
                        scale_size=1, rotation_size=8,
                        basis_type=basis, mult=mult, max_order=4, dropout=dropout)
    return model
