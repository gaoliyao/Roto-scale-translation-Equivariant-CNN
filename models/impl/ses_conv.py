'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .ses_basis import steerable_A, steerable_A1, steerable_B, steerable_C, steerable_D, steerable_D1, steerable_E, steerable_F, steerable_G, steerable_G1, steerable_H
from .ses_basis import normalize_basis_by_min_scale


class SESConv_Z2_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', num_rotations x num_scales, H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=[1.0], rotations = [0.0], stride=1, padding=0, bias=False, basis_type='A', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]        
        self.num_scales = len(scales)
        self.rotations = rotations
        self.num_rotations = len(rotations)
        self.stride = stride
        self.padding = padding

        if basis_type == 'A':
            basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'A1':
            basis = steerable_A1(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'B':
            basis = steerable_B(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'C':
            basis = steerable_C(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'D':
            basis = steerable_D(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'D1':
            basis = steerable_D1(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'E':
            basis = steerable_E(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'F':
            basis = steerable_F(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'G':
            basis = steerable_G(kernel_size, self.rotations, scales, effective_size, **kwargs)  
        elif basis_type == 'G1':
            basis = steerable_G1(kernel_size, self.rotations, scales, effective_size, **kwargs)  
        elif basis_type == 'H':
            basis = steerable_H(kernel_size, self.rotations, scales, effective_size, **kwargs)     
            
        basis = normalize_basis_by_min_scale(basis)
        print("Normalized!!")
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # basis originally has shape [num_funcs, num_rotations x num_scales, ks, ks], indexed by [k, theta alpha, u1, u2]
        # basis(k, theta alpha, u1, u2) = 2^(-2 alpha)psi_k(2^{-alpha} R_theta u)
        basis = self.basis.view(self.num_funcs, -1)
        # kernel in shape [out_channels, in_channels, num_rotationsxnum_scalesxHxW], index: [lambda, lambda', theta alpha u1 u2]
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels,
                             self.num_rotations * self.num_scales, self.kernel_size, self.kernel_size)
        
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        # now kernel in shape [out, num_rotations x num_scales, in, H, W], index: [lambda, theta alpha, lambda', u1, u2]
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        # now kernel in shape [out x num_rotations x num_scales, in, H, W], index: [lambda theta alpha, lambda', u1, u2]

        # convolution    
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        # y is in shape [batch, channel x num_rotations x num_scales, height, width], indexed by [batch, lambda theta alpha, u1, u2 ]
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_rotations * self.num_scales, H, W)
        # y is in shape [batch, channel, num_rotations x num_scales, height, width], indexed by [batch, lambda, theta alpha, u1, u2 ]

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)


class SESConv_H_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, num_rotations x num_scales, H, W] -> [B, C', num_rotaions x num_scales, H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter (interscale)
        rotation_size: Size of rotation filter (interrotation)
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scale_size, rotation_size, kernel_size, effective_size,
                 scales=[1.0], rotations = [0.0], stride=1, padding=0, bias=False, basis_type='A', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.rotation_size = rotation_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.rotations = rotations
        self.num_rotations = len(rotations)
        self.stride = stride
        self.padding = padding

        if basis_type == 'A':
            basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'A1':
            basis = steerable_A1(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'B':
            basis = steerable_B(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'C':
            basis = steerable_C(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'D':
            basis = steerable_D(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'D1':
            basis = steerable_D1(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'E':
            basis = steerable_E(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'F':
            basis = steerable_F(kernel_size, self.rotations, scales, effective_size, **kwargs)
        elif basis_type == 'G':
            basis = steerable_G(kernel_size, self.rotations, scales, effective_size, **kwargs)         
        elif basis_type == 'G1':
            basis = steerable_G1(kernel_size, self.rotations, scales, effective_size, **kwargs)  
        elif basis_type == 'H':
            basis = steerable_H(kernel_size, self.rotations, scales, effective_size, **kwargs)

        basis = normalize_basis_by_min_scale(basis)
        print("Normalized!!")
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        # weight has size [out_channels, in_channels, rotation_size x scale_size, num_funcs]
        # i.e., a(lambda,lambda',theta' alpha',k)                

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, self.rotation_size * self.scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        # basis originally has shape [num_funcs, num_rotations x num_scales, ks, ks], indexed by [k, theta alpha, u1, u2]
        # basis(k, theta alpha, u1, u2) = 2^(-2 alpha)psi_k(2^{-alpha} R_theta u)
        basis = self.basis.view(self.num_funcs, -1)

        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels, self.rotation_size * self.scale_size,
                             self.num_rotations * self.num_scales, self.kernel_size, self.kernel_size)
        # kernel now in shape [out_channels, in_channels, rotation_size x scale_size, num_rotations x num_scales, ks, ks]
        # index: [lambda, lambda', theta' alpha', theta alpha, u1, u2]
        # kernel(lambda, lambda', theta' alpha', theta alpha, u1, u2) = 2^{-2alpha}W_{lambda,lambda'}(2^{-alpha}R_theta u, thata', alpha')                

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        #kernel now indexed by kernel(theta alpha, lambda, lambda', theta'xalpha', u1, u2)        
        kernel = kernel.view(-1, self.in_channels, self.rotation_size, self.scale_size,
                             self.kernel_size, self.kernel_size)
        # kernal now indexed by kernel(thetaxalphaxlambda, lambda', theta', alpha', u1, u2)        

        # calculate padding
        # x in shape [batchsize, in_channels, num_rotationsxnum_scales, H, W], indexed by [batch, lambda', thataxalpha, u1, u2]
        B, C, S, H, W = x.shape
        # expand x into shape [batchsize x in_channels, num_rotations, num_scales, H x W], indexed by [batch x lambda', thata, alpha, u1 x u2]
        x = x.view(B * C, self.num_rotations, self.num_scales, H * W)
        if self.scale_size != 1:
            x = F.pad(x, [0, 0, 0, self.scale_size - 1])

        if self.rotation_size != 1:
            x = x.view(B, C, self.num_rotations, -1)
            x = F.pad(x, [0, 0, 0, self.rotation_size-1], mode='circular')
            
        x = x.view(B, C, self.num_rotations + self.rotation_size - 1, self.num_scales + self.scale_size - 1, H, W)
            
        output = 0.0
        # to be complete
        for i in range(self.rotation_size):
            for j in range(self.scale_size):
                # print("rotation difference")
                # print(x[:, :, i, j:j + self.num_scales]-x[:, :, i+1, j:j + self.num_scales])
                x_ = x[:, :, i:i + self.num_rotations, j:j + self.num_scales]
                # expand X
                B, C, R, S, H, W = x_.shape
                x_ = x_.permute(0, 2, 3, 1, 4, 5).contiguous()
                x_ = x_.view(B, -1, H, W)
                output += F.conv2d(x_, kernel[:, :, i, j], padding = self.padding,
                                   groups=R*S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, R*S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)


class SESConv_H_H_1x1(nn.Conv2d):
    # not yet implemented
    def __init__(self, in_channels, out_channel, stride=1, num_scales=1, bias=True):
        super().__init__(in_channels, out_channel, 1, stride=stride, bias=bias)
        self.num_scales = num_scales

    def forward(self, x):
        kernel = self.weight.unsqueeze(0)
        kernel = kernel.expand(self.num_scales, -1, -1, -1, -1).contiguous()
        kernel = kernel.view(-1, self.in_channels, 1, 1)

        B, C, S, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, -1, H, W)
        x = F.conv2d(x, kernel, stride=self.stride, groups=self.num_scales)

        B, C_, H_, W_ = x.shape
        x = x.view(B, S, -1, H_, W_).permute(0, 2, 1, 3, 4).contiguous()
        return x


class SESMaxProjection(nn.Module):
    # (H, W)
    # (0000000 1.5 1.6 0000000)

    def forward(self, x):
        # (128, 32, 32, 28, 28)
        # x = (batch, in*out, rot*scale, height, width)
        
        # (128, 32, 28, 28)
        # B, I_O, R_S, H, W = x.shape
        # return_val = x.max(2)[0]
        # print("===============")
        # print(torch.norm(x[0][0][0]))
        # print(torch.norm(x[0][0][1]))
        # print(torch.norm(x[0][0][2]))
        # print(torch.norm(x[0][0][3]))
        # print(torch.norm(x[0][0][4]))
        # print(torch.norm(x[0][0][5]))
        # print(torch.norm(x[0][0][6]))
        # print(torch.norm(x[0][0][7]))
        # print("return_val[0][0]")
        # print(torch.norm(return_val[0][0]))
        # print("===============")
        
        return x.max(2)[0]
