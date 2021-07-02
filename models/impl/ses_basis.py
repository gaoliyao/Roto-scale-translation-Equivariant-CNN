'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import numpy as np
import torch
import torch.nn.functional as F
import math
from .fb import calculate_FB_bases, calculate_FB_bases_rot_scale


def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(X, coeff)
    return func


def onescale_grid_hermite_gaussian(size, scale, max_order=None):
    max_order = max_order or size - 1
    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)
    order_y, order_x = np.indices([max_order + 1, max_order + 1])

    G = np.exp(-X**2 / (2 * scale**2)) / scale
    # print(G.shape)
    # print(hermite_poly(X / scale, 1).shape)
    # print(order_x.ravel())

    # basis_x: (49, 5)
    # basis_y: (49, 5)
    basis_x = [G * hermite_poly(X / scale, n) for n in order_x.ravel()]
    basis_y = [G * hermite_poly(Y / scale, n) for n in order_y.ravel()]
    
    # basis_x: (49, 5) tensor
    # basis_y: (49, 5) tensor
    basis_x = torch.Tensor(np.stack(basis_x))
    basis_y = torch.Tensor(np.stack(basis_y))
    
    # basis_x[:, :, None]: (49, 5, 1)
    # basis_y[:, None, :]: (49, 1, 5)
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    
    # basis: (49, 5, 5)
    return basis


def multiscale_hermite_gaussian(size, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]
    # print('hermite scales', scales)

    basis_x = []
    basis_y = []

    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)

    for scale in scales:
        G = np.exp(-X**2 / (2 * scale**2)) / scale

        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order
        
        # bx: (15, 5)
        # by: (15, 5)
        bx = [G * hermite_poly(X / scale, n) for n in order_x[mask]]
        by = [G * hermite_poly(Y / scale, n) for n in order_y[mask]]
            
        basis_x.extend(bx)
        basis_y.extend(by)
    
    # basis_x[:49]: (49, 5)
    # basis_y[:49]: (49, 5)
    basis_x = torch.Tensor(np.stack(basis_x))[:num_funcs]
    basis_y = torch.Tensor(np.stack(basis_y))[:num_funcs]
    
    # basis_x[:, :, None]: (49, 5, 1)
    # basis_y[:, None, :]: (49, 1, 5)
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    # print("multiscale basis.shape")
    # print(basis.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis


def multiscale_fourier_bessel(size, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]
    # print('hermite scales', scales)
    
    # num_basis_ind = 15
    basis_xy = np.zeros([60, size, size])
    count = 0

    for scale in scales:
        # psi_scale.shape = (25, 15)
        psi_scale, _, _ = calculate_FB_bases(int((size-1)/2), scale, 15)
        # psi_scale.shape = (15, 25)
        psi_scale = psi_scale.transpose()
        # psi_scale *= 2**(-2*scale)
        print(psi_scale.shape)
        basis_xy[count*15:(1+count)*15] = psi_scale.reshape(15, size, size)
        count += 1
    
    # basis_xy: (49, 5, 5)
    basis_xy = torch.Tensor(basis_xy)[:num_funcs]
    
    # print("multiscale basis.shape")
    # print(basis_xy.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis_xy


def multiscale_fourier_bessel_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]
    # print('hermite scales', scales)
    
    # num_basis_ind = 15
    basis_xy = np.zeros([60, size, size])
    count = 0

    for scale in scales:
        # psi_scale.shape = (25, 15)
        psi_scale, _, _ = calculate_FB_bases_rot_scale(int((size-1)/2), base_rotation, scale, 15)
        # psi_scale.shape = (15, 25)
        psi_scale = psi_scale.transpose()
        # psi_scale *= 2**(-2*scale)
        print(psi_scale.shape)
        basis_xy[count*15:(1+count)*15] = psi_scale.reshape(15, size, size)
        count += 1
    
    # basis_xy: (49, 5, 5)
    basis_xy = torch.Tensor(basis_xy)[:num_funcs]
    
    # print("multiscale basis.shape")
    # print(basis_xy.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis_xy


def steerable_A(size, scales, effective_size, **kwargs):
    max_order = effective_size - 1
    max_scale = max(scales)
    basis_tensors = []
    # perform basis padding for basis with different scales
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        basis = onescale_grid_hermite_gaussian(size_before_pad, scale, max_order)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_basis A shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 4, 15, 15)
    # Example: (0, 0, :, :) has most 0 and (0, 3, :, :) no zero
    return steerable_basis


def steerable_B(size, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        print("size_before_pad")
        print(size_before_pad)
        assert size_before_pad > 1
        basis = multiscale_hermite_gaussian(size_before_pad,
                                            base_scale=scale,
                                            max_order=max_order,
                                            mult=mult,
                                            num_funcs=num_funcs)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_basis B.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 4, 15, 15)
    return steerable_basis

# only have scale multiscale basis with Fourier-Bessel
def steerable_C(size, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        print("size_before_pad")
        print(size_before_pad)
        assert size_before_pad > 1
        basis = multiscale_fourier_bessel(size_before_pad,
                                            base_scale=scale,
                                            max_order=max_order,
                                            mult=mult,
                                            num_funcs=num_funcs)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_basis C.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 4, 15, 15)
    return steerable_basis

# add rotation channel with multiscale basis using Fourier-Bessel
def steerable_D(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            print("size_before_pad")
            print(size_before_pad)
            assert size_before_pad > 1
            basis = multiscale_fourier_bessel_rot_scale(size_before_pad,
                                                base_rotation=rotation,
                                                base_scale=scale,
                                                max_order=max_order,
                                                mult=mult,
                                                num_funcs=num_funcs)
            basis = basis[None, :, :, :]
            pad_size = (size - size_before_pad) // 2
            basis = F.pad(basis, [pad_size] * 4)[0]
            basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_D_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis



def normalize_basis_by_min_scale(basis):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()[:, [0]]
    return basis / norm