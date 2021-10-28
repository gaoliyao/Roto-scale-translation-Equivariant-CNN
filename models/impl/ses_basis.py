'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import numpy as np
import torch
import torch.nn.functional as F
import math
from .fb import calculate_FB_bases, calculate_FB_bases_rot_scale, cartesian_to_polar_coordinates, calculate_FB_bases_rot_scale_gaussian
import pdb


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

def hermite_poly_rot_scale(x, y, rot, scale, n, m):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape
    """
    coeff1 = [0] * n + [1]
    coeff2 = [0] * m + [1]
    theta, rho = cartesian_to_polar_coordinates(x, y)
    theta += rot
    func = 1/scale**2
    func *= np.polynomial.hermite_e.hermeval(rho*np.cos(theta)/scale, coeff1)
    func *= np.polynomial.hermite_e.hermeval(rho*np.sin(theta)/scale, coeff2)
    func *= np.exp(-(rho**2)/(2*scale**2))
    return func


def SL(X, n, spt):
    """Sturm Liouvielle of order n+1 calculatd at X
    """
    # spt needs to be small
    n=n+1
    y = 0*X
    mask = (X>-spt) & (X < spt)
    y[mask] = np.sin(n*np.pi*X[mask]/2/spt+n*np.pi/2)
    return y

def SL_rot_scale(x, y, spt, rot, scale, n, m):
    theta, rho = cartesian_to_polar_coordinates(x, y)
    theta += rot
    x_new = rho*np.cos(theta)
    y_new = rho*np.sin(theta)
    func = 1/scale**2
    func *= SL(x_new/scale, n, spt)
    func *= SL(y_new/scale, m, spt)
    return func

def SL_gaussian_rot_scale(x, y, spt, rot, scale, n, m):
    theta, rho = cartesian_to_polar_coordinates(x, y)
    theta += rot
    x_new = rho*np.cos(theta)
    y_new = rho*np.sin(theta)
    func = 1/scale**2
    func *= SL(x_new/scale, n, spt)
    func *= SL(y_new/scale, m, spt)
    func *= np.exp(-(rho**2)/(2*scale**2))
    return func



#####################################################################################



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

def onescale_hermite_gaussian_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    # print('hermite scales', scales)

    basis_xy = []

    X, Y = np.meshgrid(range(-(size // 2), (size // 2)+1), range(-(size // 2), (size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    order_y, order_x = np.indices([max_order + 1, max_order + 1])

    # bx: (15, 5)
    # by: (15, 5)
    bxy = []
    
    for i in range(len(order_x)):
        for j in range(len(order_y)):
            n = order_x[i][j]
            m = order_y[i][j]
            base_n_m = hermite_poly_rot_scale(ugrid[:,0], ugrid[:,1], base_rotation, base_scale, n, m)
            # print(base_n_m.shape)                
            bxy.append(base_n_m)
    # print(np.array(bxy).shape)
    basis_xy.extend(bxy)
    print("onescale_hermite_gaussian_rot_scale")
    print(np.array(basis_xy).shape)
    
    # basis_x[:49]: (49, 5)
    # print("basis_xy.shape out for loop")
    # print(np.array(basis_xy).shape)
    basis = torch.Tensor(np.stack(basis_xy))[:num_funcs]
    basis = basis.reshape(-1, size, size)
    # print(basis[1,:,:])
    # basis_x[:, :, None]: (49, 5, 1)
    # basis_y[:, None, :]: (49, 1, 5)
    # print("multiscale basis hermite rot scale.shape")
    # print(basis.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis

def onescale_fourier_bessel(size, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    # print('hermite scales', scales)
    
    # num_basis_ind = 15
    basis_xy = np.zeros([60, size, size])
    # psi_scale.shape = (25, 15)
    psi_scale, _, _ = calculate_FB_bases(int((size-1)/2), scale, 60)
    # psi_scale.shape = (15, 25)
    psi_scale = psi_scale.transpose()
    # psi_scale *= 2**(-2*scale)
    # print(psi_scale.shape)
    basis_xy[0:60] = psi_scale.reshape(60, size, size)
    
    # basis_xy: (49, 5, 5)
    basis_xy = torch.Tensor(basis_xy)[:num_funcs]
    
    # print("multiscale basis.shape")
    # print(basis_xy.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis_xy

def onescale_fourier_bessel_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    
    # num_basis_ind = 15
    basis_xy = np.zeros([60, size, size])
    # psi_scale.shape = (25, 15)
    psi_scale, _, _ = calculate_FB_bases_rot_scale(int((size-1)/2), base_rotation, base_scale, 60)
    # psi_scale.shape = (15, 25)
    psi_scale = psi_scale.transpose()
    # psi_scale *= 2**(-2*scale)
    # print(psi_scale.shape)
    basis_xy[0:60] = psi_scale.reshape(60, size, size)
    
    # basis_xy: (49, 5, 5)
    basis_xy = torch.Tensor(basis_xy)[:num_funcs]
    
    # print("multiscale basis.shape")
    # print(basis_xy.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis_xy


def multiscale_hermite_gaussian(size, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]
    basis_x = []
    basis_y = []

    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)
    

    for scale in scales:
        G = np.exp(-X**2 / (2 * scale**2)) / scale

        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order
        bx = [G * hermite_poly(X / scale, n) for n in order_x[mask]]
        by = [G * hermite_poly(Y / scale, n) for n in order_y[mask]]
            
        basis_x.extend(bx)
        basis_y.extend(by)

    basis_x = torch.Tensor(np.stack(basis_x))[:num_funcs]
    basis_y = torch.Tensor(np.stack(basis_y))[:num_funcs]
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])

    return basis

def multiscale_hermite_gaussian_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]
    # print('hermite scales', scales)

    basis_xy = []

    X, Y = np.meshgrid(range(-(size // 2), (size // 2)+1), range(-(size // 2), (size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    for scale in scales:
        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order
        
        bxy = []
        for i in range(len(order_x[mask])):
            n = order_x[mask][i]
            m = order_y[mask][i]
            base_n_m = hermite_poly_rot_scale(ugrid[:,0], ugrid[:,1], base_rotation, scale, n, m)
            bxy.append(base_n_m)
        basis_xy.extend(bxy)
    
    basis = torch.Tensor(np.stack(basis_xy))[:num_funcs]
    basis = basis.reshape(-1, size, size)
    return basis

def onescale_SL_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None, spt=3.5):
    '''
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2

    basis_xy = []

    X, Y = np.meshgrid(range(-(size // 2), (size // 2)+1), range(-(size // 2), (size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    order_y, order_x = np.indices([max_order + 1, max_order + 1])
    # mask = order_y + order_x <= max_order

    bxy = []
    for i in range(len(order_x)):
        for j in range(len(order_y)):
            n = order_x[i][j]
            m = order_y[i][j]
            base_n_m = SL_rot_scale(ugrid[:,0], ugrid[:,1], spt, base_rotation, base_scale, n, m)
            bxy.append(base_n_m)
    basis_xy.extend(bxy)
    
    basis = torch.Tensor(np.stack(basis_xy))[:num_funcs]
    basis = basis.reshape(-1, size, size)
    return basis

def multiscale_SL_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None, spt=3.5):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]

    basis_xy = []

    X, Y = np.meshgrid(range(-(size // 2), (size // 2)+1), range(-(size // 2), (size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    for scale in scales:
        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order
        
        bxy = []
        for i in range(len(order_x[mask])):
            n = order_x[mask][i]
            m = order_y[mask][i]
            base_n_m = SL_rot_scale(ugrid[:,0], ugrid[:,1], spt, base_rotation, scale, n, m)
            bxy.append(base_n_m)
        basis_xy.extend(bxy)
    
    basis = torch.Tensor(np.stack(basis_xy))[:num_funcs]
    basis = basis.reshape(-1, size, size)
    return basis




def multiscale_SL_gaussian_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None, spt=3.5):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]

    basis_xy = []

    X, Y = np.meshgrid(range(-(size // 2), (size // 2)+1), range(-(size // 2), (size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    for scale in scales:
        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order
        
        bxy = []
        for i in range(len(order_x[mask])):
            n = order_x[mask][i]
            m = order_y[mask][i]
            base_n_m = SL_gaussian_rot_scale(ugrid[:,0], ugrid[:,1], spt, base_rotation, scale, n, m)
            bxy.append(base_n_m)
        basis_xy.extend(bxy)
    
    basis = torch.Tensor(np.stack(basis_xy))[:num_funcs]
    basis = basis.reshape(-1, size, size)
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
        # print(psi_scale.shape)
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
        psi_scale, _, _ = calculate_FB_bases_rot_scale(int((size-1)/2), base_rotation, scale, num_funcs_per_scale)
        # psi_scale.shape = (15, 25)
        psi_scale = psi_scale.transpose()
        # psi_scale *= 2**(-2*scale)
        # print(psi_scale.shape)
        basis_xy[count*num_funcs_per_scale:(1+count)*num_funcs_per_scale] = psi_scale.reshape(num_funcs_per_scale, size, size)
        count += 1
    
    # basis_xy: (49, 5, 5)
    basis_xy = torch.Tensor(basis_xy)[:num_funcs]
    
    # print("multiscale basis.shape")
    # print(basis_xy.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis_xy

def multiscale_gaussian_fourier_bessel_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None):
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
        psi_scale, _, _ = calculate_FB_bases_rot_scale_gaussian(int((size-1)/2), base_rotation, scale, num_funcs_per_scale)
        # psi_scale.shape = (15, 25)
        psi_scale = psi_scale.transpose()
        # psi_scale *= 2**(-2*scale)
        # print(psi_scale.shape)
        basis_xy[count*num_funcs_per_scale:(1+count)*num_funcs_per_scale] = psi_scale.reshape(num_funcs_per_scale, size, size)
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
        # print("size_before_pad")
        # print(size_before_pad)
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

# add rotation channel with multiscale basis using Hermite polynomial
def steerable_C(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = multiscale_hermite_gaussian_rot_scale(size_before_pad,
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
    print("steerable_C_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis

# only have scale multiscale basis with Fourier-Bessel
def steerable_D(size, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        # print("size_before_pad")
        # print(size_before_pad)
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
    print("steerable_basis D.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 4, 15, 15)
    return steerable_basis

# add rotation channel with multiscale basis using Fourier-Bessel
# Revisions made to this function (23/Aug/2021)
def steerable_E(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    min_scale = min(scales)
    basis_tensors = []
    print("Steerable_E basis scales")
    # print(scales)
    # 1.7, ..., 5.1
    for rotation in rotations:
        for scale in scales:
            # [Change 1]: make this back to the original version (same as other functions)
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            assert size_before_pad > 1
            # [Change 2]: check fb.py for calculate_FB_bases_rot_scale
            basis = multiscale_fourier_bessel_rot_scale(size_before_pad,
                                                base_rotation=rotation,
                                                base_scale=scale,
                                                max_order=max_order,
                                                mult=mult,
                                                num_funcs=num_funcs)
            basis = basis[None, :, :, :]
            pad_size = (size - size_before_pad) // 2
            # (15, 15)
            # (5, 5)
            basis = F.pad(basis, [pad_size] * 4)[0]
            basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_E_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis

# add rotation channel with multiscale basis using Fourier-Bessel with Gaussian
def steerable_F(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = multiscale_gaussian_fourier_bessel_rot_scale(size_before_pad,
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
    print("steerable_F_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis

# add rotation channel with multiscale basis using SL NO gaussian
def steerable_G(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    spt = kwargs.get('spt', 3.5)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # size_before_pad = int(size * max_scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = multiscale_SL_rot_scale(size_before_pad,
                                                base_rotation=rotation,
                                                base_scale=scale,
                                                max_order=max_order,
                                                mult=mult,
                                                num_funcs=num_funcs,
                                                spt=spt)
            basis = basis[None, :, :, :]
            pad_size = (size - size_before_pad) // 2
            basis = F.pad(basis, [pad_size] * 4)[0]
            basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_G_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis

# add rotation channel with multiscale basis using SL NO gaussian
def steerable_G1(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    spt = kwargs.get('spt', 3.5)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = onescale_SL_rot_scale(size_before_pad,
                                                base_rotation=rotation,
                                                base_scale=scale,
                                                max_order=max_order,
                                                mult=mult,
                                                num_funcs=num_funcs,
                                                spt=spt)
            basis = basis[None, :, :, :]
            pad_size = (size - size_before_pad) // 2
            basis = F.pad(basis, [pad_size] * 4)[0]
            basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_G1_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis



# add rotation channel with multiscale basis using SL gaussian
def steerable_H(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    spt = kwargs.get('spt', 3.5)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = multiscale_SL_gaussian_rot_scale(size_before_pad,
                                                base_rotation=rotation,
                                                base_scale=scale,
                                                max_order=max_order,
                                                mult=mult,
                                                num_funcs=num_funcs,
                                                spt=spt)
            basis = basis[None, :, :, :]
            pad_size = (size - size_before_pad) // 2
            basis = F.pad(basis, [pad_size] * 4)[0]
            basis_tensors.append(basis)
    steerable_basis = torch.stack(basis_tensors, 1)
    print("steerable_H_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis


# only have onescale basis with Fourier-Bessel
def steerable_D1(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = kwargs.get('max_order', 4)
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    print(scales)
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * max_scale / max_scale) // 2 * 2 + 1
            print("size_before_pad")
            print(size_before_pad)
            assert size_before_pad > 1
            basis = onescale_fourier_bessel_rot_scale(size_before_pad,
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
    print("steerable_D1_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis

# only have onescale basis with Hermite Gaussian
def steerable_A1(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = effective_size - 1
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = onescale_hermite_gaussian_rot_scale(size_before_pad,
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
    print("steerable_A1_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis

def steerable_rot_scale(x, y, rot, scale, j, k):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape
    """
    theta, rho = cartesian_to_polar_coordinates(x, y)
    theta = np.array(theta, dtype=np.complex)
    # tau_j(r)
    func = np.exp(-(rho-j)**2/(2*scale**2))
    # e^(i*k*phi)
    func = func + 0j
    func *= np.exp(1j*(k+0j)*(theta+0j))
    func *= np.exp(-1j*(k+0j)*(rot+0j))
    func = func.real
    return func

def onescale_steerable_rot_scale(size, base_rotation, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    # print('hermite scales', scales)

    basis_xy = []

    X, Y = np.meshgrid(range(-(size // 2), (size // 2)+1), range(-(size // 2), (size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    order_y, order_x = np.indices([max_order + 1, max_order + 1])

    # bx: (15, 5)
    # by: (15, 5)
    bxy = []
    
    for i in range(len(order_x)):
        for j in range(len(order_y)):
            n = order_x[i][j]
            m = order_y[i][j]
            base_n_m = steerable_rot_scale(ugrid[:,0], ugrid[:,1], base_rotation, base_scale, n, m)
            # print(base_n_m.shape)                
            bxy.append(base_n_m)
    # print(np.array(bxy).shape)
    basis_xy.extend(bxy)
    print("onescale_steerable_rot_scale")
    print(np.array(basis_xy).shape)
    
    # basis_x[:49]: (49, 5)
    # print("basis_xy.shape out for loop")
    # print(np.array(basis_xy).shape)
    basis = torch.Tensor(np.stack(basis_xy))[:num_funcs]
    basis = basis.reshape(-1, size, size)
    # print(basis[1,:,:])
    # basis_x[:, :, None]: (49, 5, 1)
    # basis_y[:, None, :]: (49, 1, 5)
    # print("multiscale basis hermite rot scale.shape")
    # print(basis.shape)
    
    # 49 basis in total, 25 is the filter map
    # basis: (49, 5, 5)
    return basis

# only have onescale basis with Hermite Gaussian
def steerable_I1(size, rotations, scales, effective_size, **kwargs):
    mult = kwargs.get('mult', 1.2)
    max_order = effective_size - 1
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for rotation in rotations:
        for scale in scales:
            size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            # print("size_before_pad")
            # print(size_before_pad)
            assert size_before_pad > 1
            basis = onescale_steerable_rot_scale(size_before_pad,
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
    print("steerable_I1_basis.shape")
    print(steerable_basis.shape)
    # steerable_basis: (49, 16, 15, 15)
    return steerable_basis


def normalize_basis_by_min_scale(basis):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()[:, [0]]
    return basis / norm
