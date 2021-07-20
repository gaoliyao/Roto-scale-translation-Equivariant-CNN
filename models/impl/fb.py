'''
The code is directly translated from the matlab code 
https://github.com/xycheng/DCFNet/blob/master/calculate_FB_bases.m and utilize
https://github.com/ZeWang95/DCFNet-Pytorch/blob/master/fb.py
'''
import numpy as np 
from scipy import special

# Change based on environment
path_to_bessel = "/home/gao463/Downloads/sesn-master/models/impl/bessel.npy"

def cartesian_to_polar_coordinates(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

# Input: 
# L1: size of filter
# alpha: scale transformation
# maxK: maximum number of basis
# Output: 
# psi: FB basis with shape: (filter_map^2, num_basis)
def calculate_FB_bases(L1, alpha, maxK):
    '''
    s = 2^alpha is the scale
    alpha <= 0
    maxK is the maximum num of bases you need
    '''
    #maxK = (2 * L1 + 1)**2 - 1
    maxK = np.min([(2 * L1 + 1)**2 - 1, maxK])

    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 2.5

    if L1 < 2:
        truncate_freq_factor = 2.5

    xx, yy = np.meshgrid(range(-L, L+1), range(-L, L+1))

    xx = xx/R*2**-alpha
    yy = yy/R*2**-alpha

    ugrid = np.concatenate([yy.reshape(-1,1), xx.reshape(-1,1)], 1)
    # angleGrid, lengthGrid
    tgrid, rgrid = cartesian_to_polar_coordinates(ugrid[:,0], ugrid[:,1])

    num_grid_points = ugrid.shape[0]

    maxAngFreq = 15
    
    # change path based on environment
    bessel = np.load(path_to_bessel)

    B = bessel[(bessel[:,0] <= maxAngFreq) & (bessel[:,3]<= np.pi*R*truncate_freq_factor)]
    # print("B.shape")
    # print(B.shape)

    idxB = np.argsort(B[:,2])

    mu_ns = B[idxB, 2]**2

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns=np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases=0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid=rgrid*R_ns[i]

        F = special.jv(ki, r0grid)

        Phi = 1./np.abs(special.jv(ki+1, R_ns[i]))*F

        Phi[rgrid >=1]=0

        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+1

        else:
            Psi.append(Phi*np.cos(ki*tgrid)*np.sqrt(2))
            Psi.append(Phi*np.sin(ki*tgrid)*np.sqrt(2))
            kq_Psi.append([ki,qi,rkqi])
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+2
                        
    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    num_bases = Psi.shape[1]

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2*L+1, 2*L+1).transpose(1,2,0)
    psi = p[1:-1, 1:-1, :]
    # print(psi.shape)
    psi = psi.reshape((2*L1+1)**2, num_bases)
        
    # normalize
    # using the sum of psi_0 to normalize.
    #c = np.sqrt(np.sum(psi**2, 0).mean())
    #psi = psi/c
    c = np.sum(psi[:,0])
    
    # psi.shape example: (9, 6), (25, 6)
    # psi.shape: (filter_map^2, num_basis)
    psi = psi/c

    return psi, c, kq_Psi

# Input: 
# L1: size of filter
# theta: angle transformation (0 to 2pi)
# alpha: scale transformation
# maxK: maximum number of basis
# Output: 
# psi: FB basis with shape: (filter_map^2, num_basis)
def calculate_FB_bases_rot_scale(L1, theta, alpha, maxK):
    '''
    s = 2^alpha is the scale
    alpha <= 0
    maxK is the maximum num of bases you need
    '''
    #maxK = (2 * L1 + 1)**2 - 1
    maxK = np.min([(2 * L1 + 1)**2 - 1, maxK])

    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 2.5

    if L1 < 2:
        truncate_freq_factor = 2

    xx, yy = np.meshgrid(range(-L, L+1), range(-L, L+1))

    xx = xx/(R*alpha)
    yy = yy/(R*alpha)

    ugrid = np.concatenate([yy.reshape(-1,1), xx.reshape(-1,1)], 1)
    # angleGrid, lengthGrid
    tgrid, rgrid = cartesian_to_polar_coordinates(ugrid[:,0], ugrid[:,1])
    tgrid += theta

    num_grid_points = ugrid.shape[0]

    maxAngFreq = 15

    bessel = np.load(path_to_bessel)

    B = bessel[(bessel[:,0] <= maxAngFreq) & (bessel[:,3]<= np.pi*R*truncate_freq_factor)]
    # print("B.shape")
    # print(B.shape)

    idxB = np.argsort(B[:,2])

    mu_ns = B[idxB, 2]**2

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns=np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases=0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid=rgrid*R_ns[i]

        F = special.jv(ki, r0grid)

        Phi = 1./np.abs(special.jv(ki+1, R_ns[i]))*F

        Phi[rgrid >=1]=0

        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+1

        else:
            Psi.append(Phi*np.cos(ki*tgrid)*np.sqrt(2))
            Psi.append(Phi*np.sin(ki*tgrid)*np.sqrt(2))
            kq_Psi.append([ki,qi,rkqi])
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+2
                        
    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    num_bases = Psi.shape[1]

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2*L+1, 2*L+1).transpose(1,2,0)
    psi = p[1:-1, 1:-1, :]
    # print(psi.shape)
    psi = psi.reshape((2*L1+1)**2, num_bases)
        
    # normalize
    # using the sum of psi_0 to normalize.
    #c = np.sqrt(np.sum(psi**2, 0).mean())
    #psi = psi/c
    c = np.sum(psi[:,0])
    
    # psi.shape example: (9, 6), (25, 6)
    # psi.shape: (filter_map^2, num_basis)
    psi = psi/c

    return psi, c, kq_Psi

# Input: 
# L1: size of filter
# theta: angle transformation (0 to 2pi)
# alpha: scale transformation
# maxK: maximum number of basis
# Output: 
# psi: FB basis with shape: (filter_map^2, num_basis)
def calculate_FB_bases_rot_scale_gaussian(L1, theta, alpha, maxK):
    '''
    s = 2^alpha is the scale
    alpha <= 0
    maxK is the maximum num of bases you need
    '''
    #maxK = (2 * L1 + 1)**2 - 1
    maxK = np.min([(2 * L1 + 1)**2 - 1, maxK])

    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 2.5

    if L1 < 2:
        truncate_freq_factor = 2

    xx, yy = np.meshgrid(range(-L, L+1), range(-L, L+1))

    # xx = xx/R*2**-alpha
    # yy = yy/R*2**-alpha
    
    xx = xx/(R*alpha)
    yy = yy/(R*alpha)

    ugrid = np.concatenate([yy.reshape(-1,1), xx.reshape(-1,1)], 1)
    # angleGrid, lengthGrid
    tgrid, rgrid = cartesian_to_polar_coordinates(ugrid[:,0], ugrid[:,1])
    tgrid += theta

    num_grid_points = ugrid.shape[0]

    maxAngFreq = 15

    bessel = np.load(path_to_bessel)

    B = bessel[(bessel[:,0] <= maxAngFreq) & (bessel[:,3]<= np.pi*R*truncate_freq_factor)]
    # print("B.shape")
    # print(B.shape)

    idxB = np.argsort(B[:,2])

    mu_ns = B[idxB, 2]**2

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns=np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases=0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid=rgrid*R_ns[i]

        F = special.jv(ki, r0grid)
        
        Phi = 1./np.abs(special.jv(ki+1, R_ns[i]))*F*np.exp(-(rgrid**2)/(2*alpha**2))*(1.0/alpha)

        Phi[rgrid >=1]=0

        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+1

        else:
            Psi.append(Phi*np.cos(ki*tgrid)*np.sqrt(2))
            Psi.append(Phi*np.sin(ki*tgrid)*np.sqrt(2))
            kq_Psi.append([ki,qi,rkqi])
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+2
                        
    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    num_bases = Psi.shape[1]

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2*L+1, 2*L+1).transpose(1,2,0)
    psi = p[1:-1, 1:-1, :]
    # print(psi.shape)
    psi = psi.reshape((2*L1+1)**2, num_bases)
        
    # normalize
    # using the sum of psi_0 to normalize.
    #c = np.sqrt(np.sum(psi**2, 0).mean())
    #psi = psi/c
    c = np.sum(psi[:,0])
    
    # psi.shape example: (9, 6), (25, 6)
    # psi.shape: (filter_map^2, num_basis)
    psi = psi/c

    return psi, c, kq_Psi