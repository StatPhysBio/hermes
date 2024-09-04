

import numpy as np
import scipy
import torch

def spherical_to_cartesian__pytorch(x):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    # get cartesian coordinates
    r = x[:,0]
    t = x[:,1]
    p = x[:,2]

    # get spherical coords from cartesian
    x_ = torch.sin(t)*torch.cos(p)*r
    y_ = torch.sin(t)*torch.sin(p)*r
    z_ = torch.cos(t)*r

    # return x, y, z
    return torch.cat([x_.view(-1, 1), y_.view(-1, 1), z_.view(-1, 1)], dim=-1)

def cartesian_to_spherical__numpy(xyz):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    t = np.arccos(z/r)
    p = np.arctan2(y,x)
    return np.hstack([r.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1)])

def spherical_to_cartesian__numpy(x):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    # get cartesian coordinates
    r = x[:,0]
    t = x[:,1]
    p = x[:,2]

    # get spherical coords from cartesian
    x_ = np.sin(t)*np.cos(p)*r
    y_ = np.sin(t)*np.sin(p)*r
    z_ = np.cos(t)*r

    # return x, y, z
    return np.hstack([x_.reshape(-1, 1), y_.reshape(-1, 1), z_.reshape(-1, 1)])

def cartesian_to_spherical__pytorch(xyz):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = torch.sqrt(x*x + y*y + z*z)
    t = torch.acos(z/r)
    p = torch.atan2(y,x)
    return torch.cat([r.view(-1, 1), t.view(-1, 1), p.view(-1, 1)], dim=-1)


def change_basis_complex_to_real(l: int, dtype=None, device=None) -> np.ndarray:
    """
    Function to convert change of basis matrix from complex to real spherical
    harmonics
     
    Taken from e3nn: https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_wigner.py
    """
    # taken from e3nn
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l - abs(m)] = 1j / 2**0.5
        q[l + m, l + abs(m)] = -(-1)**abs(m) * 1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l - abs(m)] = 1 / 2**0.5
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
    q = q  # No factor of 1j

    return q
