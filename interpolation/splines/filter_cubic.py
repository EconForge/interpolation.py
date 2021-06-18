from __future__ import division

import numpy as np
import time
from numba import jit, njit

# used by njitted routines (frozen)
basis = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0])


@njit(cache=True)
def solve_deriv_interp_1d(bands, coefs):

    M = coefs.shape[0] - 2

    # Solve interpolating equations
    # First and last rows are different

    bands[0, 1] /= bands[0, 0]
    bands[0, 2] /= bands[0, 0]
    bands[0, 3] /= bands[0, 0]
    bands[0, 0] = 1.0
    bands[1, 1] -= bands[1, 0] * bands[0, 1]
    bands[1, 2] -= bands[1, 0] * bands[0, 2]
    bands[1, 3] -= bands[1, 0] * bands[0, 3]
    bands[0, 0] = 0.0
    bands[1, 2] /= bands[1, 1]
    bands[1, 3] /= bands[1, 1]
    bands[1, 1] = 1.0

    # Now do rows 2 through M+1
    for row in range(2, M + 1):
        bands[row, 1] -= bands[row, 0] * bands[row - 1, 2]
        bands[row, 3] -= bands[row, 0] * bands[row - 1, 3]
        bands[row, 2] /= bands[row, 1]
        bands[row, 3] /= bands[row, 1]
        bands[row, 0] = 0.0
        bands[row, 1] = 1.0

    # Do last row
    bands[M + 1, 1] -= bands[M + 1, 0] * bands[M - 1, 2]
    bands[M + 1, 3] -= bands[M + 1, 0] * bands[M - 1, 3]
    bands[M + 1, 2] -= bands[M + 1, 1] * bands[M, 2]
    bands[M + 1, 3] -= bands[M + 1, 1] * bands[M, 3]
    bands[M + 1, 3] /= bands[M + 1, 2]
    bands[M + 1, 2] = 1.0

    coefs[M + 1] = bands[(M + 1), 3]
    # Now back substitute up
    for row in range(M, 0, -1):
        coefs[row] = bands[row, 3] - bands[row, 2] * coefs[row + 1]

    # Finish with first row
    coefs[0] = bands[0, 3] - bands[0, 1] * coefs[1] - bands[0, 2] * coefs[2]


@njit(cache=True)
def find_coefs_1d(delta_inv, M, data, coefs):

    bands = np.zeros((M + 2, 4))

    # Setup boundary conditions
    abcd_left = np.zeros(4)
    abcd_right = np.zeros(4)

    # Left boundary
    abcd_left[0] = 1.0 * delta_inv * delta_inv
    abcd_left[1] = -2.0 * delta_inv * delta_inv
    abcd_left[2] = 1.0 * delta_inv * delta_inv
    abcd_left[3] = 0

    # Right boundary
    abcd_right[0] = 1.0 * delta_inv * delta_inv
    abcd_right[1] = -2.0 * delta_inv * delta_inv
    abcd_right[2] = 1.0 * delta_inv * delta_inv
    abcd_right[3] = 0

    for i in range(4):
        bands[0, i] = abcd_left[i]
        bands[M + 1, i] = abcd_right[i]

    for i in range(M):
        for j in range(3):
            bands[i + 1, j] = basis[j]
            bands[i + 1, 3] = data[i]

    solve_deriv_interp_1d(bands, coefs)


@njit(cache=True)
def filter_coeffs_1d(dinv, data):

    M = data.shape[0]
    N = M + 2

    coefs = np.zeros(N)
    find_coefs_1d(dinv[0], M, data, coefs)

    return coefs


@njit(cache=True)
def filter_coeffs_2d(dinv, data):

    Mx = data.shape[0]
    My = data.shape[1]

    Nx = Mx + 2
    Ny = My + 2

    coefs = np.zeros((Nx, Ny))

    # First, solve in the X-direction
    for iy in range(My):
        # print(data[:,iy].size)
        # print(spline.coefs[:,iy].size)
        find_coefs_1d(dinv[0], Mx, data[:, iy], coefs[:, iy])

    # Now, solve in the Y-direction
    for ix in range(Nx):
        find_coefs_1d(dinv[1], My, coefs[ix, :], coefs[ix, :])

    return coefs


@njit(cache=True)
def filter_coeffs_3d(dinv, data):

    Mx = data.shape[0]
    My = data.shape[1]
    Mz = data.shape[2]

    Nx = Mx + 2
    Ny = My + 2
    Nz = Mz + 2

    coefs = np.zeros((Nx, Ny, Nz))

    for iy in range(My):
        for iz in range(Mz):
            find_coefs_1d(dinv[0], Mx, data[:, iy, iz], coefs[:, iy, iz])

    # Now, solve in the Y-direction
    for ix in range(Nx):
        for iz in range(Mz):
            find_coefs_1d(dinv[1], My, coefs[ix, :, iz], coefs[ix, :, iz])

    # Now, solve in the Z-direction
    for ix in range(Nx):
        for iy in range(Ny):
            find_coefs_1d(dinv[2], Mz, coefs[ix, iy, :], coefs[ix, iy, :])

    return coefs


@njit(cache=True)
def filter_coeffs_4d(dinv, data):

    Mx = data.shape[0]
    My = data.shape[1]
    Mz = data.shape[2]
    Mz4 = data.shape[3]

    Nx = Mx + 2
    Ny = My + 2
    Nz = Mz + 2
    Nz4 = Mz4 + 2

    coefs = np.zeros((Nx, Ny, Nz, Nz4))

    # First, solve in the X-direction
    for iy in range(My):
        for iz in range(Mz):
            for iz4 in range(Mz4):
                find_coefs_1d(dinv[0], Mx, data[:, iy, iz, iz4], coefs[:, iy, iz, iz4])

    # Now, solve in the Y-direction
    for ix in range(Nx):
        for iz in range(Mz):
            for iz4 in range(Mz4):
                find_coefs_1d(dinv[1], My, coefs[ix, :, iz, iz4], coefs[ix, :, iz, iz4])

    # Now, solve in the Z-direction
    for ix in range(Nx):
        for iy in range(Ny):
            for iz4 in range(Mz4):
                find_coefs_1d(dinv[2], Mz, coefs[ix, iy, :, iz4], coefs[ix, iy, :, iz4])

    # Now, solve in the Z4-direction
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                find_coefs_1d(dinv[3], Mz4, coefs[ix, iy, iz, :], coefs[ix, iy, iz, :])

    return coefs


def filter_coeffs(smin, smax, orders, data):
    smin = np.array(smin, dtype=float)
    smax = np.array(smax, dtype=float)
    dinv = (smax - smin) / orders
    data = data.reshape(orders)
    return filter_data(dinv, data)


def filter_mcoeffs(smin, smax, orders, data):

    order = len(smin)
    n_splines = data.shape[-1]
    coefs = np.zeros(tuple([i + 2 for i in orders]) + (n_splines,))
    for i in range(n_splines):
        coefs[..., i] = filter_coeffs(smin, smax, orders, data[..., i])
    return coefs


def filter_data(dinv, data):
    if len(dinv) == 1:
        return filter_coeffs_1d(dinv, data)
    elif len(dinv) == 2:
        return filter_coeffs_2d(dinv, data)
    elif len(dinv) == 3:
        return filter_coeffs_3d(dinv, data)
    elif len(dinv) == 4:
        return filter_coeffs_4d(dinv, data)


#


if __name__ == "__main__":

    import numpy

    dinv = numpy.ones(3, dtype=float) * 0.5
    coeffs_0 = numpy.random.random([10, 10, 10])
    coeffs_1 = numpy.random.random([100, 100, 100])

    print(coeffs_0[:2, :2, :2])
    import time

    t1 = time.time()
    filter_coeffs_3d(dinv, coeffs_0)

    t2 = time.time()
    filter_coeffs_3d(dinv, coeffs_1)
    t3 = time.time()

    print("Elapsed : {}".format(t2 - t1))
    print("Elapsed : {}".format(t3 - t2))
