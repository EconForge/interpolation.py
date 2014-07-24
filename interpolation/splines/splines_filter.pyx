
from __future__ import division

import numpy as np
from cython import double

import cython
from libc.math cimport floor
from cython.parallel import parallel, prange
from cython import nogil

# Note : there is another version of this file that uses a sparse matrix solver instead of the custom one

import time

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve_deriv_interp_1d(double[:,:] bands, double[:] coefs):

    cdef int M = coefs.shape[0]-2

    cdef int row
    # Solve interpolating equations
    # First and last rows are different

    bands[0,1] /= bands[0,0]
    bands[0,2] /= bands[0,0]
    bands[0,3] /= bands[0,0]
    bands[0,0] = 1.0
    bands[1,1] -= bands[1,0]*bands[0,1]
    bands[1,2] -= bands[1,0]*bands[0,2]
    bands[1,3] -= bands[1,0]*bands[0,3]
    bands[0,0] = 0.0
    bands[1,2] /= bands[1,1]
    bands[1,3] /= bands[1,1]
    bands[1,1] = 1.0

    # Now do rows 2 through M+1
    for row in range(2,M+1):
        bands[row,1] -= bands[row,0]*bands[row-1,2]
        bands[row,3] -= bands[row,0]*bands[row-1,3]
        bands[row,2] /= bands[row,1]
        bands[row,3] /= bands[row,1]
        bands[row,0] = 0.0
        bands[row,1] = 1.0


    # Do last row
    bands[M+1,1] -= bands[M+1,0]*bands[M-1,2]
    bands[M+1,3] -= bands[M+1,0]*bands[M-1,3]
    bands[M+1,2] -= bands[M+1,1]*bands[M,2]
    bands[M+1,3] -= bands[M+1,1]*bands[M,3]
    bands[M+1,3] /= bands[M+1,2]
    bands[M+1,2] = 1.0

    coefs[M+1] = bands[(M+1),3]
    # Now back substitute up
    for row in range(M, 0, -1):
        coefs[row] = bands[row,3] - bands[row,2]*coefs[row+1]

    # Finish with first row
    coefs[0] = bands[0,3] - bands[0,1]*coefs[1] - bands[0,2]*coefs[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef find_coefs_1d(double delta_inv, int M, double[:] data, double[:] coefs):

    cdef double[:] basis = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0, 0.0])

    cdef int i,j

    cdef double[:,:] bands = np.zeros((M+2,4))


    # Setup boundary conditions
    cdef double[:] abcd_left = np.zeros(4)
    cdef double[:] abcd_right = np.zeros(4)


    # Left boundary

    abcd_left[0] = 1.0 * delta_inv * delta_inv
    abcd_left[1] =-2.0 * delta_inv * delta_inv
    abcd_left[2] = 1.0 * delta_inv * delta_inv
    abcd_left[3] = 0

    # Right boundary
    abcd_right[0] = 1.0 *delta_inv * delta_inv
    abcd_right[1] =-2.0 *delta_inv * delta_inv
    abcd_right[2] = 1.0 *delta_inv * delta_inv
    abcd_right[3] = 0

    for i in range(4):
        bands[0,i] = abcd_left[i]
        bands[M+1,i] = abcd_right[i]

    for i in range(M):
        for j in range(3):
            bands[i+1,j] = basis[j]
            bands[i+1,3] = data[i]

    solve_deriv_interp_1d(bands, coefs)

def filter_coeffs_1d(double[:] dinv, double[:] data):

  M = data.shape[0]
  N = M+2

  cdef double[:] coefs = np.zeros(N)
  find_coefs_1d(dinv[0], M, data, coefs)

  return coefs

def filter_coeffs_2d(double[:] dinv, double[:,:] data):

    cdef int Mx = data.shape[0]
    cdef int My = data.shape[1]

    cdef int Nx = Mx+2
    cdef int Ny = My+2

    cdef double[:,:] coefs = np.zeros((Nx,Ny))

    cdef int iy, ix

    # First, solve in the X-direction
    for iy in range(My):
        # print(data[:,iy].size)
        # print(spline.coefs[:,iy].size)
        find_coefs_1d(dinv[0], Mx, data[:,iy], coefs[:,iy])

    # Now, solve in the Y-direction
    for ix in range(Nx):
        find_coefs_1d(dinv[1], My, coefs[ix,:], coefs[ix,:])

    return coefs

@cython.boundscheck(False)
@cython.wraparound(False)
cdef filter_coeffs_3d(double[:] dinv, double[:,:,:] data):

    cdef int Mx = data.shape[0]
    cdef int My = data.shape[1]
    cdef int Mz = data.shape[2]

    cdef int Nx = Mx+2
    cdef int Ny = My+2
    cdef int Nz = Mz+2

    cdef double [:,:,:] coefs = np.zeros((Nx,Ny,Nz))

    cdef int iy, ix, iz

    # First, solve in the X-direction
    for iy in range(My):
        for iz in range(Mz):
            find_coefs_1d(dinv[0], Mx, data[:,iy,iz], coefs[:,iy,iz])

    # Now, solve in the Y-direction
    for ix in range(Nx):
        for iz in range(Mz):
            find_coefs_1d(dinv[1], My, coefs[ix,:,iz], coefs[ix,:,iz])

    # Now, solve in the Z-direction
    for ix in range(Nx):
        for iy in range(Ny):
            find_coefs_1d(dinv[2], Mz, coefs[ix,iy,:], coefs[ix,iy,:])

    return coefs

class USpline:

    def __init__(self, smin, smax, orders, data):
        smin = np.array(smin,dtype=np.double)
        smax = np.array(smax,dtype=np.double)
        orders = np.array(orders,dtype=np.int)
        self.smin = smin
        self.smax = smax
        self.orders = orders
        self.d = len(smin)
        self.delta = (smax-smin)/(orders-1)
        self.delta_inv = 1.0/self.delta
        self.coefs = filter_data(self.delta_inv, data)


#
# def filter_coeffs_1d(double[:] dinv, double[:] data):
#
#   M = data.shape[0]
#   N = M+2
#   coefs = np.zeros(N)
#   coefs[:] = find_coefs_1d(dinv[0], M, data)
#
#   return coefs
#
#
#
# def filter_coeffs_2d(double[:] dinv, double[:,:] data):
#     Mx = data.shape[0]
#     My = data.shape[1]
#
#     Nx = Mx+2
#     Ny = My+2
#
#     cdef double [:,:] coefs = np.zeros((Nx,Ny))
#
#     cdef int iy, ix
#
#     # First, solve in the X-direction
#     for iy in range(My):
# #        find_coefs_1d(dinv[0], Mx, data[:,iy], coefs[:,iy])
#          find_coefs_1d(dinv[0], Mx, data[:,iy], coefs[:,iy+1])
#
#
#     # Now, solve in the Y-directiona
#     for ix in range(Nx):
#          find_coefs_1d(dinv[1], My, coefs[ix,:], coefs[ix,:])
#     return coefs
#
# def filter_coeffs_3d(double[:] dinv, double[:,:,:] data):
#
#     Mx = data.shape[0]
#     My = data.shape[1]
#     Mz = data.shape[2]
#
#     Nx = Mx+2
#     Ny = My+2
#     Nz = Mz+2
#
#     cdef double [:,:,:] coefs = np.zeros((Nx,Ny,Nz))
#
#     cdef int iy, ix, iz
#
#     # First, solve in the X-direction
#     for iy in range(My):
#         for iz in range(Mz):
#             find_coefs_1d(dinv[0], Mx, data[:,iy,iz], coefs[:,iy+1,iz+1])
#     # Now, solve in the Y-direction
#     for ix in range(Nx):
#         for iz in range(Mz):
#             find_coefs_1d(dinv[1], My, coefs[ix,:,iz+1], coefs[ix,:,iz+1] )
#
#     # Now, solve in the Z-direction
#     for ix in range(Nx):
#         for iy in range(Ny):
#             find_coefs_1d(dinv[2], Mz, coefs[ix,iy,:], coefs[ix,iy,:])
#
#     return coefs
#
# def filter_coeffs_4d(double[:] dinv, double[:,:,:,:] data):
#
#     M0 = data.shape[0]
#     M1 = data.shape[1]
#     M2 = data.shape[2]
#     M3 = data.shape[3]
#
#     N0 = M0+2
#     N1 = M1+2
#     N2 = M2+2
#     N3 = M3+2
#
#
#     N0 = M0+2;
#     N1 = M1+2;
#     N2 = M2+2;
#     N3 = M3+2;
#
#
#     cdef double [:,:,:,:] coefs = np.zeros((N0,N1,N2,N3))
#
#     cdef int i0, i1, i2, i3
#
#     # First, solve in the X-direction
#     for i1 in range(M1):
#         for i2 in range(M2):
#             for i3 in range(M3):
#                 find_coefs_1d(dinv[0], M0, data[:,i1,i2,i3], coefs[:,i1+1,i2+1,i3+1])
#     for i0 in range(N0):
#         for i2 in range(M2):
#             for i3 in range(M3):
#                 find_coefs_1d(dinv[1], M1, coefs[i0,:,i2+1,i3+1], coefs[i0,:,i2+1,i3+1])
#
#     for i0 in range(N0):
#         for i1 in range(N1):
#             for i3 in range(M3):
#                 find_coefs_1d(dinv[2], M2, coefs[i0,i1,:,i3+1], coefs[i0,i1,:,i3+1])
#
#     for i0 in range(N0):
#         for i1 in range(N1):
#             for i2 in range(N2):
#                 find_coefs_1d(dinv[3], M3, coefs[i0,i1,i2,:], coefs[i0,i1,i2,:])
#
#     return coefs
#
#
#
def filter_data(dinv, data):
    if len(dinv) == 1:
        return filter_coeffs_1d(dinv,data)
    elif len(dinv) == 2:
        return filter_coeffs_2d(dinv,data)
    elif len(dinv) == 3:
        return filter_coeffs_3d(dinv,data)
    # elif len(dinv) == 4:
    #     return filter_coeffs_4d(dinv,data)
#

