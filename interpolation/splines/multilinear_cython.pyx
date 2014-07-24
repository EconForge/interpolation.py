
# generation options

# some helper functions

from libc.math cimport fmin, fmax, floor
cimport cython
from cython.parallel import prange,parallel

cimport numpy as np
import numpy as np

ctypedef fused floating:
    float
    double

def multilinear_interpolation(floating[:] smin, floating[:] smax, long[:] orders, floating[:,::1] values, floating[:,::1] s):

    cdef int d = np.size(s,0)
    cdef int n_s = np.size(s,1)
    cdef int n_v = np.size(values,0)

    if floating is float:
        dtype = np.single
    else:
        dtype = np.double

    cdef floating[:,::1] result = np.zeros((n_v,n_s), dtype=dtype)
    cdef floating[:] vals
    cdef floating[:] res


    for i in range(n_v):
        vals = values[i,:]
        res = result[i,:]
        if False:
            pass
        elif d==1:
            multilinear_interpolation_1d(smin, smax, orders, vals, n_s, s, res)
        elif d==2:
            multilinear_interpolation_2d(smin, smax, orders, vals, n_s, s, res)
        elif d==3:
            multilinear_interpolation_3d(smin, smax, orders, vals, n_s, s, res)
        elif d==4:
            multilinear_interpolation_4d(smin, smax, orders, vals, n_s, s, res)
        else:
            raise Exception("Can't interpolate in dimension strictly greater than 5")

    return np.array(result,dtype=dtype)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef multilinear_interpolation_1d(floating[:] smin, floating[:] smax,
                                  long[:] orders, floating[:] V,
                                  int n_s, floating[:,::1] s, floating[:] output):

    cdef int d = 1

    cdef int i


    cdef floating lam_0, s_0, sn_0, snt_0
    cdef int order_0 = orders[0]
    cdef int q_0
    cdef floating v_0
    cdef floating v_1
    with nogil, parallel():
        for i in prange(n_s):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ 0 , i ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0

            # v_ij: values on vertices of hypercube "containing" the point
            v_0 = V[(q_0)]
            v_1 = V[(q_0+1)]

            # interpolated/extrapolated value
            output[i] = (1-lam_0)*(v_0) + (lam_0)*(v_1)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef multilinear_interpolation_2d(floating[:] smin, floating[:] smax,
                                  long[:] orders, floating[:] V,
                                  int n_s, floating[:,::1] s, floating[:] output):

    cdef int d = 2

    cdef int i


    cdef floating lam_0, s_0, sn_0, snt_0
    cdef floating lam_1, s_1, sn_1, snt_1
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int q_0
    cdef int q_1
    cdef int M_0 = order_1
    cdef floating v_00
    cdef floating v_01
    cdef floating v_10
    cdef floating v_11
    with nogil, parallel():
        for i in prange(n_s):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ 0 , i ]
            s_1 = s[ 1 , i ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1

            # v_ij: values on vertices of hypercube "containing" the point
            v_00 = V[M_0*(q_0) + (q_1)]
            v_01 = V[M_0*(q_0) + (q_1+1)]
            v_10 = V[M_0*(q_0+1) + (q_1)]
            v_11 = V[M_0*(q_0+1) + (q_1+1)]

            # interpolated/extrapolated value
            output[i] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef multilinear_interpolation_3d(floating[:] smin, floating[:] smax,
                                  long[:] orders, floating[:] V,
                                  int n_s, floating[:,::1] s, floating[:] output):

    cdef int d = 3

    cdef int i


    cdef floating lam_0, s_0, sn_0, snt_0
    cdef floating lam_1, s_1, sn_1, snt_1
    cdef floating lam_2, s_2, sn_2, snt_2
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef int q_0
    cdef int q_1
    cdef int q_2
    cdef int M_0 = order_1*order_2
    cdef int M_1 = order_2
    cdef floating v_000
    cdef floating v_001
    cdef floating v_010
    cdef floating v_011
    cdef floating v_100
    cdef floating v_101
    cdef floating v_110
    cdef floating v_111
    with nogil, parallel():
        for i in prange(n_s):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ 0 , i ]
            s_1 = s[ 1 , i ]
            s_2 = s[ 2 , i ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
            sn_2 = (s_2-smin[2])/(smax[2]-smin[2])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-2) ), 0 )
            q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1
            lam_2 = sn_2*(order_2-1) - q_2

            # v_ij: values on vertices of hypercube "containing" the point
            v_000 = V[M_0*(q_0) + M_1*(q_1) + (q_2)]
            v_001 = V[M_0*(q_0) + M_1*(q_1) + (q_2+1)]
            v_010 = V[M_0*(q_0) + M_1*(q_1+1) + (q_2)]
            v_011 = V[M_0*(q_0) + M_1*(q_1+1) + (q_2+1)]
            v_100 = V[M_0*(q_0+1) + M_1*(q_1) + (q_2)]
            v_101 = V[M_0*(q_0+1) + M_1*(q_1) + (q_2+1)]
            v_110 = V[M_0*(q_0+1) + M_1*(q_1+1) + (q_2)]
            v_111 = V[M_0*(q_0+1) + M_1*(q_1+1) + (q_2+1)]

            # interpolated/extrapolated value
            output[i] = (1-lam_0)*((1-lam_1)*((1-lam_2)*(v_000) + (lam_2)*(v_001)) + (lam_1)*((1-lam_2)*(v_010) + (lam_2)*(v_011))) + (lam_0)*((1-lam_1)*((1-lam_2)*(v_100) + (lam_2)*(v_101)) + (lam_1)*((1-lam_2)*(v_110) + (lam_2)*(v_111)))


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef multilinear_interpolation_4d(floating[:] smin, floating[:] smax,
                                  long[:] orders, floating[:] V,
                                  int n_s, floating[:,::1] s, floating[:] output):

    cdef int d = 4

    cdef int i


    cdef floating lam_0, s_0, sn_0, snt_0
    cdef floating lam_1, s_1, sn_1, snt_1
    cdef floating lam_2, s_2, sn_2, snt_2
    cdef floating lam_3, s_3, sn_3, snt_3
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef int order_3 = orders[3]
    cdef int q_0
    cdef int q_1
    cdef int q_2
    cdef int q_3
    cdef int M_0 = order_1*order_2*order_3
    cdef int M_1 = order_2*order_3
    cdef int M_2 = order_3
    cdef floating v_0000
    cdef floating v_0001
    cdef floating v_0010
    cdef floating v_0011
    cdef floating v_0100
    cdef floating v_0101
    cdef floating v_0110
    cdef floating v_0111
    cdef floating v_1000
    cdef floating v_1001
    cdef floating v_1010
    cdef floating v_1011
    cdef floating v_1100
    cdef floating v_1101
    cdef floating v_1110
    cdef floating v_1111
    with nogil, parallel():
        for i in prange(n_s):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ 0 , i ]
            s_1 = s[ 1 , i ]
            s_2 = s[ 2 , i ]
            s_3 = s[ 3 , i ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
            sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
            sn_3 = (s_3-smin[3])/(smax[3]-smin[3])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-2) ), 0 )
            q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-2) ), 0 )
            q_3 = max( min( <int>(sn_3 *(order_3-1)), (order_3-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1
            lam_2 = sn_2*(order_2-1) - q_2
            lam_3 = sn_3*(order_3-1) - q_3

            # v_ij: values on vertices of hypercube "containing" the point
            v_0000 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2) + (q_3)]
            v_0001 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2) + (q_3+1)]
            v_0010 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2+1) + (q_3)]
            v_0011 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2+1) + (q_3+1)]
            v_0100 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2) + (q_3)]
            v_0101 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2) + (q_3+1)]
            v_0110 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2+1) + (q_3)]
            v_0111 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2+1) + (q_3+1)]
            v_1000 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2) + (q_3)]
            v_1001 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2) + (q_3+1)]
            v_1010 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2+1) + (q_3)]
            v_1011 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2+1) + (q_3+1)]
            v_1100 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2) + (q_3)]
            v_1101 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2) + (q_3+1)]
            v_1110 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2+1) + (q_3)]
            v_1111 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2+1) + (q_3+1)]

            # interpolated/extrapolated value
            output[i] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_0000) + (lam_3)*(v_0001)) + (lam_2)*((1-lam_3)*(v_0010) + (lam_3)*(v_0011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_0100) + (lam_3)*(v_0101)) + (lam_2)*((1-lam_3)*(v_0110) + (lam_3)*(v_0111)))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_1000) + (lam_3)*(v_1001)) + (lam_2)*((1-lam_3)*(v_1010) + (lam_3)*(v_1011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_1100) + (lam_3)*(v_1101)) + (lam_2)*((1-lam_3)*(v_1110) + (lam_3)*(v_1111))))


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef multilinear_interpolation_5d(floating[:] smin, floating[:] smax,
                                  long[:] orders, floating[:] V,
                                  int n_s, floating[:,::1] s, floating[:] output):

    cdef int d = 5

    cdef int i


    cdef floating lam_0, s_0, sn_0, snt_0
    cdef floating lam_1, s_1, sn_1, snt_1
    cdef floating lam_2, s_2, sn_2, snt_2
    cdef floating lam_3, s_3, sn_3, snt_3
    cdef floating lam_4, s_4, sn_4, snt_4
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef int order_3 = orders[3]
    cdef int order_4 = orders[4]
    cdef int q_0
    cdef int q_1
    cdef int q_2
    cdef int q_3
    cdef int q_4
    cdef int M_0 = order_1*order_2*order_3*order_4
    cdef int M_1 = order_2*order_3*order_4
    cdef int M_2 = order_3*order_4
    cdef int M_3 = order_4
    cdef floating v_00000
    cdef floating v_00001
    cdef floating v_00010
    cdef floating v_00011
    cdef floating v_00100
    cdef floating v_00101
    cdef floating v_00110
    cdef floating v_00111
    cdef floating v_01000
    cdef floating v_01001
    cdef floating v_01010
    cdef floating v_01011
    cdef floating v_01100
    cdef floating v_01101
    cdef floating v_01110
    cdef floating v_01111
    cdef floating v_10000
    cdef floating v_10001
    cdef floating v_10010
    cdef floating v_10011
    cdef floating v_10100
    cdef floating v_10101
    cdef floating v_10110
    cdef floating v_10111
    cdef floating v_11000
    cdef floating v_11001
    cdef floating v_11010
    cdef floating v_11011
    cdef floating v_11100
    cdef floating v_11101
    cdef floating v_11110
    cdef floating v_11111
    with nogil, parallel():
        for i in prange(n_s):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ 0 , i ]
            s_1 = s[ 1 , i ]
            s_2 = s[ 2 , i ]
            s_3 = s[ 3 , i ]
            s_4 = s[ 4 , i ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
            sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
            sn_3 = (s_3-smin[3])/(smax[3]-smin[3])
            sn_4 = (s_4-smin[4])/(smax[4]-smin[4])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-2) ), 0 )
            q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-2) ), 0 )
            q_3 = max( min( <int>(sn_3 *(order_3-1)), (order_3-2) ), 0 )
            q_4 = max( min( <int>(sn_4 *(order_4-1)), (order_4-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1
            lam_2 = sn_2*(order_2-1) - q_2
            lam_3 = sn_3*(order_3-1) - q_3
            lam_4 = sn_4*(order_4-1) - q_4

            # v_ij: values on vertices of hypercube "containing" the point
            v_00000 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3) + (q_4)]
            v_00001 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3) + (q_4+1)]
            v_00010 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3+1) + (q_4)]
            v_00011 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3+1) + (q_4+1)]
            v_00100 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3) + (q_4)]
            v_00101 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3) + (q_4+1)]
            v_00110 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4)]
            v_00111 = V[M_0*(q_0) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4+1)]
            v_01000 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3) + (q_4)]
            v_01001 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3) + (q_4+1)]
            v_01010 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3+1) + (q_4)]
            v_01011 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3+1) + (q_4+1)]
            v_01100 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3) + (q_4)]
            v_01101 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3) + (q_4+1)]
            v_01110 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4)]
            v_01111 = V[M_0*(q_0) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4+1)]
            v_10000 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3) + (q_4)]
            v_10001 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3) + (q_4+1)]
            v_10010 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3+1) + (q_4)]
            v_10011 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2) + M_3*(q_3+1) + (q_4+1)]
            v_10100 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3) + (q_4)]
            v_10101 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3) + (q_4+1)]
            v_10110 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4)]
            v_10111 = V[M_0*(q_0+1) + M_1*(q_1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4+1)]
            v_11000 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3) + (q_4)]
            v_11001 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3) + (q_4+1)]
            v_11010 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3+1) + (q_4)]
            v_11011 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2) + M_3*(q_3+1) + (q_4+1)]
            v_11100 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3) + (q_4)]
            v_11101 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3) + (q_4+1)]
            v_11110 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4)]
            v_11111 = V[M_0*(q_0+1) + M_1*(q_1+1) + M_2*(q_2+1) + M_3*(q_3+1) + (q_4+1)]

            # interpolated/extrapolated value
            output[i] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_00000) + (lam_4)*(v_00001)) + (lam_3)*((1-lam_4)*(v_00010) + (lam_4)*(v_00011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_00100) + (lam_4)*(v_00101)) + (lam_3)*((1-lam_4)*(v_00110) + (lam_4)*(v_00111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_01000) + (lam_4)*(v_01001)) + (lam_3)*((1-lam_4)*(v_01010) + (lam_4)*(v_01011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_01100) + (lam_4)*(v_01101)) + (lam_3)*((1-lam_4)*(v_01110) + (lam_4)*(v_01111))))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_10000) + (lam_4)*(v_10001)) + (lam_3)*((1-lam_4)*(v_10010) + (lam_4)*(v_10011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_10100) + (lam_4)*(v_10101)) + (lam_3)*((1-lam_4)*(v_10110) + (lam_4)*(v_10111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_11000) + (lam_4)*(v_11001)) + (lam_3)*((1-lam_4)*(v_11010) + (lam_4)*(v_11011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_11100) + (lam_4)*(v_11101)) + (lam_3)*((1-lam_4)*(v_11110) + (lam_4)*(v_11111)))))


