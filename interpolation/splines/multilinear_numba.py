import numpy as np
from numba import njit

def vec_multilinear_interpolation(smin, smax, orders, values, s, out=None):

    d = smin.shape[0]
    N = s.shape[0]
    n_sp = values.shape[-1]

    if out is None:
        out = np.zeros((N,n_sp))

    for i in range(n_sp):
        multilinear_interpolation(smin, smax, orders, values[...,i], s, out[:,i])

    return out

def multilinear_interpolation(smin, smax, orders, values, s, out=None):

    d = smin.shape[0]
    N = s.shape[0]

    if out is None:
        out = np.zeros(N)

    if False:
        pass
    elif d==1:
        multilinear_interpolation_1d(smin, smax, orders, values, s, out)
    elif d==2:
        multilinear_interpolation_2d(smin, smax, orders, values, s, out)
    elif d==3:
        multilinear_interpolation_3d(smin, smax, orders, values, s, out)
    elif d==4:
        multilinear_interpolation_4d(smin, smax, orders, values, s, out)
    else:
        raise Exception("Can't interpolate in dimension strictly greater than 5")

    return out


@njit(cache=True)
def multilinear_interpolation_1d(smin, smax, orders, V, s, output):

    d = 1
    N = s.shape[0]

    order_0 = orders[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0

            # v_ij: values on vertices of hypercube "containing" the point
            v_0 = V[(q_0)]
            v_1 = V[(q_0+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*(v_0) + (lam_0)*(v_1)


@njit(cache=True)
def multilinear_interpolation_2d(smin, smax, orders, V, s, output):

    d = 2
    N = s.shape[0]

    order_0 = orders[0]
    order_1 = orders[1]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1

            # v_ij: values on vertices of hypercube "containing" the point
            v_00 = V[(q_0) , (q_1)]
            v_01 = V[(q_0) , (q_1+1)]
            v_10 = V[(q_0+1) , (q_1)]
            v_11 = V[(q_0+1) , (q_1+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))


@njit(cache=True)
def multilinear_interpolation_3d(smin, smax, orders, V, s, output):

    d = 3
    N = s.shape[0]

    order_0 = orders[0]
    order_1 = orders[1]
    order_2 = orders[2]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]
            s_2 = s[ n, 2 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
            sn_2 = (s_2-smin[2])/(smax[2]-smin[2])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )
            q_2 = max( min( int(sn_2 *(order_2-1)), (order_2-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1
            lam_2 = sn_2*(order_2-1) - q_2

            # v_ij: values on vertices of hypercube "containing" the point
            v_000 = V[(q_0) , (q_1) , (q_2)]
            v_001 = V[(q_0) , (q_1) , (q_2+1)]
            v_010 = V[(q_0) , (q_1+1) , (q_2)]
            v_011 = V[(q_0) , (q_1+1) , (q_2+1)]
            v_100 = V[(q_0+1) , (q_1) , (q_2)]
            v_101 = V[(q_0+1) , (q_1) , (q_2+1)]
            v_110 = V[(q_0+1) , (q_1+1) , (q_2)]
            v_111 = V[(q_0+1) , (q_1+1) , (q_2+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*((1-lam_2)*(v_000) + (lam_2)*(v_001)) + (lam_1)*((1-lam_2)*(v_010) + (lam_2)*(v_011))) + (lam_0)*((1-lam_1)*((1-lam_2)*(v_100) + (lam_2)*(v_101)) + (lam_1)*((1-lam_2)*(v_110) + (lam_2)*(v_111)))


@njit(cache=True)
def multilinear_interpolation_4d(smin, smax, orders, V, s, output):

    d = 4
    N = s.shape[0]

    order_0 = orders[0]
    order_1 = orders[1]
    order_2 = orders[2]
    order_3 = orders[3]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]
            s_2 = s[ n, 2 ]
            s_3 = s[ n, 3 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
            sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
            sn_3 = (s_3-smin[3])/(smax[3]-smin[3])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )
            q_2 = max( min( int(sn_2 *(order_2-1)), (order_2-2) ), 0 )
            q_3 = max( min( int(sn_3 *(order_3-1)), (order_3-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1
            lam_2 = sn_2*(order_2-1) - q_2
            lam_3 = sn_3*(order_3-1) - q_3

            # v_ij: values on vertices of hypercube "containing" the point
            v_0000 = V[(q_0) , (q_1) , (q_2) , (q_3)]
            v_0001 = V[(q_0) , (q_1) , (q_2) , (q_3+1)]
            v_0010 = V[(q_0) , (q_1) , (q_2+1) , (q_3)]
            v_0011 = V[(q_0) , (q_1) , (q_2+1) , (q_3+1)]
            v_0100 = V[(q_0) , (q_1+1) , (q_2) , (q_3)]
            v_0101 = V[(q_0) , (q_1+1) , (q_2) , (q_3+1)]
            v_0110 = V[(q_0) , (q_1+1) , (q_2+1) , (q_3)]
            v_0111 = V[(q_0) , (q_1+1) , (q_2+1) , (q_3+1)]
            v_1000 = V[(q_0+1) , (q_1) , (q_2) , (q_3)]
            v_1001 = V[(q_0+1) , (q_1) , (q_2) , (q_3+1)]
            v_1010 = V[(q_0+1) , (q_1) , (q_2+1) , (q_3)]
            v_1011 = V[(q_0+1) , (q_1) , (q_2+1) , (q_3+1)]
            v_1100 = V[(q_0+1) , (q_1+1) , (q_2) , (q_3)]
            v_1101 = V[(q_0+1) , (q_1+1) , (q_2) , (q_3+1)]
            v_1110 = V[(q_0+1) , (q_1+1) , (q_2+1) , (q_3)]
            v_1111 = V[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_0000) + (lam_3)*(v_0001)) + (lam_2)*((1-lam_3)*(v_0010) + (lam_3)*(v_0011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_0100) + (lam_3)*(v_0101)) + (lam_2)*((1-lam_3)*(v_0110) + (lam_3)*(v_0111)))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_1000) + (lam_3)*(v_1001)) + (lam_2)*((1-lam_3)*(v_1010) + (lam_3)*(v_1011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_1100) + (lam_3)*(v_1101)) + (lam_2)*((1-lam_3)*(v_1110) + (lam_3)*(v_1111))))


@njit(cache=True)
def multilinear_interpolation_5d(smin, smax, orders, V, s, output):

    d = 5
    N = s.shape[0]

    order_0 = orders[0]
    order_1 = orders[1]
    order_2 = orders[2]
    order_3 = orders[3]
    order_4 = orders[4]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]
            s_2 = s[ n, 2 ]
            s_3 = s[ n, 3 ]
            s_4 = s[ n, 4 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
            sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
            sn_3 = (s_3-smin[3])/(smax[3]-smin[3])
            sn_4 = (s_4-smin[4])/(smax[4]-smin[4])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )
            q_2 = max( min( int(sn_2 *(order_2-1)), (order_2-2) ), 0 )
            q_3 = max( min( int(sn_3 *(order_3-1)), (order_3-2) ), 0 )
            q_4 = max( min( int(sn_4 *(order_4-1)), (order_4-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1
            lam_2 = sn_2*(order_2-1) - q_2
            lam_3 = sn_3*(order_3-1) - q_3
            lam_4 = sn_4*(order_4-1) - q_4

            # v_ij: values on vertices of hypercube "containing" the point
            v_00000 = V[(q_0) , (q_1) , (q_2) , (q_3) , (q_4)]
            v_00001 = V[(q_0) , (q_1) , (q_2) , (q_3) , (q_4+1)]
            v_00010 = V[(q_0) , (q_1) , (q_2) , (q_3+1) , (q_4)]
            v_00011 = V[(q_0) , (q_1) , (q_2) , (q_3+1) , (q_4+1)]
            v_00100 = V[(q_0) , (q_1) , (q_2+1) , (q_3) , (q_4)]
            v_00101 = V[(q_0) , (q_1) , (q_2+1) , (q_3) , (q_4+1)]
            v_00110 = V[(q_0) , (q_1) , (q_2+1) , (q_3+1) , (q_4)]
            v_00111 = V[(q_0) , (q_1) , (q_2+1) , (q_3+1) , (q_4+1)]
            v_01000 = V[(q_0) , (q_1+1) , (q_2) , (q_3) , (q_4)]
            v_01001 = V[(q_0) , (q_1+1) , (q_2) , (q_3) , (q_4+1)]
            v_01010 = V[(q_0) , (q_1+1) , (q_2) , (q_3+1) , (q_4)]
            v_01011 = V[(q_0) , (q_1+1) , (q_2) , (q_3+1) , (q_4+1)]
            v_01100 = V[(q_0) , (q_1+1) , (q_2+1) , (q_3) , (q_4)]
            v_01101 = V[(q_0) , (q_1+1) , (q_2+1) , (q_3) , (q_4+1)]
            v_01110 = V[(q_0) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4)]
            v_01111 = V[(q_0) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4+1)]
            v_10000 = V[(q_0+1) , (q_1) , (q_2) , (q_3) , (q_4)]
            v_10001 = V[(q_0+1) , (q_1) , (q_2) , (q_3) , (q_4+1)]
            v_10010 = V[(q_0+1) , (q_1) , (q_2) , (q_3+1) , (q_4)]
            v_10011 = V[(q_0+1) , (q_1) , (q_2) , (q_3+1) , (q_4+1)]
            v_10100 = V[(q_0+1) , (q_1) , (q_2+1) , (q_3) , (q_4)]
            v_10101 = V[(q_0+1) , (q_1) , (q_2+1) , (q_3) , (q_4+1)]
            v_10110 = V[(q_0+1) , (q_1) , (q_2+1) , (q_3+1) , (q_4)]
            v_10111 = V[(q_0+1) , (q_1) , (q_2+1) , (q_3+1) , (q_4+1)]
            v_11000 = V[(q_0+1) , (q_1+1) , (q_2) , (q_3) , (q_4)]
            v_11001 = V[(q_0+1) , (q_1+1) , (q_2) , (q_3) , (q_4+1)]
            v_11010 = V[(q_0+1) , (q_1+1) , (q_2) , (q_3+1) , (q_4)]
            v_11011 = V[(q_0+1) , (q_1+1) , (q_2) , (q_3+1) , (q_4+1)]
            v_11100 = V[(q_0+1) , (q_1+1) , (q_2+1) , (q_3) , (q_4)]
            v_11101 = V[(q_0+1) , (q_1+1) , (q_2+1) , (q_3) , (q_4+1)]
            v_11110 = V[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4)]
            v_11111 = V[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_00000) + (lam_4)*(v_00001)) + (lam_3)*((1-lam_4)*(v_00010) + (lam_4)*(v_00011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_00100) + (lam_4)*(v_00101)) + (lam_3)*((1-lam_4)*(v_00110) + (lam_4)*(v_00111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_01000) + (lam_4)*(v_01001)) + (lam_3)*((1-lam_4)*(v_01010) + (lam_4)*(v_01011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_01100) + (lam_4)*(v_01101)) + (lam_3)*((1-lam_4)*(v_01110) + (lam_4)*(v_01111))))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_10000) + (lam_4)*(v_10001)) + (lam_3)*((1-lam_4)*(v_10010) + (lam_4)*(v_10011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_10100) + (lam_4)*(v_10101)) + (lam_3)*((1-lam_4)*(v_10110) + (lam_4)*(v_10111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_11000) + (lam_4)*(v_11001)) + (lam_3)*((1-lam_4)*(v_11010) + (lam_4)*(v_11011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_11100) + (lam_4)*(v_11101)) + (lam_3)*((1-lam_4)*(v_11110) + (lam_4)*(v_11111)))))
