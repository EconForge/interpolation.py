import numpy as np
from numba import njit

@njit(cache=True)
def multilinear_irregular_1d_nonvec(x0, y, u0):

    order_0 = x0.shape[0]

    # (s_1, ..., s_d) : evaluation point
    u_0 = u0

    # q_k : index of the interval "containing" s_k
    q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )

    # lam_k : barycentric coordinate in interval k
    lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])

    # v_ij: values on vertices of hypercube "containing" the point
    v_0 = y[(q_0)]
    v_1 = y[(q_0+1)]

    # interpolated/extrapolated value
    return (1-lam_0)*(v_0) + (lam_0)*(v_1)

@njit(cache=True)
def multilinear_irregular_1d(x0, y, u, output):

    d = 1
    N = u.shape[0]

    order_0 = x0.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])

            # v_ij: values on vertices of hypercube "containing" the point
            v_0 = y[(q_0)]
            v_1 = y[(q_0+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*(v_0) + (lam_0)*(v_1)

@njit(cache=True)
def multilinear_irregular_2d(x0, x1, y, u, output):

    d = 2
    N = u.shape[0]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])

            # v_ij: values on vertices of hypercube "containing" the point
            v_00 = y[(q_0) , (q_1)]
            v_01 = y[(q_0) , (q_1+1)]
            v_10 = y[(q_0+1) , (q_1)]
            v_11 = y[(q_0+1) , (q_1+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))

@njit(cache=True)
def multilinear_irregular_3d(x0, x1, x2, y, u, output):

    d = 3
    N = u.shape[0]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]
    order_2 = x2.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]
            u_2 = u[ n, 2 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )
            q_2 = max( min( np.searchsorted(x2, u_2)-1, (order_2-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])
            lam_2 =  (u_2-x2[q_2])/(x2[q_2+1]-x2[q_2])

            # v_ij: values on vertices of hypercube "containing" the point
            v_000 = y[(q_0) , (q_1) , (q_2)]
            v_001 = y[(q_0) , (q_1) , (q_2+1)]
            v_010 = y[(q_0) , (q_1+1) , (q_2)]
            v_011 = y[(q_0) , (q_1+1) , (q_2+1)]
            v_100 = y[(q_0+1) , (q_1) , (q_2)]
            v_101 = y[(q_0+1) , (q_1) , (q_2+1)]
            v_110 = y[(q_0+1) , (q_1+1) , (q_2)]
            v_111 = y[(q_0+1) , (q_1+1) , (q_2+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*((1-lam_2)*(v_000) + (lam_2)*(v_001)) + (lam_1)*((1-lam_2)*(v_010) + (lam_2)*(v_011))) + (lam_0)*((1-lam_1)*((1-lam_2)*(v_100) + (lam_2)*(v_101)) + (lam_1)*((1-lam_2)*(v_110) + (lam_2)*(v_111)))

@njit(cache=True)
def multilinear_irregular_4d(x0, x1, x2, x3, y, u, output):

    d = 4
    N = u.shape[0]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]
    order_2 = x2.shape[0]
    order_3 = x3.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]
            u_2 = u[ n, 2 ]
            u_3 = u[ n, 3 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )
            q_2 = max( min( np.searchsorted(x2, u_2)-1, (order_2-2) ), 0 )
            q_3 = max( min( np.searchsorted(x3, u_3)-1, (order_3-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])
            lam_2 =  (u_2-x2[q_2])/(x2[q_2+1]-x2[q_2])
            lam_3 =  (u_3-x3[q_3])/(x3[q_3+1]-x3[q_3])

            # v_ij: values on vertices of hypercube "containing" the point
            v_0000 = y[(q_0) , (q_1) , (q_2) , (q_3)]
            v_0001 = y[(q_0) , (q_1) , (q_2) , (q_3+1)]
            v_0010 = y[(q_0) , (q_1) , (q_2+1) , (q_3)]
            v_0011 = y[(q_0) , (q_1) , (q_2+1) , (q_3+1)]
            v_0100 = y[(q_0) , (q_1+1) , (q_2) , (q_3)]
            v_0101 = y[(q_0) , (q_1+1) , (q_2) , (q_3+1)]
            v_0110 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3)]
            v_0111 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3+1)]
            v_1000 = y[(q_0+1) , (q_1) , (q_2) , (q_3)]
            v_1001 = y[(q_0+1) , (q_1) , (q_2) , (q_3+1)]
            v_1010 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3)]
            v_1011 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3+1)]
            v_1100 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3)]
            v_1101 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3+1)]
            v_1110 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3)]
            v_1111 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_0000) + (lam_3)*(v_0001)) + (lam_2)*((1-lam_3)*(v_0010) + (lam_3)*(v_0011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_0100) + (lam_3)*(v_0101)) + (lam_2)*((1-lam_3)*(v_0110) + (lam_3)*(v_0111)))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_1000) + (lam_3)*(v_1001)) + (lam_2)*((1-lam_3)*(v_1010) + (lam_3)*(v_1011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_1100) + (lam_3)*(v_1101)) + (lam_2)*((1-lam_3)*(v_1110) + (lam_3)*(v_1111))))

@njit(cache=True)
def multilinear_irregular_5d(x0, x1, x2, x3, x4, y, u, output):

    d = 5
    N = u.shape[0]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]
    order_2 = x2.shape[0]
    order_3 = x3.shape[0]
    order_4 = x4.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]
            u_2 = u[ n, 2 ]
            u_3 = u[ n, 3 ]
            u_4 = u[ n, 4 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )
            q_2 = max( min( np.searchsorted(x2, u_2)-1, (order_2-2) ), 0 )
            q_3 = max( min( np.searchsorted(x3, u_3)-1, (order_3-2) ), 0 )
            q_4 = max( min( np.searchsorted(x4, u_4)-1, (order_4-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])
            lam_2 =  (u_2-x2[q_2])/(x2[q_2+1]-x2[q_2])
            lam_3 =  (u_3-x3[q_3])/(x3[q_3+1]-x3[q_3])
            lam_4 =  (u_4-x4[q_4])/(x4[q_4+1]-x4[q_4])

            # v_ij: values on vertices of hypercube "containing" the point
            v_00000 = y[(q_0) , (q_1) , (q_2) , (q_3) , (q_4)]
            v_00001 = y[(q_0) , (q_1) , (q_2) , (q_3) , (q_4+1)]
            v_00010 = y[(q_0) , (q_1) , (q_2) , (q_3+1) , (q_4)]
            v_00011 = y[(q_0) , (q_1) , (q_2) , (q_3+1) , (q_4+1)]
            v_00100 = y[(q_0) , (q_1) , (q_2+1) , (q_3) , (q_4)]
            v_00101 = y[(q_0) , (q_1) , (q_2+1) , (q_3) , (q_4+1)]
            v_00110 = y[(q_0) , (q_1) , (q_2+1) , (q_3+1) , (q_4)]
            v_00111 = y[(q_0) , (q_1) , (q_2+1) , (q_3+1) , (q_4+1)]
            v_01000 = y[(q_0) , (q_1+1) , (q_2) , (q_3) , (q_4)]
            v_01001 = y[(q_0) , (q_1+1) , (q_2) , (q_3) , (q_4+1)]
            v_01010 = y[(q_0) , (q_1+1) , (q_2) , (q_3+1) , (q_4)]
            v_01011 = y[(q_0) , (q_1+1) , (q_2) , (q_3+1) , (q_4+1)]
            v_01100 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3) , (q_4)]
            v_01101 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3) , (q_4+1)]
            v_01110 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4)]
            v_01111 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4+1)]
            v_10000 = y[(q_0+1) , (q_1) , (q_2) , (q_3) , (q_4)]
            v_10001 = y[(q_0+1) , (q_1) , (q_2) , (q_3) , (q_4+1)]
            v_10010 = y[(q_0+1) , (q_1) , (q_2) , (q_3+1) , (q_4)]
            v_10011 = y[(q_0+1) , (q_1) , (q_2) , (q_3+1) , (q_4+1)]
            v_10100 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3) , (q_4)]
            v_10101 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3) , (q_4+1)]
            v_10110 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3+1) , (q_4)]
            v_10111 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3+1) , (q_4+1)]
            v_11000 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3) , (q_4)]
            v_11001 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3) , (q_4+1)]
            v_11010 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3+1) , (q_4)]
            v_11011 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3+1) , (q_4+1)]
            v_11100 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3) , (q_4)]
            v_11101 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3) , (q_4+1)]
            v_11110 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4)]
            v_11111 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_00000) + (lam_4)*(v_00001)) + (lam_3)*((1-lam_4)*(v_00010) + (lam_4)*(v_00011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_00100) + (lam_4)*(v_00101)) + (lam_3)*((1-lam_4)*(v_00110) + (lam_4)*(v_00111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_01000) + (lam_4)*(v_01001)) + (lam_3)*((1-lam_4)*(v_01010) + (lam_4)*(v_01011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_01100) + (lam_4)*(v_01101)) + (lam_3)*((1-lam_4)*(v_01110) + (lam_4)*(v_01111))))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_10000) + (lam_4)*(v_10001)) + (lam_3)*((1-lam_4)*(v_10010) + (lam_4)*(v_10011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_10100) + (lam_4)*(v_10101)) + (lam_3)*((1-lam_4)*(v_10110) + (lam_4)*(v_10111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_11000) + (lam_4)*(v_11001)) + (lam_3)*((1-lam_4)*(v_11010) + (lam_4)*(v_11011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_11100) + (lam_4)*(v_11101)) + (lam_3)*((1-lam_4)*(v_11110) + (lam_4)*(v_11111)))))

@njit(cache=True)
def multilinear_irregular_vector_1d(x0, y,u, output):

    d = 1
    N = u.shape[0]
    n_x = y.shape[1]

    order_0 = x0.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])

            for i_x in range(n_x):

                # v_ij: values on vertices of hypercube "containing" the point
                v_0 = y[(q_0), i_x]
                v_1 = y[(q_0+1), i_x]

                # interpolated/extrapolated value
                output[n, i_x] = (1-lam_0)*(v_0) + (lam_0)*(v_1)

@njit(cache=True)
def multilinear_irregular_vector_2d(x0, x1, y,u, output):

    d = 2
    N = u.shape[0]
    n_x = y.shape[2]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])

            for i_x in range(n_x):

                # v_ij: values on vertices of hypercube "containing" the point
                v_00 = y[(q_0) , (q_1), i_x]
                v_01 = y[(q_0) , (q_1+1), i_x]
                v_10 = y[(q_0+1) , (q_1), i_x]
                v_11 = y[(q_0+1) , (q_1+1), i_x]

                # interpolated/extrapolated value
                output[n, i_x] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))

@njit(cache=True)
def multilinear_irregular_vector_3d(x0, x1, x2, y,u, output):

    d = 3
    N = u.shape[0]
    n_x = y.shape[3]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]
    order_2 = x2.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]
            u_2 = u[ n, 2 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )
            q_2 = max( min( np.searchsorted(x2, u_2)-1, (order_2-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])
            lam_2 =  (u_2-x2[q_2])/(x2[q_2+1]-x2[q_2])

            for i_x in range(n_x):

                # v_ij: values on vertices of hypercube "containing" the point
                v_000 = y[(q_0) , (q_1) , (q_2), i_x]
                v_001 = y[(q_0) , (q_1) , (q_2+1), i_x]
                v_010 = y[(q_0) , (q_1+1) , (q_2), i_x]
                v_011 = y[(q_0) , (q_1+1) , (q_2+1), i_x]
                v_100 = y[(q_0+1) , (q_1) , (q_2), i_x]
                v_101 = y[(q_0+1) , (q_1) , (q_2+1), i_x]
                v_110 = y[(q_0+1) , (q_1+1) , (q_2), i_x]
                v_111 = y[(q_0+1) , (q_1+1) , (q_2+1), i_x]

                # interpolated/extrapolated value
                output[n, i_x] = (1-lam_0)*((1-lam_1)*((1-lam_2)*(v_000) + (lam_2)*(v_001)) + (lam_1)*((1-lam_2)*(v_010) + (lam_2)*(v_011))) + (lam_0)*((1-lam_1)*((1-lam_2)*(v_100) + (lam_2)*(v_101)) + (lam_1)*((1-lam_2)*(v_110) + (lam_2)*(v_111)))

@njit(cache=True)
def multilinear_irregular_vector_4d(x0, x1, x2, x3, y,u, output):

    d = 4
    N = u.shape[0]
    n_x = y.shape[4]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]
    order_2 = x2.shape[0]
    order_3 = x3.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]
            u_2 = u[ n, 2 ]
            u_3 = u[ n, 3 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )
            q_2 = max( min( np.searchsorted(x2, u_2)-1, (order_2-2) ), 0 )
            q_3 = max( min( np.searchsorted(x3, u_3)-1, (order_3-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])
            lam_2 =  (u_2-x2[q_2])/(x2[q_2+1]-x2[q_2])
            lam_3 =  (u_3-x3[q_3])/(x3[q_3+1]-x3[q_3])

            for i_x in range(n_x):

                # v_ij: values on vertices of hypercube "containing" the point
                v_0000 = y[(q_0) , (q_1) , (q_2) , (q_3), i_x]
                v_0001 = y[(q_0) , (q_1) , (q_2) , (q_3+1), i_x]
                v_0010 = y[(q_0) , (q_1) , (q_2+1) , (q_3), i_x]
                v_0011 = y[(q_0) , (q_1) , (q_2+1) , (q_3+1), i_x]
                v_0100 = y[(q_0) , (q_1+1) , (q_2) , (q_3), i_x]
                v_0101 = y[(q_0) , (q_1+1) , (q_2) , (q_3+1), i_x]
                v_0110 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3), i_x]
                v_0111 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3+1), i_x]
                v_1000 = y[(q_0+1) , (q_1) , (q_2) , (q_3), i_x]
                v_1001 = y[(q_0+1) , (q_1) , (q_2) , (q_3+1), i_x]
                v_1010 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3), i_x]
                v_1011 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3+1), i_x]
                v_1100 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3), i_x]
                v_1101 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3+1), i_x]
                v_1110 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3), i_x]
                v_1111 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1), i_x]

                # interpolated/extrapolated value
                output[n, i_x] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_0000) + (lam_3)*(v_0001)) + (lam_2)*((1-lam_3)*(v_0010) + (lam_3)*(v_0011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_0100) + (lam_3)*(v_0101)) + (lam_2)*((1-lam_3)*(v_0110) + (lam_3)*(v_0111)))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_1000) + (lam_3)*(v_1001)) + (lam_2)*((1-lam_3)*(v_1010) + (lam_3)*(v_1011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_1100) + (lam_3)*(v_1101)) + (lam_2)*((1-lam_3)*(v_1110) + (lam_3)*(v_1111))))

@njit(cache=True)
def multilinear_irregular_vector_5d(x0, x1, x2, x3, x4, y,u, output):

    d = 5
    N = u.shape[0]
    n_x = y.shape[5]

    order_0 = x0.shape[0]
    order_1 = x1.shape[0]
    order_2 = x2.shape[0]
    order_3 = x3.shape[0]
    order_4 = x4.shape[0]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            u_0 = u[ n, 0 ]
            u_1 = u[ n, 1 ]
            u_2 = u[ n, 2 ]
            u_3 = u[ n, 3 ]
            u_4 = u[ n, 4 ]

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( np.searchsorted(x0, u_0)-1, (order_0-2) ), 0 )
            q_1 = max( min( np.searchsorted(x1, u_1)-1, (order_1-2) ), 0 )
            q_2 = max( min( np.searchsorted(x2, u_2)-1, (order_2-2) ), 0 )
            q_3 = max( min( np.searchsorted(x3, u_3)-1, (order_3-2) ), 0 )
            q_4 = max( min( np.searchsorted(x4, u_4)-1, (order_4-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 =  (u_0-x0[q_0])/(x0[q_0+1]-x0[q_0])
            lam_1 =  (u_1-x1[q_1])/(x1[q_1+1]-x1[q_1])
            lam_2 =  (u_2-x2[q_2])/(x2[q_2+1]-x2[q_2])
            lam_3 =  (u_3-x3[q_3])/(x3[q_3+1]-x3[q_3])
            lam_4 =  (u_4-x4[q_4])/(x4[q_4+1]-x4[q_4])

            for i_x in range(n_x):

                # v_ij: values on vertices of hypercube "containing" the point
                v_00000 = y[(q_0) , (q_1) , (q_2) , (q_3) , (q_4), i_x]
                v_00001 = y[(q_0) , (q_1) , (q_2) , (q_3) , (q_4+1), i_x]
                v_00010 = y[(q_0) , (q_1) , (q_2) , (q_3+1) , (q_4), i_x]
                v_00011 = y[(q_0) , (q_1) , (q_2) , (q_3+1) , (q_4+1), i_x]
                v_00100 = y[(q_0) , (q_1) , (q_2+1) , (q_3) , (q_4), i_x]
                v_00101 = y[(q_0) , (q_1) , (q_2+1) , (q_3) , (q_4+1), i_x]
                v_00110 = y[(q_0) , (q_1) , (q_2+1) , (q_3+1) , (q_4), i_x]
                v_00111 = y[(q_0) , (q_1) , (q_2+1) , (q_3+1) , (q_4+1), i_x]
                v_01000 = y[(q_0) , (q_1+1) , (q_2) , (q_3) , (q_4), i_x]
                v_01001 = y[(q_0) , (q_1+1) , (q_2) , (q_3) , (q_4+1), i_x]
                v_01010 = y[(q_0) , (q_1+1) , (q_2) , (q_3+1) , (q_4), i_x]
                v_01011 = y[(q_0) , (q_1+1) , (q_2) , (q_3+1) , (q_4+1), i_x]
                v_01100 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3) , (q_4), i_x]
                v_01101 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3) , (q_4+1), i_x]
                v_01110 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4), i_x]
                v_01111 = y[(q_0) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4+1), i_x]
                v_10000 = y[(q_0+1) , (q_1) , (q_2) , (q_3) , (q_4), i_x]
                v_10001 = y[(q_0+1) , (q_1) , (q_2) , (q_3) , (q_4+1), i_x]
                v_10010 = y[(q_0+1) , (q_1) , (q_2) , (q_3+1) , (q_4), i_x]
                v_10011 = y[(q_0+1) , (q_1) , (q_2) , (q_3+1) , (q_4+1), i_x]
                v_10100 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3) , (q_4), i_x]
                v_10101 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3) , (q_4+1), i_x]
                v_10110 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3+1) , (q_4), i_x]
                v_10111 = y[(q_0+1) , (q_1) , (q_2+1) , (q_3+1) , (q_4+1), i_x]
                v_11000 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3) , (q_4), i_x]
                v_11001 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3) , (q_4+1), i_x]
                v_11010 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3+1) , (q_4), i_x]
                v_11011 = y[(q_0+1) , (q_1+1) , (q_2) , (q_3+1) , (q_4+1), i_x]
                v_11100 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3) , (q_4), i_x]
                v_11101 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3) , (q_4+1), i_x]
                v_11110 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4), i_x]
                v_11111 = y[(q_0+1) , (q_1+1) , (q_2+1) , (q_3+1) , (q_4+1), i_x]

                # interpolated/extrapolated value
                output[n, i_x] = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_00000) + (lam_4)*(v_00001)) + (lam_3)*((1-lam_4)*(v_00010) + (lam_4)*(v_00011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_00100) + (lam_4)*(v_00101)) + (lam_3)*((1-lam_4)*(v_00110) + (lam_4)*(v_00111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_01000) + (lam_4)*(v_01001)) + (lam_3)*((1-lam_4)*(v_01010) + (lam_4)*(v_01011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_01100) + (lam_4)*(v_01101)) + (lam_3)*((1-lam_4)*(v_01110) + (lam_4)*(v_01111))))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_10000) + (lam_4)*(v_10001)) + (lam_3)*((1-lam_4)*(v_10010) + (lam_4)*(v_10011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_10100) + (lam_4)*(v_10101)) + (lam_3)*((1-lam_4)*(v_10110) + (lam_4)*(v_10111)))) + (lam_1)*((1-lam_2)*((1-lam_3)*((1-lam_4)*(v_11000) + (lam_4)*(v_11001)) + (lam_3)*((1-lam_4)*(v_11010) + (lam_4)*(v_11011))) + (lam_2)*((1-lam_3)*((1-lam_4)*(v_11100) + (lam_4)*(v_11101)) + (lam_3)*((1-lam_4)*(v_11110) + (lam_4)*(v_11111)))))
