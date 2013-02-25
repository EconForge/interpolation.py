from libc.math cimport fmin, fmax, floor

cimport numpy as np
import numpy as np



def multilinear_interpolation_float(np.ndarray[np.float_t, ndim=1] smin, np.ndarray[np.float_t, ndim=1] smax, np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.float_t, ndim=2] values, np.ndarray[np.float_t, ndim=2] s):

    cdef int d = np.size(s,0)
    cdef int n_s = np.size(s,1)
    cdef int n_v = np.size(values,0)

    cdef np.ndarray[np.float_t, ndim=2] result = np.zeros((n_v,n_s))
    cdef np.ndarray[np.float_t, ndim=1] vals
    cdef np.ndarray[np.float_t, ndim=1] res

    for i in range(n_v):
        vals = values[i,:]
        res = result[i,:]
        if False:
            pass
        elif d==1:
            multilinear_interpolation_1d_float(smin, smax, orders, vals, n_s, s, res)
        elif d==2:
            multilinear_interpolation_2d_float(smin, smax, orders, vals, n_s, s, res)
        elif d==3:
            multilinear_interpolation_3d_float(smin, smax, orders, vals, n_s, s, res)
    return result




def multilinear_interpolation_double(np.ndarray[np.double_t, ndim=1] smin, np.ndarray[np.double_t, ndim=1] smax, np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=2] values, np.ndarray[np.double_t, ndim=2] s):

    cdef int d = np.size(s,0)
    cdef int n_s = np.size(s,1)
    cdef int n_v = np.size(values,0)

    cdef np.ndarray[np.double_t, ndim=2] result = np.zeros((n_v,n_s))
    cdef np.ndarray[np.double_t, ndim=1] vals
    cdef np.ndarray[np.double_t, ndim=1] res

    for i in range(n_v):
        vals = values[i,:]
        res = result[i,:]
        if False:
            pass
        elif d==1:
            multilinear_interpolation_1d_double(smin, smax, orders, vals, n_s, s, res)
        elif d==2:
            multilinear_interpolation_2d_double(smin, smax, orders, vals, n_s, s, res)
        elif d==3:
            multilinear_interpolation_3d_double(smin, smax, orders, vals, n_s, s, res)
    return result




cdef multilinear_interpolation_1d_float(np.ndarray[np.float_t, ndim=1] a_smin, np.ndarray[np.float_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.float_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.float_t, ndim=2] a_s, np.ndarray[np.float_t, ndim=1] a_output):

    cdef int d = 1

    cdef float* smin = <float*> a_smin.data
    cdef float* smax = <float*> a_smax.data
    cdef float* V = <float*> a_V.data
    cdef float* s = <float*> a_s.data
    cdef float* output = <float*> a_output.data

    cdef int i
 

    cdef float lam_0, s_0, sn_0, snt_0
    cdef int order_0 = orders[0]
    cdef float v_0
    cdef float v_1
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )

        # lam_k : barycentric coordinate in interval k
        lam_0 = sn_0*(order_0-1) - q_0

        # v_ij: values on vertices of hypercube "containing" the point
        v_0 = V[(q_0)]
        v_1 = V[(q_0+1)]

        # interpolated/extrapolated value
        output[i] = (1-lam_0)*(v_0) + (lam_0)*(v_1)




cdef multilinear_interpolation_2d_float(np.ndarray[np.float_t, ndim=1] a_smin, np.ndarray[np.float_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.float_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.float_t, ndim=2] a_s, np.ndarray[np.float_t, ndim=1] a_output):

    cdef int d = 2

    cdef float* smin = <float*> a_smin.data
    cdef float* smax = <float*> a_smax.data
    cdef float* V = <float*> a_V.data
    cdef float* s = <float*> a_s.data
    cdef float* output = <float*> a_output.data

    cdef int i
 

    cdef float lam_0, s_0, sn_0, snt_0
    cdef float lam_1, s_1, sn_1, snt_1
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef M_0 = order_1
    cdef float v_00
    cdef float v_01
    cdef float v_10
    cdef float v_11
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]
        s_1 = s[ 1*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )

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




cdef multilinear_interpolation_3d_float(np.ndarray[np.float_t, ndim=1] a_smin, np.ndarray[np.float_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.float_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.float_t, ndim=2] a_s, np.ndarray[np.float_t, ndim=1] a_output):

    cdef int d = 3

    cdef float* smin = <float*> a_smin.data
    cdef float* smax = <float*> a_smax.data
    cdef float* V = <float*> a_V.data
    cdef float* s = <float*> a_s.data
    cdef float* output = <float*> a_output.data

    cdef int i
 

    cdef float lam_0, s_0, sn_0, snt_0
    cdef float lam_1, s_1, sn_1, snt_1
    cdef float lam_2, s_2, sn_2, snt_2
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef M_0 = order_1*order_2
    cdef M_1 = order_2
    cdef float v_000
    cdef float v_001
    cdef float v_010
    cdef float v_011
    cdef float v_100
    cdef float v_101
    cdef float v_110
    cdef float v_111
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]
        s_1 = s[ 1*n_s + i ]
        s_2 = s[ 2*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )
        q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-1) ), 0 )

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




cdef multilinear_interpolation_4d_float(np.ndarray[np.float_t, ndim=1] a_smin, np.ndarray[np.float_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.float_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.float_t, ndim=2] a_s, np.ndarray[np.float_t, ndim=1] a_output):

    cdef int d = 4

    cdef float* smin = <float*> a_smin.data
    cdef float* smax = <float*> a_smax.data
    cdef float* V = <float*> a_V.data
    cdef float* s = <float*> a_s.data
    cdef float* output = <float*> a_output.data

    cdef int i
 

    cdef float lam_0, s_0, sn_0, snt_0
    cdef float lam_1, s_1, sn_1, snt_1
    cdef float lam_2, s_2, sn_2, snt_2
    cdef float lam_3, s_3, sn_3, snt_3
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef int order_3 = orders[3]
    cdef M_0 = order_1*order_2*order_3
    cdef M_1 = order_2*order_3
    cdef M_2 = order_3
    cdef float v_0000
    cdef float v_0001
    cdef float v_0010
    cdef float v_0011
    cdef float v_0100
    cdef float v_0101
    cdef float v_0110
    cdef float v_0111
    cdef float v_1000
    cdef float v_1001
    cdef float v_1010
    cdef float v_1011
    cdef float v_1100
    cdef float v_1101
    cdef float v_1110
    cdef float v_1111
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]
        s_1 = s[ 1*n_s + i ]
        s_2 = s[ 2*n_s + i ]
        s_3 = s[ 3*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
        sn_3 = (s_3-smin[3])/(smax[3]-smin[3])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )
        q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-1) ), 0 )
        q_3 = max( min( <int>(sn_3 *(order_3-1)), (order_3-1) ), 0 )

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




cdef multilinear_interpolation_1d_double(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 1

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data

    cdef int i
 

    cdef double lam_0, s_0, sn_0, snt_0
    cdef int order_0 = orders[0]
    cdef double v_0
    cdef double v_1
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )

        # lam_k : barycentric coordinate in interval k
        lam_0 = sn_0*(order_0-1) - q_0

        # v_ij: values on vertices of hypercube "containing" the point
        v_0 = V[(q_0)]
        v_1 = V[(q_0+1)]

        # interpolated/extrapolated value
        output[i] = (1-lam_0)*(v_0) + (lam_0)*(v_1)




cdef multilinear_interpolation_2d_double(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 2

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data

    cdef int i
 

    cdef double lam_0, s_0, sn_0, snt_0
    cdef double lam_1, s_1, sn_1, snt_1
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef M_0 = order_1
    cdef double v_00
    cdef double v_01
    cdef double v_10
    cdef double v_11
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]
        s_1 = s[ 1*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )

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




cdef multilinear_interpolation_3d_double(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 3

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data

    cdef int i
 

    cdef double lam_0, s_0, sn_0, snt_0
    cdef double lam_1, s_1, sn_1, snt_1
    cdef double lam_2, s_2, sn_2, snt_2
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef M_0 = order_1*order_2
    cdef M_1 = order_2
    cdef double v_000
    cdef double v_001
    cdef double v_010
    cdef double v_011
    cdef double v_100
    cdef double v_101
    cdef double v_110
    cdef double v_111
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]
        s_1 = s[ 1*n_s + i ]
        s_2 = s[ 2*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )
        q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-1) ), 0 )

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




cdef multilinear_interpolation_4d_double(np.ndarray[np.double_t, ndim=1] a_smin, np.ndarray[np.double_t, ndim=1] a_smax,
                                  np.ndarray[np.int_t, ndim=1] orders, np.ndarray[np.double_t, ndim=1] a_V,
                                  int n_s, np.ndarray[np.double_t, ndim=2] a_s, np.ndarray[np.double_t, ndim=1] a_output):

    cdef int d = 4

    cdef double* smin = <double*> a_smin.data
    cdef double* smax = <double*> a_smax.data
    cdef double* V = <double*> a_V.data
    cdef double* s = <double*> a_s.data
    cdef double* output = <double*> a_output.data

    cdef int i
 

    cdef double lam_0, s_0, sn_0, snt_0
    cdef double lam_1, s_1, sn_1, snt_1
    cdef double lam_2, s_2, sn_2, snt_2
    cdef double lam_3, s_3, sn_3, snt_3
    cdef int order_0 = orders[0]
    cdef int order_1 = orders[1]
    cdef int order_2 = orders[2]
    cdef int order_3 = orders[3]
    cdef M_0 = order_1*order_2*order_3
    cdef M_1 = order_2*order_3
    cdef M_2 = order_3
    cdef double v_0000
    cdef double v_0001
    cdef double v_0010
    cdef double v_0011
    cdef double v_0100
    cdef double v_0101
    cdef double v_0110
    cdef double v_0111
    cdef double v_1000
    cdef double v_1001
    cdef double v_1010
    cdef double v_1011
    cdef double v_1100
    cdef double v_1101
    cdef double v_1110
    cdef double v_1111
    #pragma omp parallel for
    for i in range(n_s):

        # (s_1, ..., s_d) : evaluation point
        s_0 = s[ 0*n_s + i ]
        s_1 = s[ 1*n_s + i ]
        s_2 = s[ 2*n_s + i ]
        s_3 = s[ 3*n_s + i ]

        # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
        sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
        sn_1 = (s_1-smin[1])/(smax[1]-smin[1])
        sn_2 = (s_2-smin[2])/(smax[2]-smin[2])
        sn_3 = (s_3-smin[3])/(smax[3]-smin[3])

        # q_k : index of the interval "containing" s_k
        q_0 = max( min( <int>(sn_0 *(order_0-1)), (order_0-1) ), 0 )
        q_1 = max( min( <int>(sn_1 *(order_1-1)), (order_1-1) ), 0 )
        q_2 = max( min( <int>(sn_2 *(order_2-1)), (order_2-1) ), 0 )
        q_3 = max( min( <int>(sn_3 *(order_3-1)), (order_3-1) ), 0 )

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


