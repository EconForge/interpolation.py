from __future__ import division

import numpy as np
from cython import double, float

ctypedef fused floating:
    float 
    double



from splines_filter import filter_data

class USpline:

    def __init__(self, smin, smax, orders, data, dtype=np.double):
        smin = np.array(smin,dtype=dtype)
        smax = np.array(smax,dtype=dtype)
        orders = np.array(orders,dtype=np.int)
        self.smin = smin
        self.smax = smax
        self.orders = orders
        self.d = len(smin)
        self.delta = (smax-smin)/(orders-1)
        self.delta_inv = 1.0/self.delta
        data = np.ascontiguousarray(data, dtype=np.double)   # filter should accept floats
        coefs = filter_data( np.array(self.delta_inv, dtype=np.double), data)
        self.coefs = np.ascontiguousarray( coefs, dtype=dtype )


class MUSpline:

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
        self.n_m = data.shape[0]
        coefs = np.concatenate( [filter_data(self.delta_inv, data[i,...]) for i in range(self.n_m) ] )
        self.coefs = coefs.reshape( (self.n_m, -1) )



A44d = np.array([
   [-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0],
   [ 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0],
   [-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0],
   [ 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0]
])

dA44d = np.array([
   [ 0.0, -0.5,  1.0, -0.5],
   [ 0.0,  1.5, -2.0,  0.0],
   [ 0.0, -1.5,  1.0,  0.5],
   [ 0.0,  0.5,  0.0,  0.0]
])

d2A44d = np.array([
   [ 0.0, 0.0, -1.0,  1.0],
   [ 0.0, 0.0,  3.0, -2.0],
   [ 0.0, 0.0, -3.0,  1.0],
   [ 0.0, 0.0,  1.0,  0.0]
])


import cython
from libc.math cimport floor
from cython.parallel import parallel, prange
from cython import nogil


def eval_MUBspline( smin, smax, orders, coefs, svec, diff=False):

    resp = np.empty( (coefs.shape[0], svec.shape[1] ),  )
    if diff:
        dresp = np.empty( (coefs.shape[0], svec.shape[0], svec.shape[1]) )

    for i in range( coefs.shape[0] ):
        if not diff:
            resp[i,:] = eval_UBspline( smin, smax, orders, coefs[i,...], svec, diff=False)
        else:
            [v,dv] = eval_UBspline( smin, smax, orders, coefs[i,...], svec, diff=True)
            resp[i,:] = v
            dresp[i,:,:] = dv

    if diff:
        return resp
    else:
        return [resp, dresp]



def eval_UBspline( smin, smax, orders, coefs, svec, diff=False):

    order = coefs.ndim
    # check that coefs and svec have consistent dimensions 

    if order not in range(1,4+1):
        raise Exception('Evaluation of {}-d splines not implemented')

    elif order == 1:
        if not diff:
            if smin.dtype == np.float64:
                return eval_UBspline_1[double]( smin, smax, orders, coefs, svec )
            elif smin.dtype == np.float32:
                return eval_UBspline_1[float]( smin, smax, orders, coefs, svec )
            else:
                raise Exception('Unsupported type')
        else:
            return eval_UBspline_1_g( smin, smax, orders, coefs, svec )
    elif order == 2:
        if not diff:
            if smin.dtype == np.float64:
                return eval_UBspline_2[double]( smin, smax, orders, coefs, svec )
            elif smin.dtype == np.float32:
                return eval_UBspline_2[float]( smin, smax, orders, coefs, svec )
            else:
                raise Exception('Unsupported type')
        else:
            return eval_UBspline_2_g( smin, smax, orders, coefs, svec )
    elif order == 3:
        if not diff:
            if smin.dtype == np.float64:
                return eval_UBspline_3[double]( smin, smax, orders, coefs, svec )
            elif smin.dtype == np.float32:
                return eval_UBspline_3[float]( smin, smax, orders, coefs, svec )
            else:
                raise Exception('Unsupported type')
        else:
            return eval_UBspline_3_g( smin, smax, orders, coefs, svec )
    elif order == 4:
        if not diff:
            if smin.dtype == np.float64:
                return eval_UBspline_4[double]( smin, smax, orders, coefs, svec )
            elif smin.dtype == np.float32:
                return eval_UBspline_4[float]( smin, smax, orders, coefs, svec )
            else:
                raise Exception('Unsupported type')
        else:
            return eval_UBspline_4_g( smin, smax, orders, coefs, svec )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_1( floating[:] smin, floating[:] smax, long[:] orders,  floating[::1] coefs, floating[:,::1] svec):
        
        
    cdef int M0 = orders[0]
    cdef floating start0 = smin[0]
    cdef floating dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
                    
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32
        
    cdef floating val

    cdef int N = svec.shape[1]

    cdef int n

    cdef floating[:,::1] Ad = np.array(A44d, dtype=dtype)
    cdef floating[:,::1] dAd = np.array(dA44d, dtype=dtype)

    cdef int i0
    cdef floating x0
    cdef floating u0
    cdef floating t0
    cdef floating extrap0
    
    cdef floating Phi0_0, Phi0_1, Phi0_2, Phi0_3
    cdef floating tp0_0, tp0_1, tp0_2, tp0_3

    cdef floating [::1] C = coefs
    cdef floating [:] vals = np.zeros(N, dtype=dtype)

    cdef floating tpx_0, tpx_1, tpx_2, tpx_3
    cdef floating tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            i0 = max( min(i0,M0-2), 0 )
            t0 = u0-i0


            # 
            # extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            # 

            tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;

            if t0 < 0:
                Phi0_0 = dAd[0,3]*t0 + Ad[0,3]
                Phi0_1 = dAd[1,3]*t0 + Ad[1,3]
                Phi0_2 = dAd[2,3]*t0 + Ad[2,3]
                Phi0_3 = dAd[3,3]*t0 + Ad[3,3]
            elif t0 > 1:
                Phi0_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi0_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi0_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi0_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
                Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
                Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
                Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

            vals[n] = Phi0_0*(C[i0+0]) + Phi0_1*(C[i0+1]) + Phi0_2*(C[i0+2]) + Phi0_3*(C[i0+3])

    return vals


@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_2( floating[:] smin, floating[:] smax, long[:] orders,  floating[:,::1] coefs, floating[:,::1] svec):
        
        
    cdef int M0 = orders[0]
    cdef floating start0 = smin[0]
    cdef floating dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef int M1 = orders[1]
    cdef floating start1 = smin[1]
    cdef floating dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])
                    
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32
        
    cdef floating val

    cdef int N = svec.shape[1]

    cdef int n

    cdef floating[:,::1] Ad = np.array(A44d, dtype=dtype)
    cdef floating[:,::1] dAd = np.array(dA44d, dtype=dtype)

    cdef int i0, i1
    cdef floating x0, x1
    cdef floating u0, u1
    cdef floating t0, t1
    cdef floating extrap0, extrap1
    
    cdef floating Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3
    cdef floating tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3

    cdef floating [:,::1] C = coefs
    cdef floating [:] vals = np.zeros(N, dtype=dtype)

    cdef floating tpx_0, tpx_1, tpx_2, tpx_3
    cdef floating tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            x1 = svec[1,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            i0 = max( min(i0,M0-2), 0 )
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            i1 = max( min(i1,M1-2), 0 )
            t1 = u1-i1


            # 
            # extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            # 
            # extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            # 

            tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;

            if t0 < 0:
                Phi0_0 = dAd[0,3]*t0 + Ad[0,3]
                Phi0_1 = dAd[1,3]*t0 + Ad[1,3]
                Phi0_2 = dAd[2,3]*t0 + Ad[2,3]
                Phi0_3 = dAd[3,3]*t0 + Ad[3,3]
            elif t0 > 1:
                Phi0_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi0_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi0_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi0_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
                Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
                Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
                Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

            if t1 < 0:
                Phi1_0 = dAd[0,3]*t1 + Ad[0,3]
                Phi1_1 = dAd[1,3]*t1 + Ad[1,3]
                Phi1_2 = dAd[2,3]*t1 + Ad[2,3]
                Phi1_3 = dAd[3,3]*t1 + Ad[3,3]
            elif t1 > 1:
                Phi1_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t1-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi1_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t1-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi1_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t1-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi1_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t1-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
                Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
                Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
                Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)

            vals[n] = Phi0_0*(Phi1_0*(C[i0+0,i1+0]) + Phi1_1*(C[i0+0,i1+1]) + Phi1_2*(C[i0+0,i1+2]) + Phi1_3*(C[i0+0,i1+3])) + Phi0_1*(Phi1_0*(C[i0+1,i1+0]) + Phi1_1*(C[i0+1,i1+1]) + Phi1_2*(C[i0+1,i1+2]) + Phi1_3*(C[i0+1,i1+3])) + Phi0_2*(Phi1_0*(C[i0+2,i1+0]) + Phi1_1*(C[i0+2,i1+1]) + Phi1_2*(C[i0+2,i1+2]) + Phi1_3*(C[i0+2,i1+3])) + Phi0_3*(Phi1_0*(C[i0+3,i1+0]) + Phi1_1*(C[i0+3,i1+1]) + Phi1_2*(C[i0+3,i1+2]) + Phi1_3*(C[i0+3,i1+3]))

    return vals


@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_3( floating[:] smin, floating[:] smax, long[:] orders,  floating[:,:,::1] coefs, floating[:,::1] svec):
        
        
    cdef int M0 = orders[0]
    cdef floating start0 = smin[0]
    cdef floating dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef int M1 = orders[1]
    cdef floating start1 = smin[1]
    cdef floating dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])
    cdef int M2 = orders[2]
    cdef floating start2 = smin[2]
    cdef floating dinv2 = (orders[2]-1.0)/(smax[2]-smin[2])
                    
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32
        
    cdef floating val

    cdef int N = svec.shape[1]

    cdef int n

    cdef floating[:,::1] Ad = np.array(A44d, dtype=dtype)
    cdef floating[:,::1] dAd = np.array(dA44d, dtype=dtype)

    cdef int i0, i1, i2
    cdef floating x0, x1, x2
    cdef floating u0, u1, u2
    cdef floating t0, t1, t2
    cdef floating extrap0, extrap1, extrap2
    
    cdef floating Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3, Phi2_0, Phi2_1, Phi2_2, Phi2_3
    cdef floating tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3, tp2_0, tp2_1, tp2_2, tp2_3

    cdef floating [:,:,::1] C = coefs
    cdef floating [:] vals = np.zeros(N, dtype=dtype)

    cdef floating tpx_0, tpx_1, tpx_2, tpx_3
    cdef floating tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            x1 = svec[1,n]
            x2 = svec[2,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            i0 = max( min(i0,M0-2), 0 )
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            i1 = max( min(i1,M1-2), 0 )
            t1 = u1-i1
            u2 = (x2 - start2)*dinv2
            i2 = <int> u2
            i2 = max( min(i2,M2-2), 0 )
            t2 = u2-i2


            # 
            # extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            # 
            # extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            # 
            # extrap2 = 0 if (t2 < 0 or t2 >= 1) else 1
            # 

            tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
            tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;

            if t0 < 0:
                Phi0_0 = dAd[0,3]*t0 + Ad[0,3]
                Phi0_1 = dAd[1,3]*t0 + Ad[1,3]
                Phi0_2 = dAd[2,3]*t0 + Ad[2,3]
                Phi0_3 = dAd[3,3]*t0 + Ad[3,3]
            elif t0 > 1:
                Phi0_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi0_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi0_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi0_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
                Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
                Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
                Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

            if t1 < 0:
                Phi1_0 = dAd[0,3]*t1 + Ad[0,3]
                Phi1_1 = dAd[1,3]*t1 + Ad[1,3]
                Phi1_2 = dAd[2,3]*t1 + Ad[2,3]
                Phi1_3 = dAd[3,3]*t1 + Ad[3,3]
            elif t1 > 1:
                Phi1_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t1-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi1_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t1-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi1_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t1-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi1_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t1-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
                Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
                Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
                Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)

            if t2 < 0:
                Phi2_0 = dAd[0,3]*t2 + Ad[0,3]
                Phi2_1 = dAd[1,3]*t2 + Ad[1,3]
                Phi2_2 = dAd[2,3]*t2 + Ad[2,3]
                Phi2_3 = dAd[3,3]*t2 + Ad[3,3]
            elif t2 > 1:
                Phi2_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t2-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi2_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t2-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi2_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t2-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi2_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t2-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi2_0 = (Ad[0,0]*tp2_0 + Ad[0,1]*tp2_1 + Ad[0,2]*tp2_2 + Ad[0,3]*tp2_3)
                Phi2_1 = (Ad[1,0]*tp2_0 + Ad[1,1]*tp2_1 + Ad[1,2]*tp2_2 + Ad[1,3]*tp2_3)
                Phi2_2 = (Ad[2,0]*tp2_0 + Ad[2,1]*tp2_1 + Ad[2,2]*tp2_2 + Ad[2,3]*tp2_3)
                Phi2_3 = (Ad[3,0]*tp2_0 + Ad[3,1]*tp2_1 + Ad[3,2]*tp2_2 + Ad[3,3]*tp2_3)

            vals[n] = Phi0_0*(Phi1_0*(Phi2_0*(C[i0+0,i1+0,i2+0]) + Phi2_1*(C[i0+0,i1+0,i2+1]) + Phi2_2*(C[i0+0,i1+0,i2+2]) + Phi2_3*(C[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+0,i1+1,i2+0]) + Phi2_1*(C[i0+0,i1+1,i2+1]) + Phi2_2*(C[i0+0,i1+1,i2+2]) + Phi2_3*(C[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+0,i1+2,i2+0]) + Phi2_1*(C[i0+0,i1+2,i2+1]) + Phi2_2*(C[i0+0,i1+2,i2+2]) + Phi2_3*(C[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+0,i1+3,i2+0]) + Phi2_1*(C[i0+0,i1+3,i2+1]) + Phi2_2*(C[i0+0,i1+3,i2+2]) + Phi2_3*(C[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(C[i0+1,i1+0,i2+0]) + Phi2_1*(C[i0+1,i1+0,i2+1]) + Phi2_2*(C[i0+1,i1+0,i2+2]) + Phi2_3*(C[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+1,i1+1,i2+0]) + Phi2_1*(C[i0+1,i1+1,i2+1]) + Phi2_2*(C[i0+1,i1+1,i2+2]) + Phi2_3*(C[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+1,i1+2,i2+0]) + Phi2_1*(C[i0+1,i1+2,i2+1]) + Phi2_2*(C[i0+1,i1+2,i2+2]) + Phi2_3*(C[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+1,i1+3,i2+0]) + Phi2_1*(C[i0+1,i1+3,i2+1]) + Phi2_2*(C[i0+1,i1+3,i2+2]) + Phi2_3*(C[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(C[i0+2,i1+0,i2+0]) + Phi2_1*(C[i0+2,i1+0,i2+1]) + Phi2_2*(C[i0+2,i1+0,i2+2]) + Phi2_3*(C[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+2,i1+1,i2+0]) + Phi2_1*(C[i0+2,i1+1,i2+1]) + Phi2_2*(C[i0+2,i1+1,i2+2]) + Phi2_3*(C[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+2,i1+2,i2+0]) + Phi2_1*(C[i0+2,i1+2,i2+1]) + Phi2_2*(C[i0+2,i1+2,i2+2]) + Phi2_3*(C[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+2,i1+3,i2+0]) + Phi2_1*(C[i0+2,i1+3,i2+1]) + Phi2_2*(C[i0+2,i1+3,i2+2]) + Phi2_3*(C[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(C[i0+3,i1+0,i2+0]) + Phi2_1*(C[i0+3,i1+0,i2+1]) + Phi2_2*(C[i0+3,i1+0,i2+2]) + Phi2_3*(C[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+3,i1+1,i2+0]) + Phi2_1*(C[i0+3,i1+1,i2+1]) + Phi2_2*(C[i0+3,i1+1,i2+2]) + Phi2_3*(C[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+3,i1+2,i2+0]) + Phi2_1*(C[i0+3,i1+2,i2+1]) + Phi2_2*(C[i0+3,i1+2,i2+2]) + Phi2_3*(C[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+3,i1+3,i2+0]) + Phi2_1*(C[i0+3,i1+3,i2+1]) + Phi2_2*(C[i0+3,i1+3,i2+2]) + Phi2_3*(C[i0+3,i1+3,i2+3])))

    return vals


@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_4( floating[:] smin, floating[:] smax, long[:] orders,  floating[:,:,:,::1] coefs, floating[:,::1] svec):
        
        
    cdef int M0 = orders[0]
    cdef floating start0 = smin[0]
    cdef floating dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef int M1 = orders[1]
    cdef floating start1 = smin[1]
    cdef floating dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])
    cdef int M2 = orders[2]
    cdef floating start2 = smin[2]
    cdef floating dinv2 = (orders[2]-1.0)/(smax[2]-smin[2])
    cdef int M3 = orders[3]
    cdef floating start3 = smin[3]
    cdef floating dinv3 = (orders[3]-1.0)/(smax[3]-smin[3])
                    
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32
        
    cdef floating val

    cdef int N = svec.shape[1]

    cdef int n

    cdef floating[:,::1] Ad = np.array(A44d, dtype=dtype)
    cdef floating[:,::1] dAd = np.array(dA44d, dtype=dtype)

    cdef int i0, i1, i2, i3
    cdef floating x0, x1, x2, x3
    cdef floating u0, u1, u2, u3
    cdef floating t0, t1, t2, t3
    cdef floating extrap0, extrap1, extrap2, extrap3
    
    cdef floating Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3, Phi2_0, Phi2_1, Phi2_2, Phi2_3, Phi3_0, Phi3_1, Phi3_2, Phi3_3
    cdef floating tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3, tp2_0, tp2_1, tp2_2, tp2_3, tp3_0, tp3_1, tp3_2, tp3_3

    cdef floating [:,:,:,::1] C = coefs
    cdef floating [:] vals = np.zeros(N, dtype=dtype)

    cdef floating tpx_0, tpx_1, tpx_2, tpx_3
    cdef floating tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            x1 = svec[1,n]
            x2 = svec[2,n]
            x3 = svec[3,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            i0 = max( min(i0,M0-2), 0 )
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            i1 = max( min(i1,M1-2), 0 )
            t1 = u1-i1
            u2 = (x2 - start2)*dinv2
            i2 = <int> u2
            i2 = max( min(i2,M2-2), 0 )
            t2 = u2-i2
            u3 = (x3 - start3)*dinv3
            i3 = <int> u3
            i3 = max( min(i3,M3-2), 0 )
            t3 = u3-i3


            # 
            # extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            # 
            # extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            # 
            # extrap2 = 0 if (t2 < 0 or t2 >= 1) else 1
            # 
            # extrap3 = 0 if (t3 < 0 or t3 >= 1) else 1
            # 

            tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
            tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
            tp3_0 = t3*t3*t3;  tp3_1 = t3*t3;  tp3_2 = t3;  tp3_3 = 1.0;

            if t0 < 0:
                Phi0_0 = dAd[0,3]*t0 + Ad[0,3]
                Phi0_1 = dAd[1,3]*t0 + Ad[1,3]
                Phi0_2 = dAd[2,3]*t0 + Ad[2,3]
                Phi0_3 = dAd[3,3]*t0 + Ad[3,3]
            elif t0 > 1:
                Phi0_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi0_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi0_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi0_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
                Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
                Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
                Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

            if t1 < 0:
                Phi1_0 = dAd[0,3]*t1 + Ad[0,3]
                Phi1_1 = dAd[1,3]*t1 + Ad[1,3]
                Phi1_2 = dAd[2,3]*t1 + Ad[2,3]
                Phi1_3 = dAd[3,3]*t1 + Ad[3,3]
            elif t1 > 1:
                Phi1_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t1-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi1_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t1-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi1_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t1-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi1_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t1-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
                Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
                Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
                Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)

            if t2 < 0:
                Phi2_0 = dAd[0,3]*t2 + Ad[0,3]
                Phi2_1 = dAd[1,3]*t2 + Ad[1,3]
                Phi2_2 = dAd[2,3]*t2 + Ad[2,3]
                Phi2_3 = dAd[3,3]*t2 + Ad[3,3]
            elif t2 > 1:
                Phi2_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t2-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi2_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t2-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi2_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t2-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi2_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t2-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi2_0 = (Ad[0,0]*tp2_0 + Ad[0,1]*tp2_1 + Ad[0,2]*tp2_2 + Ad[0,3]*tp2_3)
                Phi2_1 = (Ad[1,0]*tp2_0 + Ad[1,1]*tp2_1 + Ad[1,2]*tp2_2 + Ad[1,3]*tp2_3)
                Phi2_2 = (Ad[2,0]*tp2_0 + Ad[2,1]*tp2_1 + Ad[2,2]*tp2_2 + Ad[2,3]*tp2_3)
                Phi2_3 = (Ad[3,0]*tp2_0 + Ad[3,1]*tp2_1 + Ad[3,2]*tp2_2 + Ad[3,3]*tp2_3)

            if t3 < 0:
                Phi3_0 = dAd[0,3]*t3 + Ad[0,3]
                Phi3_1 = dAd[1,3]*t3 + Ad[1,3]
                Phi3_2 = dAd[2,3]*t3 + Ad[2,3]
                Phi3_3 = dAd[3,3]*t3 + Ad[3,3]
            elif t3 > 1:
                Phi3_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t3-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi3_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t3-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi3_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t3-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi3_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t3-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi3_0 = (Ad[0,0]*tp3_0 + Ad[0,1]*tp3_1 + Ad[0,2]*tp3_2 + Ad[0,3]*tp3_3)
                Phi3_1 = (Ad[1,0]*tp3_0 + Ad[1,1]*tp3_1 + Ad[1,2]*tp3_2 + Ad[1,3]*tp3_3)
                Phi3_2 = (Ad[2,0]*tp3_0 + Ad[2,1]*tp3_1 + Ad[2,2]*tp3_2 + Ad[2,3]*tp3_3)
                Phi3_3 = (Ad[3,0]*tp3_0 + Ad[3,1]*tp3_1 + Ad[3,2]*tp3_2 + Ad[3,3]*tp3_3)

            vals[n] = Phi0_0*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+0,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+0,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+0,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+0,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+3,i3+3])))) + Phi0_1*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+1,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+1,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+1,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+1,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+3,i3+3])))) + Phi0_2*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+2,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+2,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+2,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+2,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+3,i3+3])))) + Phi0_3*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+3,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+3,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+3,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+3,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+3,i3+3]))))

    return vals

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_1_g( smin, smax, orders, coefs, double[:,::1] svec):
        
        
    cdef double start0 = smin[0]
    cdef double dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
                    

    cdef double val

    cdef int N = svec.shape[1]

    cdef int n

    cdef double[:,::1] Ad = A44d
    cdef double[:,::1] dAd = dA44d

    cdef int i0
    cdef double x0
    cdef double u0
    cdef double t0
    cdef double extrap0
    
    cdef double Phi0_0, Phi0_1, Phi0_2, Phi0_3
    cdef double dPhi0_0, dPhi0_1, dPhi0_2, dPhi0_3
    cdef double tp0_0, tp0_1, tp0_2, tp0_3

    cdef double[::1] C = coefs

    cdef double[:] vals = np.zeros(N)
    cdef double[:,::1] dvals = np.zeros((1,N))

    cdef double tpx_0, tpx_1, tpx_2, tpx_3
    cdef double tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            t0 = u0-i0
            extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            tp0_0 = t0*t0*t0*extrap0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
            Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
            Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
            Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)
            dPhi0_0 = (dAd[0,0]*tp0_0 + dAd[0,1]*tp0_1 + dAd[0,2]*tp0_2 + dAd[0,3]*tp0_3)*dinv0
            dPhi0_1 = (dAd[1,0]*tp0_0 + dAd[1,1]*tp0_1 + dAd[1,2]*tp0_2 + dAd[1,3]*tp0_3)*dinv0
            dPhi0_2 = (dAd[2,0]*tp0_0 + dAd[2,1]*tp0_1 + dAd[2,2]*tp0_2 + dAd[2,3]*tp0_3)*dinv0
            dPhi0_3 = (dAd[3,0]*tp0_0 + dAd[3,1]*tp0_1 + dAd[3,2]*tp0_2 + dAd[3,3]*tp0_3)*dinv0


            vals[n] = Phi0_0*(C[i0+0]) + Phi0_1*(C[i0+1]) + Phi0_2*(C[i0+2]) + Phi0_3*(C[i0+3])

            dvals[0,n] = dPhi0_0*(C[i0+0]) + dPhi0_1*(C[i0+1]) + dPhi0_2*(C[i0+2]) + dPhi0_3*(C[i0+3]) 

    return [vals,dvals]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_2_g( smin, smax, orders, coefs, double[:,::1] svec):
        
        
    cdef double start0 = smin[0]
    cdef double dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef double start1 = smin[1]
    cdef double dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])
                    

    cdef double val

    cdef int N = svec.shape[1]

    cdef int n

    cdef double[:,::1] Ad = A44d
    cdef double[:,::1] dAd = dA44d

    cdef int i0, i1
    cdef double x0, x1
    cdef double u0, u1
    cdef double t0, t1
    cdef double extrap0, extrap1
    
    cdef double Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3
    cdef double dPhi0_0, dPhi0_1, dPhi0_2, dPhi0_3, dPhi1_0, dPhi1_1, dPhi1_2, dPhi1_3
    cdef double tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3

    cdef double[:,::1] C = coefs

    cdef double[:] vals = np.zeros(N)
    cdef double[:,::1] dvals = np.zeros((2,N))

    cdef double tpx_0, tpx_1, tpx_2, tpx_3
    cdef double tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            x1 = svec[1,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            t1 = u1-i1
            extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            tp0_0 = t0*t0*t0*extrap0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1*extrap1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
            Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
            Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
            Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
            Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)
            Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
            Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
            Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
            Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)
            dPhi0_0 = (dAd[0,0]*tp0_0 + dAd[0,1]*tp0_1 + dAd[0,2]*tp0_2 + dAd[0,3]*tp0_3)*dinv0
            dPhi0_1 = (dAd[1,0]*tp0_0 + dAd[1,1]*tp0_1 + dAd[1,2]*tp0_2 + dAd[1,3]*tp0_3)*dinv0
            dPhi0_2 = (dAd[2,0]*tp0_0 + dAd[2,1]*tp0_1 + dAd[2,2]*tp0_2 + dAd[2,3]*tp0_3)*dinv0
            dPhi0_3 = (dAd[3,0]*tp0_0 + dAd[3,1]*tp0_1 + dAd[3,2]*tp0_2 + dAd[3,3]*tp0_3)*dinv0
            dPhi1_0 = (dAd[0,0]*tp1_0 + dAd[0,1]*tp1_1 + dAd[0,2]*tp1_2 + dAd[0,3]*tp1_3)*dinv1
            dPhi1_1 = (dAd[1,0]*tp1_0 + dAd[1,1]*tp1_1 + dAd[1,2]*tp1_2 + dAd[1,3]*tp1_3)*dinv1
            dPhi1_2 = (dAd[2,0]*tp1_0 + dAd[2,1]*tp1_1 + dAd[2,2]*tp1_2 + dAd[2,3]*tp1_3)*dinv1
            dPhi1_3 = (dAd[3,0]*tp1_0 + dAd[3,1]*tp1_1 + dAd[3,2]*tp1_2 + dAd[3,3]*tp1_3)*dinv1


            vals[n] = Phi0_0*(Phi1_0*(C[i0+0,i1+0]) + Phi1_1*(C[i0+0,i1+1]) + Phi1_2*(C[i0+0,i1+2]) + Phi1_3*(C[i0+0,i1+3])) + Phi0_1*(Phi1_0*(C[i0+1,i1+0]) + Phi1_1*(C[i0+1,i1+1]) + Phi1_2*(C[i0+1,i1+2]) + Phi1_3*(C[i0+1,i1+3])) + Phi0_2*(Phi1_0*(C[i0+2,i1+0]) + Phi1_1*(C[i0+2,i1+1]) + Phi1_2*(C[i0+2,i1+2]) + Phi1_3*(C[i0+2,i1+3])) + Phi0_3*(Phi1_0*(C[i0+3,i1+0]) + Phi1_1*(C[i0+3,i1+1]) + Phi1_2*(C[i0+3,i1+2]) + Phi1_3*(C[i0+3,i1+3]))

            dvals[0,n] = dPhi0_0*(Phi1_0*(C[i0+0,i1+0]) + Phi1_1*(C[i0+0,i1+1]) + Phi1_2*(C[i0+0,i1+2]) + Phi1_3*(C[i0+0,i1+3])) + dPhi0_1*(Phi1_0*(C[i0+1,i1+0]) + Phi1_1*(C[i0+1,i1+1]) + Phi1_2*(C[i0+1,i1+2]) + Phi1_3*(C[i0+1,i1+3])) + dPhi0_2*(Phi1_0*(C[i0+2,i1+0]) + Phi1_1*(C[i0+2,i1+1]) + Phi1_2*(C[i0+2,i1+2]) + Phi1_3*(C[i0+2,i1+3])) + dPhi0_3*(Phi1_0*(C[i0+3,i1+0]) + Phi1_1*(C[i0+3,i1+1]) + Phi1_2*(C[i0+3,i1+2]) + Phi1_3*(C[i0+3,i1+3])) 
            dvals[1,n] = Phi0_0*(dPhi1_0*(C[i0+0,i1+0]) + dPhi1_1*(C[i0+0,i1+1]) + dPhi1_2*(C[i0+0,i1+2]) + dPhi1_3*(C[i0+0,i1+3])) + Phi0_1*(dPhi1_0*(C[i0+1,i1+0]) + dPhi1_1*(C[i0+1,i1+1]) + dPhi1_2*(C[i0+1,i1+2]) + dPhi1_3*(C[i0+1,i1+3])) + Phi0_2*(dPhi1_0*(C[i0+2,i1+0]) + dPhi1_1*(C[i0+2,i1+1]) + dPhi1_2*(C[i0+2,i1+2]) + dPhi1_3*(C[i0+2,i1+3])) + Phi0_3*(dPhi1_0*(C[i0+3,i1+0]) + dPhi1_1*(C[i0+3,i1+1]) + dPhi1_2*(C[i0+3,i1+2]) + dPhi1_3*(C[i0+3,i1+3])) 

    return [vals,dvals]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_3_g( smin, smax, orders, coefs, double[:,::1] svec):
        
        
    cdef double start0 = smin[0]
    cdef double dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef double start1 = smin[1]
    cdef double dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])
    cdef double start2 = smin[2]
    cdef double dinv2 = (orders[2]-1.0)/(smax[2]-smin[2])
                    

    cdef double val

    cdef int N = svec.shape[1]

    cdef int n

    cdef double[:,::1] Ad = A44d
    cdef double[:,::1] dAd = dA44d

    cdef int i0, i1, i2
    cdef double x0, x1, x2
    cdef double u0, u1, u2
    cdef double t0, t1, t2
    cdef double extrap0, extrap1, extrap2
    
    cdef double Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3, Phi2_0, Phi2_1, Phi2_2, Phi2_3
    cdef double dPhi0_0, dPhi0_1, dPhi0_2, dPhi0_3, dPhi1_0, dPhi1_1, dPhi1_2, dPhi1_3, dPhi2_0, dPhi2_1, dPhi2_2, dPhi2_3
    cdef double tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3, tp2_0, tp2_1, tp2_2, tp2_3

    cdef double[:,:,::1] C = coefs

    cdef double[:] vals = np.zeros(N)
    cdef double[:,::1] dvals = np.zeros((3,N))

    cdef double tpx_0, tpx_1, tpx_2, tpx_3
    cdef double tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            x1 = svec[1,n]
            x2 = svec[2,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            t1 = u1-i1
            u2 = (x2 - start2)*dinv2
            i2 = <int> u2
            t2 = u2-i2
            extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            extrap2 = 0 if (t2 < 0 or t2 >= 1) else 1
            tp0_0 = t0*t0*t0*extrap0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1*extrap1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
            tp2_0 = t2*t2*t2*extrap2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
            Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
            Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
            Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
            Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)
            Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
            Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
            Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
            Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)
            Phi2_0 = (Ad[0,0]*tp2_0 + Ad[0,1]*tp2_1 + Ad[0,2]*tp2_2 + Ad[0,3]*tp2_3)
            Phi2_1 = (Ad[1,0]*tp2_0 + Ad[1,1]*tp2_1 + Ad[1,2]*tp2_2 + Ad[1,3]*tp2_3)
            Phi2_2 = (Ad[2,0]*tp2_0 + Ad[2,1]*tp2_1 + Ad[2,2]*tp2_2 + Ad[2,3]*tp2_3)
            Phi2_3 = (Ad[3,0]*tp2_0 + Ad[3,1]*tp2_1 + Ad[3,2]*tp2_2 + Ad[3,3]*tp2_3)
            dPhi0_0 = (dAd[0,0]*tp0_0 + dAd[0,1]*tp0_1 + dAd[0,2]*tp0_2 + dAd[0,3]*tp0_3)*dinv0
            dPhi0_1 = (dAd[1,0]*tp0_0 + dAd[1,1]*tp0_1 + dAd[1,2]*tp0_2 + dAd[1,3]*tp0_3)*dinv0
            dPhi0_2 = (dAd[2,0]*tp0_0 + dAd[2,1]*tp0_1 + dAd[2,2]*tp0_2 + dAd[2,3]*tp0_3)*dinv0
            dPhi0_3 = (dAd[3,0]*tp0_0 + dAd[3,1]*tp0_1 + dAd[3,2]*tp0_2 + dAd[3,3]*tp0_3)*dinv0
            dPhi1_0 = (dAd[0,0]*tp1_0 + dAd[0,1]*tp1_1 + dAd[0,2]*tp1_2 + dAd[0,3]*tp1_3)*dinv1
            dPhi1_1 = (dAd[1,0]*tp1_0 + dAd[1,1]*tp1_1 + dAd[1,2]*tp1_2 + dAd[1,3]*tp1_3)*dinv1
            dPhi1_2 = (dAd[2,0]*tp1_0 + dAd[2,1]*tp1_1 + dAd[2,2]*tp1_2 + dAd[2,3]*tp1_3)*dinv1
            dPhi1_3 = (dAd[3,0]*tp1_0 + dAd[3,1]*tp1_1 + dAd[3,2]*tp1_2 + dAd[3,3]*tp1_3)*dinv1
            dPhi2_0 = (dAd[0,0]*tp2_0 + dAd[0,1]*tp2_1 + dAd[0,2]*tp2_2 + dAd[0,3]*tp2_3)*dinv2
            dPhi2_1 = (dAd[1,0]*tp2_0 + dAd[1,1]*tp2_1 + dAd[1,2]*tp2_2 + dAd[1,3]*tp2_3)*dinv2
            dPhi2_2 = (dAd[2,0]*tp2_0 + dAd[2,1]*tp2_1 + dAd[2,2]*tp2_2 + dAd[2,3]*tp2_3)*dinv2
            dPhi2_3 = (dAd[3,0]*tp2_0 + dAd[3,1]*tp2_1 + dAd[3,2]*tp2_2 + dAd[3,3]*tp2_3)*dinv2


            vals[n] = Phi0_0*(Phi1_0*(Phi2_0*(C[i0+0,i1+0,i2+0]) + Phi2_1*(C[i0+0,i1+0,i2+1]) + Phi2_2*(C[i0+0,i1+0,i2+2]) + Phi2_3*(C[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+0,i1+1,i2+0]) + Phi2_1*(C[i0+0,i1+1,i2+1]) + Phi2_2*(C[i0+0,i1+1,i2+2]) + Phi2_3*(C[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+0,i1+2,i2+0]) + Phi2_1*(C[i0+0,i1+2,i2+1]) + Phi2_2*(C[i0+0,i1+2,i2+2]) + Phi2_3*(C[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+0,i1+3,i2+0]) + Phi2_1*(C[i0+0,i1+3,i2+1]) + Phi2_2*(C[i0+0,i1+3,i2+2]) + Phi2_3*(C[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(C[i0+1,i1+0,i2+0]) + Phi2_1*(C[i0+1,i1+0,i2+1]) + Phi2_2*(C[i0+1,i1+0,i2+2]) + Phi2_3*(C[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+1,i1+1,i2+0]) + Phi2_1*(C[i0+1,i1+1,i2+1]) + Phi2_2*(C[i0+1,i1+1,i2+2]) + Phi2_3*(C[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+1,i1+2,i2+0]) + Phi2_1*(C[i0+1,i1+2,i2+1]) + Phi2_2*(C[i0+1,i1+2,i2+2]) + Phi2_3*(C[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+1,i1+3,i2+0]) + Phi2_1*(C[i0+1,i1+3,i2+1]) + Phi2_2*(C[i0+1,i1+3,i2+2]) + Phi2_3*(C[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(C[i0+2,i1+0,i2+0]) + Phi2_1*(C[i0+2,i1+0,i2+1]) + Phi2_2*(C[i0+2,i1+0,i2+2]) + Phi2_3*(C[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+2,i1+1,i2+0]) + Phi2_1*(C[i0+2,i1+1,i2+1]) + Phi2_2*(C[i0+2,i1+1,i2+2]) + Phi2_3*(C[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+2,i1+2,i2+0]) + Phi2_1*(C[i0+2,i1+2,i2+1]) + Phi2_2*(C[i0+2,i1+2,i2+2]) + Phi2_3*(C[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+2,i1+3,i2+0]) + Phi2_1*(C[i0+2,i1+3,i2+1]) + Phi2_2*(C[i0+2,i1+3,i2+2]) + Phi2_3*(C[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(C[i0+3,i1+0,i2+0]) + Phi2_1*(C[i0+3,i1+0,i2+1]) + Phi2_2*(C[i0+3,i1+0,i2+2]) + Phi2_3*(C[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+3,i1+1,i2+0]) + Phi2_1*(C[i0+3,i1+1,i2+1]) + Phi2_2*(C[i0+3,i1+1,i2+2]) + Phi2_3*(C[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+3,i1+2,i2+0]) + Phi2_1*(C[i0+3,i1+2,i2+1]) + Phi2_2*(C[i0+3,i1+2,i2+2]) + Phi2_3*(C[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+3,i1+3,i2+0]) + Phi2_1*(C[i0+3,i1+3,i2+1]) + Phi2_2*(C[i0+3,i1+3,i2+2]) + Phi2_3*(C[i0+3,i1+3,i2+3])))

            dvals[0,n] = dPhi0_0*(Phi1_0*(Phi2_0*(C[i0+0,i1+0,i2+0]) + Phi2_1*(C[i0+0,i1+0,i2+1]) + Phi2_2*(C[i0+0,i1+0,i2+2]) + Phi2_3*(C[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+0,i1+1,i2+0]) + Phi2_1*(C[i0+0,i1+1,i2+1]) + Phi2_2*(C[i0+0,i1+1,i2+2]) + Phi2_3*(C[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+0,i1+2,i2+0]) + Phi2_1*(C[i0+0,i1+2,i2+1]) + Phi2_2*(C[i0+0,i1+2,i2+2]) + Phi2_3*(C[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+0,i1+3,i2+0]) + Phi2_1*(C[i0+0,i1+3,i2+1]) + Phi2_2*(C[i0+0,i1+3,i2+2]) + Phi2_3*(C[i0+0,i1+3,i2+3]))) + dPhi0_1*(Phi1_0*(Phi2_0*(C[i0+1,i1+0,i2+0]) + Phi2_1*(C[i0+1,i1+0,i2+1]) + Phi2_2*(C[i0+1,i1+0,i2+2]) + Phi2_3*(C[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+1,i1+1,i2+0]) + Phi2_1*(C[i0+1,i1+1,i2+1]) + Phi2_2*(C[i0+1,i1+1,i2+2]) + Phi2_3*(C[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+1,i1+2,i2+0]) + Phi2_1*(C[i0+1,i1+2,i2+1]) + Phi2_2*(C[i0+1,i1+2,i2+2]) + Phi2_3*(C[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+1,i1+3,i2+0]) + Phi2_1*(C[i0+1,i1+3,i2+1]) + Phi2_2*(C[i0+1,i1+3,i2+2]) + Phi2_3*(C[i0+1,i1+3,i2+3]))) + dPhi0_2*(Phi1_0*(Phi2_0*(C[i0+2,i1+0,i2+0]) + Phi2_1*(C[i0+2,i1+0,i2+1]) + Phi2_2*(C[i0+2,i1+0,i2+2]) + Phi2_3*(C[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+2,i1+1,i2+0]) + Phi2_1*(C[i0+2,i1+1,i2+1]) + Phi2_2*(C[i0+2,i1+1,i2+2]) + Phi2_3*(C[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+2,i1+2,i2+0]) + Phi2_1*(C[i0+2,i1+2,i2+1]) + Phi2_2*(C[i0+2,i1+2,i2+2]) + Phi2_3*(C[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+2,i1+3,i2+0]) + Phi2_1*(C[i0+2,i1+3,i2+1]) + Phi2_2*(C[i0+2,i1+3,i2+2]) + Phi2_3*(C[i0+2,i1+3,i2+3]))) + dPhi0_3*(Phi1_0*(Phi2_0*(C[i0+3,i1+0,i2+0]) + Phi2_1*(C[i0+3,i1+0,i2+1]) + Phi2_2*(C[i0+3,i1+0,i2+2]) + Phi2_3*(C[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(C[i0+3,i1+1,i2+0]) + Phi2_1*(C[i0+3,i1+1,i2+1]) + Phi2_2*(C[i0+3,i1+1,i2+2]) + Phi2_3*(C[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(C[i0+3,i1+2,i2+0]) + Phi2_1*(C[i0+3,i1+2,i2+1]) + Phi2_2*(C[i0+3,i1+2,i2+2]) + Phi2_3*(C[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(C[i0+3,i1+3,i2+0]) + Phi2_1*(C[i0+3,i1+3,i2+1]) + Phi2_2*(C[i0+3,i1+3,i2+2]) + Phi2_3*(C[i0+3,i1+3,i2+3]))) 
            dvals[1,n] = Phi0_0*(dPhi1_0*(Phi2_0*(C[i0+0,i1+0,i2+0]) + Phi2_1*(C[i0+0,i1+0,i2+1]) + Phi2_2*(C[i0+0,i1+0,i2+2]) + Phi2_3*(C[i0+0,i1+0,i2+3])) + dPhi1_1*(Phi2_0*(C[i0+0,i1+1,i2+0]) + Phi2_1*(C[i0+0,i1+1,i2+1]) + Phi2_2*(C[i0+0,i1+1,i2+2]) + Phi2_3*(C[i0+0,i1+1,i2+3])) + dPhi1_2*(Phi2_0*(C[i0+0,i1+2,i2+0]) + Phi2_1*(C[i0+0,i1+2,i2+1]) + Phi2_2*(C[i0+0,i1+2,i2+2]) + Phi2_3*(C[i0+0,i1+2,i2+3])) + dPhi1_3*(Phi2_0*(C[i0+0,i1+3,i2+0]) + Phi2_1*(C[i0+0,i1+3,i2+1]) + Phi2_2*(C[i0+0,i1+3,i2+2]) + Phi2_3*(C[i0+0,i1+3,i2+3]))) + Phi0_1*(dPhi1_0*(Phi2_0*(C[i0+1,i1+0,i2+0]) + Phi2_1*(C[i0+1,i1+0,i2+1]) + Phi2_2*(C[i0+1,i1+0,i2+2]) + Phi2_3*(C[i0+1,i1+0,i2+3])) + dPhi1_1*(Phi2_0*(C[i0+1,i1+1,i2+0]) + Phi2_1*(C[i0+1,i1+1,i2+1]) + Phi2_2*(C[i0+1,i1+1,i2+2]) + Phi2_3*(C[i0+1,i1+1,i2+3])) + dPhi1_2*(Phi2_0*(C[i0+1,i1+2,i2+0]) + Phi2_1*(C[i0+1,i1+2,i2+1]) + Phi2_2*(C[i0+1,i1+2,i2+2]) + Phi2_3*(C[i0+1,i1+2,i2+3])) + dPhi1_3*(Phi2_0*(C[i0+1,i1+3,i2+0]) + Phi2_1*(C[i0+1,i1+3,i2+1]) + Phi2_2*(C[i0+1,i1+3,i2+2]) + Phi2_3*(C[i0+1,i1+3,i2+3]))) + Phi0_2*(dPhi1_0*(Phi2_0*(C[i0+2,i1+0,i2+0]) + Phi2_1*(C[i0+2,i1+0,i2+1]) + Phi2_2*(C[i0+2,i1+0,i2+2]) + Phi2_3*(C[i0+2,i1+0,i2+3])) + dPhi1_1*(Phi2_0*(C[i0+2,i1+1,i2+0]) + Phi2_1*(C[i0+2,i1+1,i2+1]) + Phi2_2*(C[i0+2,i1+1,i2+2]) + Phi2_3*(C[i0+2,i1+1,i2+3])) + dPhi1_2*(Phi2_0*(C[i0+2,i1+2,i2+0]) + Phi2_1*(C[i0+2,i1+2,i2+1]) + Phi2_2*(C[i0+2,i1+2,i2+2]) + Phi2_3*(C[i0+2,i1+2,i2+3])) + dPhi1_3*(Phi2_0*(C[i0+2,i1+3,i2+0]) + Phi2_1*(C[i0+2,i1+3,i2+1]) + Phi2_2*(C[i0+2,i1+3,i2+2]) + Phi2_3*(C[i0+2,i1+3,i2+3]))) + Phi0_3*(dPhi1_0*(Phi2_0*(C[i0+3,i1+0,i2+0]) + Phi2_1*(C[i0+3,i1+0,i2+1]) + Phi2_2*(C[i0+3,i1+0,i2+2]) + Phi2_3*(C[i0+3,i1+0,i2+3])) + dPhi1_1*(Phi2_0*(C[i0+3,i1+1,i2+0]) + Phi2_1*(C[i0+3,i1+1,i2+1]) + Phi2_2*(C[i0+3,i1+1,i2+2]) + Phi2_3*(C[i0+3,i1+1,i2+3])) + dPhi1_2*(Phi2_0*(C[i0+3,i1+2,i2+0]) + Phi2_1*(C[i0+3,i1+2,i2+1]) + Phi2_2*(C[i0+3,i1+2,i2+2]) + Phi2_3*(C[i0+3,i1+2,i2+3])) + dPhi1_3*(Phi2_0*(C[i0+3,i1+3,i2+0]) + Phi2_1*(C[i0+3,i1+3,i2+1]) + Phi2_2*(C[i0+3,i1+3,i2+2]) + Phi2_3*(C[i0+3,i1+3,i2+3]))) 
            dvals[2,n] = Phi0_0*(Phi1_0*(dPhi2_0*(C[i0+0,i1+0,i2+0]) + dPhi2_1*(C[i0+0,i1+0,i2+1]) + dPhi2_2*(C[i0+0,i1+0,i2+2]) + dPhi2_3*(C[i0+0,i1+0,i2+3])) + Phi1_1*(dPhi2_0*(C[i0+0,i1+1,i2+0]) + dPhi2_1*(C[i0+0,i1+1,i2+1]) + dPhi2_2*(C[i0+0,i1+1,i2+2]) + dPhi2_3*(C[i0+0,i1+1,i2+3])) + Phi1_2*(dPhi2_0*(C[i0+0,i1+2,i2+0]) + dPhi2_1*(C[i0+0,i1+2,i2+1]) + dPhi2_2*(C[i0+0,i1+2,i2+2]) + dPhi2_3*(C[i0+0,i1+2,i2+3])) + Phi1_3*(dPhi2_0*(C[i0+0,i1+3,i2+0]) + dPhi2_1*(C[i0+0,i1+3,i2+1]) + dPhi2_2*(C[i0+0,i1+3,i2+2]) + dPhi2_3*(C[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(dPhi2_0*(C[i0+1,i1+0,i2+0]) + dPhi2_1*(C[i0+1,i1+0,i2+1]) + dPhi2_2*(C[i0+1,i1+0,i2+2]) + dPhi2_3*(C[i0+1,i1+0,i2+3])) + Phi1_1*(dPhi2_0*(C[i0+1,i1+1,i2+0]) + dPhi2_1*(C[i0+1,i1+1,i2+1]) + dPhi2_2*(C[i0+1,i1+1,i2+2]) + dPhi2_3*(C[i0+1,i1+1,i2+3])) + Phi1_2*(dPhi2_0*(C[i0+1,i1+2,i2+0]) + dPhi2_1*(C[i0+1,i1+2,i2+1]) + dPhi2_2*(C[i0+1,i1+2,i2+2]) + dPhi2_3*(C[i0+1,i1+2,i2+3])) + Phi1_3*(dPhi2_0*(C[i0+1,i1+3,i2+0]) + dPhi2_1*(C[i0+1,i1+3,i2+1]) + dPhi2_2*(C[i0+1,i1+3,i2+2]) + dPhi2_3*(C[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(dPhi2_0*(C[i0+2,i1+0,i2+0]) + dPhi2_1*(C[i0+2,i1+0,i2+1]) + dPhi2_2*(C[i0+2,i1+0,i2+2]) + dPhi2_3*(C[i0+2,i1+0,i2+3])) + Phi1_1*(dPhi2_0*(C[i0+2,i1+1,i2+0]) + dPhi2_1*(C[i0+2,i1+1,i2+1]) + dPhi2_2*(C[i0+2,i1+1,i2+2]) + dPhi2_3*(C[i0+2,i1+1,i2+3])) + Phi1_2*(dPhi2_0*(C[i0+2,i1+2,i2+0]) + dPhi2_1*(C[i0+2,i1+2,i2+1]) + dPhi2_2*(C[i0+2,i1+2,i2+2]) + dPhi2_3*(C[i0+2,i1+2,i2+3])) + Phi1_3*(dPhi2_0*(C[i0+2,i1+3,i2+0]) + dPhi2_1*(C[i0+2,i1+3,i2+1]) + dPhi2_2*(C[i0+2,i1+3,i2+2]) + dPhi2_3*(C[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(dPhi2_0*(C[i0+3,i1+0,i2+0]) + dPhi2_1*(C[i0+3,i1+0,i2+1]) + dPhi2_2*(C[i0+3,i1+0,i2+2]) + dPhi2_3*(C[i0+3,i1+0,i2+3])) + Phi1_1*(dPhi2_0*(C[i0+3,i1+1,i2+0]) + dPhi2_1*(C[i0+3,i1+1,i2+1]) + dPhi2_2*(C[i0+3,i1+1,i2+2]) + dPhi2_3*(C[i0+3,i1+1,i2+3])) + Phi1_2*(dPhi2_0*(C[i0+3,i1+2,i2+0]) + dPhi2_1*(C[i0+3,i1+2,i2+1]) + dPhi2_2*(C[i0+3,i1+2,i2+2]) + dPhi2_3*(C[i0+3,i1+2,i2+3])) + Phi1_3*(dPhi2_0*(C[i0+3,i1+3,i2+0]) + dPhi2_1*(C[i0+3,i1+3,i2+1]) + dPhi2_2*(C[i0+3,i1+3,i2+2]) + dPhi2_3*(C[i0+3,i1+3,i2+3]))) 

    return [vals,dvals]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_UBspline_4_g( smin, smax, orders, coefs, double[:,::1] svec):
        
        
    cdef double start0 = smin[0]
    cdef double dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef double start1 = smin[1]
    cdef double dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])
    cdef double start2 = smin[2]
    cdef double dinv2 = (orders[2]-1.0)/(smax[2]-smin[2])
    cdef double start3 = smin[3]
    cdef double dinv3 = (orders[3]-1.0)/(smax[3]-smin[3])
                    

    cdef double val

    cdef int N = svec.shape[1]

    cdef int n

    cdef double[:,::1] Ad = A44d
    cdef double[:,::1] dAd = dA44d

    cdef int i0, i1, i2, i3
    cdef double x0, x1, x2, x3
    cdef double u0, u1, u2, u3
    cdef double t0, t1, t2, t3
    cdef double extrap0, extrap1, extrap2, extrap3
    
    cdef double Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3, Phi2_0, Phi2_1, Phi2_2, Phi2_3, Phi3_0, Phi3_1, Phi3_2, Phi3_3
    cdef double dPhi0_0, dPhi0_1, dPhi0_2, dPhi0_3, dPhi1_0, dPhi1_1, dPhi1_2, dPhi1_3, dPhi2_0, dPhi2_1, dPhi2_2, dPhi2_3, dPhi3_0, dPhi3_1, dPhi3_2, dPhi3_3
    cdef double tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3, tp2_0, tp2_1, tp2_2, tp2_3, tp3_0, tp3_1, tp3_2, tp3_3

    cdef double[:,:,:,::1] C = coefs

    cdef double[:] vals = np.zeros(N)
    cdef double[:,::1] dvals = np.zeros((4,N))

    cdef double tpx_0, tpx_1, tpx_2, tpx_3
    cdef double tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[0,n]
            x1 = svec[1,n]
            x2 = svec[2,n]
            x3 = svec[3,n]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            t1 = u1-i1
            u2 = (x2 - start2)*dinv2
            i2 = <int> u2
            t2 = u2-i2
            u3 = (x3 - start3)*dinv3
            i3 = <int> u3
            t3 = u3-i3
            extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            extrap2 = 0 if (t2 < 0 or t2 >= 1) else 1
            extrap3 = 0 if (t3 < 0 or t3 >= 1) else 1
            tp0_0 = t0*t0*t0*extrap0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1*extrap1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
            tp2_0 = t2*t2*t2*extrap2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
            tp3_0 = t3*t3*t3*extrap3;  tp3_1 = t3*t3;  tp3_2 = t3;  tp3_3 = 1.0;
            Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
            Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
            Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
            Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)
            Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
            Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
            Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
            Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)
            Phi2_0 = (Ad[0,0]*tp2_0 + Ad[0,1]*tp2_1 + Ad[0,2]*tp2_2 + Ad[0,3]*tp2_3)
            Phi2_1 = (Ad[1,0]*tp2_0 + Ad[1,1]*tp2_1 + Ad[1,2]*tp2_2 + Ad[1,3]*tp2_3)
            Phi2_2 = (Ad[2,0]*tp2_0 + Ad[2,1]*tp2_1 + Ad[2,2]*tp2_2 + Ad[2,3]*tp2_3)
            Phi2_3 = (Ad[3,0]*tp2_0 + Ad[3,1]*tp2_1 + Ad[3,2]*tp2_2 + Ad[3,3]*tp2_3)
            Phi3_0 = (Ad[0,0]*tp3_0 + Ad[0,1]*tp3_1 + Ad[0,2]*tp3_2 + Ad[0,3]*tp3_3)
            Phi3_1 = (Ad[1,0]*tp3_0 + Ad[1,1]*tp3_1 + Ad[1,2]*tp3_2 + Ad[1,3]*tp3_3)
            Phi3_2 = (Ad[2,0]*tp3_0 + Ad[2,1]*tp3_1 + Ad[2,2]*tp3_2 + Ad[2,3]*tp3_3)
            Phi3_3 = (Ad[3,0]*tp3_0 + Ad[3,1]*tp3_1 + Ad[3,2]*tp3_2 + Ad[3,3]*tp3_3)
            dPhi0_0 = (dAd[0,0]*tp0_0 + dAd[0,1]*tp0_1 + dAd[0,2]*tp0_2 + dAd[0,3]*tp0_3)*dinv0
            dPhi0_1 = (dAd[1,0]*tp0_0 + dAd[1,1]*tp0_1 + dAd[1,2]*tp0_2 + dAd[1,3]*tp0_3)*dinv0
            dPhi0_2 = (dAd[2,0]*tp0_0 + dAd[2,1]*tp0_1 + dAd[2,2]*tp0_2 + dAd[2,3]*tp0_3)*dinv0
            dPhi0_3 = (dAd[3,0]*tp0_0 + dAd[3,1]*tp0_1 + dAd[3,2]*tp0_2 + dAd[3,3]*tp0_3)*dinv0
            dPhi1_0 = (dAd[0,0]*tp1_0 + dAd[0,1]*tp1_1 + dAd[0,2]*tp1_2 + dAd[0,3]*tp1_3)*dinv1
            dPhi1_1 = (dAd[1,0]*tp1_0 + dAd[1,1]*tp1_1 + dAd[1,2]*tp1_2 + dAd[1,3]*tp1_3)*dinv1
            dPhi1_2 = (dAd[2,0]*tp1_0 + dAd[2,1]*tp1_1 + dAd[2,2]*tp1_2 + dAd[2,3]*tp1_3)*dinv1
            dPhi1_3 = (dAd[3,0]*tp1_0 + dAd[3,1]*tp1_1 + dAd[3,2]*tp1_2 + dAd[3,3]*tp1_3)*dinv1
            dPhi2_0 = (dAd[0,0]*tp2_0 + dAd[0,1]*tp2_1 + dAd[0,2]*tp2_2 + dAd[0,3]*tp2_3)*dinv2
            dPhi2_1 = (dAd[1,0]*tp2_0 + dAd[1,1]*tp2_1 + dAd[1,2]*tp2_2 + dAd[1,3]*tp2_3)*dinv2
            dPhi2_2 = (dAd[2,0]*tp2_0 + dAd[2,1]*tp2_1 + dAd[2,2]*tp2_2 + dAd[2,3]*tp2_3)*dinv2
            dPhi2_3 = (dAd[3,0]*tp2_0 + dAd[3,1]*tp2_1 + dAd[3,2]*tp2_2 + dAd[3,3]*tp2_3)*dinv2
            dPhi3_0 = (dAd[0,0]*tp3_0 + dAd[0,1]*tp3_1 + dAd[0,2]*tp3_2 + dAd[0,3]*tp3_3)*dinv3
            dPhi3_1 = (dAd[1,0]*tp3_0 + dAd[1,1]*tp3_1 + dAd[1,2]*tp3_2 + dAd[1,3]*tp3_3)*dinv3
            dPhi3_2 = (dAd[2,0]*tp3_0 + dAd[2,1]*tp3_1 + dAd[2,2]*tp3_2 + dAd[2,3]*tp3_3)*dinv3
            dPhi3_3 = (dAd[3,0]*tp3_0 + dAd[3,1]*tp3_1 + dAd[3,2]*tp3_2 + dAd[3,3]*tp3_3)*dinv3


            vals[n] = Phi0_0*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+0,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+0,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+0,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+0,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+3,i3+3])))) + Phi0_1*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+1,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+1,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+1,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+1,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+3,i3+3])))) + Phi0_2*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+2,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+2,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+2,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+2,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+3,i3+3])))) + Phi0_3*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+3,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+3,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+3,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+3,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+3,i3+3]))))

            dvals[0,n] = dPhi0_0*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+0,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+0,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+0,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+0,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+3,i3+3])))) + dPhi0_1*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+1,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+1,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+1,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+1,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+3,i3+3])))) + dPhi0_2*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+2,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+2,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+2,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+2,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+3,i3+3])))) + dPhi0_3*(Phi1_0*(Phi2_0*(Phi3_0*(C[i0+3,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(Phi3_0*(C[i0+3,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(Phi3_0*(C[i0+3,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(Phi3_0*(C[i0+3,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+3,i3+3])))) 
            dvals[1,n] = Phi0_0*(dPhi1_0*(Phi2_0*(Phi3_0*(C[i0+0,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+3,i3+3]))) + dPhi1_1*(Phi2_0*(Phi3_0*(C[i0+0,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+3,i3+3]))) + dPhi1_2*(Phi2_0*(Phi3_0*(C[i0+0,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+3,i3+3]))) + dPhi1_3*(Phi2_0*(Phi3_0*(C[i0+0,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+0,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+0,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+0,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+3,i3+3])))) + Phi0_1*(dPhi1_0*(Phi2_0*(Phi3_0*(C[i0+1,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+3,i3+3]))) + dPhi1_1*(Phi2_0*(Phi3_0*(C[i0+1,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+3,i3+3]))) + dPhi1_2*(Phi2_0*(Phi3_0*(C[i0+1,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+3,i3+3]))) + dPhi1_3*(Phi2_0*(Phi3_0*(C[i0+1,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+1,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+1,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+1,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+3,i3+3])))) + Phi0_2*(dPhi1_0*(Phi2_0*(Phi3_0*(C[i0+2,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+3,i3+3]))) + dPhi1_1*(Phi2_0*(Phi3_0*(C[i0+2,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+3,i3+3]))) + dPhi1_2*(Phi2_0*(Phi3_0*(C[i0+2,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+3,i3+3]))) + dPhi1_3*(Phi2_0*(Phi3_0*(C[i0+2,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+2,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+2,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+2,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+3,i3+3])))) + Phi0_3*(dPhi1_0*(Phi2_0*(Phi3_0*(C[i0+3,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+3,i3+3]))) + dPhi1_1*(Phi2_0*(Phi3_0*(C[i0+3,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+3,i3+3]))) + dPhi1_2*(Phi2_0*(Phi3_0*(C[i0+3,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+3,i3+3]))) + dPhi1_3*(Phi2_0*(Phi3_0*(C[i0+3,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+0,i3+3])) + Phi2_1*(Phi3_0*(C[i0+3,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+1,i3+3])) + Phi2_2*(Phi3_0*(C[i0+3,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+2,i3+3])) + Phi2_3*(Phi3_0*(C[i0+3,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+3,i3+3])))) 
            dvals[2,n] = Phi0_0*(Phi1_0*(dPhi2_0*(Phi3_0*(C[i0+0,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+0,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+0,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+0,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+0,i2+3,i3+3]))) + Phi1_1*(dPhi2_0*(Phi3_0*(C[i0+0,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+0,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+0,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+0,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+1,i2+3,i3+3]))) + Phi1_2*(dPhi2_0*(Phi3_0*(C[i0+0,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+0,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+0,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+0,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+2,i2+3,i3+3]))) + Phi1_3*(dPhi2_0*(Phi3_0*(C[i0+0,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+0,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+0,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+0,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+0,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+0,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+0,i1+3,i2+3,i3+3])))) + Phi0_1*(Phi1_0*(dPhi2_0*(Phi3_0*(C[i0+1,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+1,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+1,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+1,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+0,i2+3,i3+3]))) + Phi1_1*(dPhi2_0*(Phi3_0*(C[i0+1,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+1,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+1,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+1,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+1,i2+3,i3+3]))) + Phi1_2*(dPhi2_0*(Phi3_0*(C[i0+1,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+1,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+1,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+1,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+2,i2+3,i3+3]))) + Phi1_3*(dPhi2_0*(Phi3_0*(C[i0+1,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+1,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+1,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+1,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+1,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+1,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+1,i1+3,i2+3,i3+3])))) + Phi0_2*(Phi1_0*(dPhi2_0*(Phi3_0*(C[i0+2,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+2,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+2,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+2,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+0,i2+3,i3+3]))) + Phi1_1*(dPhi2_0*(Phi3_0*(C[i0+2,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+2,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+2,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+2,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+1,i2+3,i3+3]))) + Phi1_2*(dPhi2_0*(Phi3_0*(C[i0+2,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+2,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+2,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+2,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+2,i2+3,i3+3]))) + Phi1_3*(dPhi2_0*(Phi3_0*(C[i0+2,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+2,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+2,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+2,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+2,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+2,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+2,i1+3,i2+3,i3+3])))) + Phi0_3*(Phi1_0*(dPhi2_0*(Phi3_0*(C[i0+3,i1+0,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+3,i1+0,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+3,i1+0,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+3,i1+0,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+0,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+0,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+0,i2+3,i3+3]))) + Phi1_1*(dPhi2_0*(Phi3_0*(C[i0+3,i1+1,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+3,i1+1,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+3,i1+1,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+3,i1+1,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+1,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+1,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+1,i2+3,i3+3]))) + Phi1_2*(dPhi2_0*(Phi3_0*(C[i0+3,i1+2,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+3,i1+2,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+3,i1+2,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+3,i1+2,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+2,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+2,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+2,i2+3,i3+3]))) + Phi1_3*(dPhi2_0*(Phi3_0*(C[i0+3,i1+3,i2+0,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+0,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+0,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+0,i3+3])) + dPhi2_1*(Phi3_0*(C[i0+3,i1+3,i2+1,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+1,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+1,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+1,i3+3])) + dPhi2_2*(Phi3_0*(C[i0+3,i1+3,i2+2,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+2,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+2,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+2,i3+3])) + dPhi2_3*(Phi3_0*(C[i0+3,i1+3,i2+3,i3+0]) + Phi3_1*(C[i0+3,i1+3,i2+3,i3+1]) + Phi3_2*(C[i0+3,i1+3,i2+3,i3+2]) + Phi3_3*(C[i0+3,i1+3,i2+3,i3+3])))) 
            dvals[3,n] = Phi0_0*(Phi1_0*(Phi2_0*(dPhi3_0*(C[i0+0,i1+0,i2+0,i3+0]) + dPhi3_1*(C[i0+0,i1+0,i2+0,i3+1]) + dPhi3_2*(C[i0+0,i1+0,i2+0,i3+2]) + dPhi3_3*(C[i0+0,i1+0,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+0,i1+0,i2+1,i3+0]) + dPhi3_1*(C[i0+0,i1+0,i2+1,i3+1]) + dPhi3_2*(C[i0+0,i1+0,i2+1,i3+2]) + dPhi3_3*(C[i0+0,i1+0,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+0,i1+0,i2+2,i3+0]) + dPhi3_1*(C[i0+0,i1+0,i2+2,i3+1]) + dPhi3_2*(C[i0+0,i1+0,i2+2,i3+2]) + dPhi3_3*(C[i0+0,i1+0,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+0,i1+0,i2+3,i3+0]) + dPhi3_1*(C[i0+0,i1+0,i2+3,i3+1]) + dPhi3_2*(C[i0+0,i1+0,i2+3,i3+2]) + dPhi3_3*(C[i0+0,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(dPhi3_0*(C[i0+0,i1+1,i2+0,i3+0]) + dPhi3_1*(C[i0+0,i1+1,i2+0,i3+1]) + dPhi3_2*(C[i0+0,i1+1,i2+0,i3+2]) + dPhi3_3*(C[i0+0,i1+1,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+0,i1+1,i2+1,i3+0]) + dPhi3_1*(C[i0+0,i1+1,i2+1,i3+1]) + dPhi3_2*(C[i0+0,i1+1,i2+1,i3+2]) + dPhi3_3*(C[i0+0,i1+1,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+0,i1+1,i2+2,i3+0]) + dPhi3_1*(C[i0+0,i1+1,i2+2,i3+1]) + dPhi3_2*(C[i0+0,i1+1,i2+2,i3+2]) + dPhi3_3*(C[i0+0,i1+1,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+0,i1+1,i2+3,i3+0]) + dPhi3_1*(C[i0+0,i1+1,i2+3,i3+1]) + dPhi3_2*(C[i0+0,i1+1,i2+3,i3+2]) + dPhi3_3*(C[i0+0,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(dPhi3_0*(C[i0+0,i1+2,i2+0,i3+0]) + dPhi3_1*(C[i0+0,i1+2,i2+0,i3+1]) + dPhi3_2*(C[i0+0,i1+2,i2+0,i3+2]) + dPhi3_3*(C[i0+0,i1+2,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+0,i1+2,i2+1,i3+0]) + dPhi3_1*(C[i0+0,i1+2,i2+1,i3+1]) + dPhi3_2*(C[i0+0,i1+2,i2+1,i3+2]) + dPhi3_3*(C[i0+0,i1+2,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+0,i1+2,i2+2,i3+0]) + dPhi3_1*(C[i0+0,i1+2,i2+2,i3+1]) + dPhi3_2*(C[i0+0,i1+2,i2+2,i3+2]) + dPhi3_3*(C[i0+0,i1+2,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+0,i1+2,i2+3,i3+0]) + dPhi3_1*(C[i0+0,i1+2,i2+3,i3+1]) + dPhi3_2*(C[i0+0,i1+2,i2+3,i3+2]) + dPhi3_3*(C[i0+0,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(dPhi3_0*(C[i0+0,i1+3,i2+0,i3+0]) + dPhi3_1*(C[i0+0,i1+3,i2+0,i3+1]) + dPhi3_2*(C[i0+0,i1+3,i2+0,i3+2]) + dPhi3_3*(C[i0+0,i1+3,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+0,i1+3,i2+1,i3+0]) + dPhi3_1*(C[i0+0,i1+3,i2+1,i3+1]) + dPhi3_2*(C[i0+0,i1+3,i2+1,i3+2]) + dPhi3_3*(C[i0+0,i1+3,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+0,i1+3,i2+2,i3+0]) + dPhi3_1*(C[i0+0,i1+3,i2+2,i3+1]) + dPhi3_2*(C[i0+0,i1+3,i2+2,i3+2]) + dPhi3_3*(C[i0+0,i1+3,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+0,i1+3,i2+3,i3+0]) + dPhi3_1*(C[i0+0,i1+3,i2+3,i3+1]) + dPhi3_2*(C[i0+0,i1+3,i2+3,i3+2]) + dPhi3_3*(C[i0+0,i1+3,i2+3,i3+3])))) + Phi0_1*(Phi1_0*(Phi2_0*(dPhi3_0*(C[i0+1,i1+0,i2+0,i3+0]) + dPhi3_1*(C[i0+1,i1+0,i2+0,i3+1]) + dPhi3_2*(C[i0+1,i1+0,i2+0,i3+2]) + dPhi3_3*(C[i0+1,i1+0,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+1,i1+0,i2+1,i3+0]) + dPhi3_1*(C[i0+1,i1+0,i2+1,i3+1]) + dPhi3_2*(C[i0+1,i1+0,i2+1,i3+2]) + dPhi3_3*(C[i0+1,i1+0,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+1,i1+0,i2+2,i3+0]) + dPhi3_1*(C[i0+1,i1+0,i2+2,i3+1]) + dPhi3_2*(C[i0+1,i1+0,i2+2,i3+2]) + dPhi3_3*(C[i0+1,i1+0,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+1,i1+0,i2+3,i3+0]) + dPhi3_1*(C[i0+1,i1+0,i2+3,i3+1]) + dPhi3_2*(C[i0+1,i1+0,i2+3,i3+2]) + dPhi3_3*(C[i0+1,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(dPhi3_0*(C[i0+1,i1+1,i2+0,i3+0]) + dPhi3_1*(C[i0+1,i1+1,i2+0,i3+1]) + dPhi3_2*(C[i0+1,i1+1,i2+0,i3+2]) + dPhi3_3*(C[i0+1,i1+1,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+1,i1+1,i2+1,i3+0]) + dPhi3_1*(C[i0+1,i1+1,i2+1,i3+1]) + dPhi3_2*(C[i0+1,i1+1,i2+1,i3+2]) + dPhi3_3*(C[i0+1,i1+1,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+1,i1+1,i2+2,i3+0]) + dPhi3_1*(C[i0+1,i1+1,i2+2,i3+1]) + dPhi3_2*(C[i0+1,i1+1,i2+2,i3+2]) + dPhi3_3*(C[i0+1,i1+1,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+1,i1+1,i2+3,i3+0]) + dPhi3_1*(C[i0+1,i1+1,i2+3,i3+1]) + dPhi3_2*(C[i0+1,i1+1,i2+3,i3+2]) + dPhi3_3*(C[i0+1,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(dPhi3_0*(C[i0+1,i1+2,i2+0,i3+0]) + dPhi3_1*(C[i0+1,i1+2,i2+0,i3+1]) + dPhi3_2*(C[i0+1,i1+2,i2+0,i3+2]) + dPhi3_3*(C[i0+1,i1+2,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+1,i1+2,i2+1,i3+0]) + dPhi3_1*(C[i0+1,i1+2,i2+1,i3+1]) + dPhi3_2*(C[i0+1,i1+2,i2+1,i3+2]) + dPhi3_3*(C[i0+1,i1+2,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+1,i1+2,i2+2,i3+0]) + dPhi3_1*(C[i0+1,i1+2,i2+2,i3+1]) + dPhi3_2*(C[i0+1,i1+2,i2+2,i3+2]) + dPhi3_3*(C[i0+1,i1+2,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+1,i1+2,i2+3,i3+0]) + dPhi3_1*(C[i0+1,i1+2,i2+3,i3+1]) + dPhi3_2*(C[i0+1,i1+2,i2+3,i3+2]) + dPhi3_3*(C[i0+1,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(dPhi3_0*(C[i0+1,i1+3,i2+0,i3+0]) + dPhi3_1*(C[i0+1,i1+3,i2+0,i3+1]) + dPhi3_2*(C[i0+1,i1+3,i2+0,i3+2]) + dPhi3_3*(C[i0+1,i1+3,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+1,i1+3,i2+1,i3+0]) + dPhi3_1*(C[i0+1,i1+3,i2+1,i3+1]) + dPhi3_2*(C[i0+1,i1+3,i2+1,i3+2]) + dPhi3_3*(C[i0+1,i1+3,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+1,i1+3,i2+2,i3+0]) + dPhi3_1*(C[i0+1,i1+3,i2+2,i3+1]) + dPhi3_2*(C[i0+1,i1+3,i2+2,i3+2]) + dPhi3_3*(C[i0+1,i1+3,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+1,i1+3,i2+3,i3+0]) + dPhi3_1*(C[i0+1,i1+3,i2+3,i3+1]) + dPhi3_2*(C[i0+1,i1+3,i2+3,i3+2]) + dPhi3_3*(C[i0+1,i1+3,i2+3,i3+3])))) + Phi0_2*(Phi1_0*(Phi2_0*(dPhi3_0*(C[i0+2,i1+0,i2+0,i3+0]) + dPhi3_1*(C[i0+2,i1+0,i2+0,i3+1]) + dPhi3_2*(C[i0+2,i1+0,i2+0,i3+2]) + dPhi3_3*(C[i0+2,i1+0,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+2,i1+0,i2+1,i3+0]) + dPhi3_1*(C[i0+2,i1+0,i2+1,i3+1]) + dPhi3_2*(C[i0+2,i1+0,i2+1,i3+2]) + dPhi3_3*(C[i0+2,i1+0,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+2,i1+0,i2+2,i3+0]) + dPhi3_1*(C[i0+2,i1+0,i2+2,i3+1]) + dPhi3_2*(C[i0+2,i1+0,i2+2,i3+2]) + dPhi3_3*(C[i0+2,i1+0,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+2,i1+0,i2+3,i3+0]) + dPhi3_1*(C[i0+2,i1+0,i2+3,i3+1]) + dPhi3_2*(C[i0+2,i1+0,i2+3,i3+2]) + dPhi3_3*(C[i0+2,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(dPhi3_0*(C[i0+2,i1+1,i2+0,i3+0]) + dPhi3_1*(C[i0+2,i1+1,i2+0,i3+1]) + dPhi3_2*(C[i0+2,i1+1,i2+0,i3+2]) + dPhi3_3*(C[i0+2,i1+1,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+2,i1+1,i2+1,i3+0]) + dPhi3_1*(C[i0+2,i1+1,i2+1,i3+1]) + dPhi3_2*(C[i0+2,i1+1,i2+1,i3+2]) + dPhi3_3*(C[i0+2,i1+1,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+2,i1+1,i2+2,i3+0]) + dPhi3_1*(C[i0+2,i1+1,i2+2,i3+1]) + dPhi3_2*(C[i0+2,i1+1,i2+2,i3+2]) + dPhi3_3*(C[i0+2,i1+1,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+2,i1+1,i2+3,i3+0]) + dPhi3_1*(C[i0+2,i1+1,i2+3,i3+1]) + dPhi3_2*(C[i0+2,i1+1,i2+3,i3+2]) + dPhi3_3*(C[i0+2,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(dPhi3_0*(C[i0+2,i1+2,i2+0,i3+0]) + dPhi3_1*(C[i0+2,i1+2,i2+0,i3+1]) + dPhi3_2*(C[i0+2,i1+2,i2+0,i3+2]) + dPhi3_3*(C[i0+2,i1+2,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+2,i1+2,i2+1,i3+0]) + dPhi3_1*(C[i0+2,i1+2,i2+1,i3+1]) + dPhi3_2*(C[i0+2,i1+2,i2+1,i3+2]) + dPhi3_3*(C[i0+2,i1+2,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+2,i1+2,i2+2,i3+0]) + dPhi3_1*(C[i0+2,i1+2,i2+2,i3+1]) + dPhi3_2*(C[i0+2,i1+2,i2+2,i3+2]) + dPhi3_3*(C[i0+2,i1+2,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+2,i1+2,i2+3,i3+0]) + dPhi3_1*(C[i0+2,i1+2,i2+3,i3+1]) + dPhi3_2*(C[i0+2,i1+2,i2+3,i3+2]) + dPhi3_3*(C[i0+2,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(dPhi3_0*(C[i0+2,i1+3,i2+0,i3+0]) + dPhi3_1*(C[i0+2,i1+3,i2+0,i3+1]) + dPhi3_2*(C[i0+2,i1+3,i2+0,i3+2]) + dPhi3_3*(C[i0+2,i1+3,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+2,i1+3,i2+1,i3+0]) + dPhi3_1*(C[i0+2,i1+3,i2+1,i3+1]) + dPhi3_2*(C[i0+2,i1+3,i2+1,i3+2]) + dPhi3_3*(C[i0+2,i1+3,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+2,i1+3,i2+2,i3+0]) + dPhi3_1*(C[i0+2,i1+3,i2+2,i3+1]) + dPhi3_2*(C[i0+2,i1+3,i2+2,i3+2]) + dPhi3_3*(C[i0+2,i1+3,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+2,i1+3,i2+3,i3+0]) + dPhi3_1*(C[i0+2,i1+3,i2+3,i3+1]) + dPhi3_2*(C[i0+2,i1+3,i2+3,i3+2]) + dPhi3_3*(C[i0+2,i1+3,i2+3,i3+3])))) + Phi0_3*(Phi1_0*(Phi2_0*(dPhi3_0*(C[i0+3,i1+0,i2+0,i3+0]) + dPhi3_1*(C[i0+3,i1+0,i2+0,i3+1]) + dPhi3_2*(C[i0+3,i1+0,i2+0,i3+2]) + dPhi3_3*(C[i0+3,i1+0,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+3,i1+0,i2+1,i3+0]) + dPhi3_1*(C[i0+3,i1+0,i2+1,i3+1]) + dPhi3_2*(C[i0+3,i1+0,i2+1,i3+2]) + dPhi3_3*(C[i0+3,i1+0,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+3,i1+0,i2+2,i3+0]) + dPhi3_1*(C[i0+3,i1+0,i2+2,i3+1]) + dPhi3_2*(C[i0+3,i1+0,i2+2,i3+2]) + dPhi3_3*(C[i0+3,i1+0,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+3,i1+0,i2+3,i3+0]) + dPhi3_1*(C[i0+3,i1+0,i2+3,i3+1]) + dPhi3_2*(C[i0+3,i1+0,i2+3,i3+2]) + dPhi3_3*(C[i0+3,i1+0,i2+3,i3+3]))) + Phi1_1*(Phi2_0*(dPhi3_0*(C[i0+3,i1+1,i2+0,i3+0]) + dPhi3_1*(C[i0+3,i1+1,i2+0,i3+1]) + dPhi3_2*(C[i0+3,i1+1,i2+0,i3+2]) + dPhi3_3*(C[i0+3,i1+1,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+3,i1+1,i2+1,i3+0]) + dPhi3_1*(C[i0+3,i1+1,i2+1,i3+1]) + dPhi3_2*(C[i0+3,i1+1,i2+1,i3+2]) + dPhi3_3*(C[i0+3,i1+1,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+3,i1+1,i2+2,i3+0]) + dPhi3_1*(C[i0+3,i1+1,i2+2,i3+1]) + dPhi3_2*(C[i0+3,i1+1,i2+2,i3+2]) + dPhi3_3*(C[i0+3,i1+1,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+3,i1+1,i2+3,i3+0]) + dPhi3_1*(C[i0+3,i1+1,i2+3,i3+1]) + dPhi3_2*(C[i0+3,i1+1,i2+3,i3+2]) + dPhi3_3*(C[i0+3,i1+1,i2+3,i3+3]))) + Phi1_2*(Phi2_0*(dPhi3_0*(C[i0+3,i1+2,i2+0,i3+0]) + dPhi3_1*(C[i0+3,i1+2,i2+0,i3+1]) + dPhi3_2*(C[i0+3,i1+2,i2+0,i3+2]) + dPhi3_3*(C[i0+3,i1+2,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+3,i1+2,i2+1,i3+0]) + dPhi3_1*(C[i0+3,i1+2,i2+1,i3+1]) + dPhi3_2*(C[i0+3,i1+2,i2+1,i3+2]) + dPhi3_3*(C[i0+3,i1+2,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+3,i1+2,i2+2,i3+0]) + dPhi3_1*(C[i0+3,i1+2,i2+2,i3+1]) + dPhi3_2*(C[i0+3,i1+2,i2+2,i3+2]) + dPhi3_3*(C[i0+3,i1+2,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+3,i1+2,i2+3,i3+0]) + dPhi3_1*(C[i0+3,i1+2,i2+3,i3+1]) + dPhi3_2*(C[i0+3,i1+2,i2+3,i3+2]) + dPhi3_3*(C[i0+3,i1+2,i2+3,i3+3]))) + Phi1_3*(Phi2_0*(dPhi3_0*(C[i0+3,i1+3,i2+0,i3+0]) + dPhi3_1*(C[i0+3,i1+3,i2+0,i3+1]) + dPhi3_2*(C[i0+3,i1+3,i2+0,i3+2]) + dPhi3_3*(C[i0+3,i1+3,i2+0,i3+3])) + Phi2_1*(dPhi3_0*(C[i0+3,i1+3,i2+1,i3+0]) + dPhi3_1*(C[i0+3,i1+3,i2+1,i3+1]) + dPhi3_2*(C[i0+3,i1+3,i2+1,i3+2]) + dPhi3_3*(C[i0+3,i1+3,i2+1,i3+3])) + Phi2_2*(dPhi3_0*(C[i0+3,i1+3,i2+2,i3+0]) + dPhi3_1*(C[i0+3,i1+3,i2+2,i3+1]) + dPhi3_2*(C[i0+3,i1+3,i2+2,i3+2]) + dPhi3_3*(C[i0+3,i1+3,i2+2,i3+3])) + Phi2_3*(dPhi3_0*(C[i0+3,i1+3,i2+3,i3+0]) + dPhi3_1*(C[i0+3,i1+3,i2+3,i3+1]) + dPhi3_2*(C[i0+3,i1+3,i2+3,i3+2]) + dPhi3_3*(C[i0+3,i1+3,i2+3,i3+3])))) 

    return [vals,dvals]
