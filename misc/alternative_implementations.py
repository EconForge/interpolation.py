'''
Run 'python alternative_implementations.py`  (no external dependences).
We compare several variants of the code to interpolate cubic splines based on the automatically generated one. They are all included below.
    
    - eval_cubic_spline_3 and eval_cubic_spline_3: the first interpolates at one point and the second repeatedly calls it
    - vec_eval_cubic_spline_3_inlined: everything is copied in the loop manually
    - vec_eval_cubic_spline_3_inlined_columns: same as above (but points are ordered differently)
    - kernel and vec_eval_cubic_spline_3_kernel: the kernel is called repeatedly (it takes all arrays and an index)
    - vec_eval_cubic_spline_3_lesswork: fully inlined version which does not rescale and does not interpolate (for comparisons with interpolations.jl)

The benchmark test consists in interpolating random coefficients from a 50x50x50 grid on N=10^6 points. Here are the results:


Repeated calls:                       0.24852252006530762
Manually inlined (points in rows):    0.123809814453125
Manually inlined (points in columns): 0.12022709846496582
cuda-like kernel:                     0.22097206115722656
No extrap / no rescale:               0.0850365161895752
Cythonized (not shown):               0.11784672737121582 

It seems that inlining is not efficient in this case or that it doesn't get enabled.
The performance without extrapolation/rescaling are very similar to those of interpolations.jl which also uses llvm. The penalty cost of those operations seem hard to interpret. (rescaling seems pretty trivial and extrapolation doesn't seem to cost anything).
The Cython version (in eval_cubic_spines_cython.pyx run from main.py) is only slightly faster than numba's.

'''

from numba import njit
from math import floor
from numpy import array, zeros

Ad = array([
#      t^3       t^2        t        1
   [-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0],
   [ 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0],
   [-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0],
   [ 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0]
])

dAd = zeros((4,4))
for i in range(1,4):
    dAd[:,i] = Ad[:,i-1]*(4-i)


d2Ad = zeros((4,4))
for i in range(1,4):
    d2Ad[:,i] = dAd[:,i-1]*(4-i)


@njit(cache=True)
def eval_cubic_spline_3(a, b, orders, coefs, point, Ad, dAd):
    """
    Evaluate the b - spline b.

    Args:
        a: (todo): write your description
        b: (todo): write your description
        orders: (int): write your description
        coefs: (todo): write your description
        point: (array): write your description
        Ad: (todo): write your description
        dAd: (todo): write your description
    """

    M0 = orders[0]
    start0 = a[0]
    dinv0 = (orders[0]-1.0)/(b[0]-a[0])
    M1 = orders[1]
    start1 = a[1]
    dinv1 = (orders[1]-1.0)/(b[1]-a[1])
    M2 = orders[2]
    start2 = a[2]
    dinv2 = (orders[2]-1.0)/(b[2]-a[2])
    x0 = point[0]
    x1 = point[1]
    x2 = point[2]
    u0 = (x0 - start0)*dinv0
    i0 = int( floor( u0 ) )
    i0 = max( min(i0,M0-2), 0 )
    t0 = u0-i0
    u1 = (x1 - start1)*dinv1
    i1 = int( floor( u1 ) )
    i1 = max( min(i1,M1-2), 0 )
    t1 = u1-i1
    u2 = (x2 - start2)*dinv2
    i2 = int( floor( u2 ) )
    i2 = max( min(i2,M2-2), 0 )
    t2 = u2-i2
    tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
    tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
    tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
    Phi0_0 = 0
    Phi0_1 = 0
    Phi0_2 = 0
    Phi0_3 = 0
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

    Phi1_0 = 0
    Phi1_1 = 0
    Phi1_2 = 0
    Phi1_3 = 0
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

    Phi2_0 = 0
    Phi2_1 = 0
    Phi2_2 = 0
    Phi2_3 = 0
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


    t = Phi0_0*(Phi1_0*(Phi2_0*(coefs[i0+0,i1+0,i2+0]) + Phi2_1*(coefs[i0+0,i1+0,i2+1]) + Phi2_2*(coefs[i0+0,i1+0,i2+2]) + Phi2_3*(coefs[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+0,i1+1,i2+0]) + Phi2_1*(coefs[i0+0,i1+1,i2+1]) + Phi2_2*(coefs[i0+0,i1+1,i2+2]) + Phi2_3*(coefs[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+0,i1+2,i2+0]) + Phi2_1*(coefs[i0+0,i1+2,i2+1]) + Phi2_2*(coefs[i0+0,i1+2,i2+2]) + Phi2_3*(coefs[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+0,i1+3,i2+0]) + Phi2_1*(coefs[i0+0,i1+3,i2+1]) + Phi2_2*(coefs[i0+0,i1+3,i2+2]) + Phi2_3*(coefs[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(coefs[i0+1,i1+0,i2+0]) + Phi2_1*(coefs[i0+1,i1+0,i2+1]) + Phi2_2*(coefs[i0+1,i1+0,i2+2]) + Phi2_3*(coefs[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+1,i1+1,i2+0]) + Phi2_1*(coefs[i0+1,i1+1,i2+1]) + Phi2_2*(coefs[i0+1,i1+1,i2+2]) + Phi2_3*(coefs[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+1,i1+2,i2+0]) + Phi2_1*(coefs[i0+1,i1+2,i2+1]) + Phi2_2*(coefs[i0+1,i1+2,i2+2]) + Phi2_3*(coefs[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+1,i1+3,i2+0]) + Phi2_1*(coefs[i0+1,i1+3,i2+1]) + Phi2_2*(coefs[i0+1,i1+3,i2+2]) + Phi2_3*(coefs[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(coefs[i0+2,i1+0,i2+0]) + Phi2_1*(coefs[i0+2,i1+0,i2+1]) + Phi2_2*(coefs[i0+2,i1+0,i2+2]) + Phi2_3*(coefs[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+2,i1+1,i2+0]) + Phi2_1*(coefs[i0+2,i1+1,i2+1]) + Phi2_2*(coefs[i0+2,i1+1,i2+2]) + Phi2_3*(coefs[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+2,i1+2,i2+0]) + Phi2_1*(coefs[i0+2,i1+2,i2+1]) + Phi2_2*(coefs[i0+2,i1+2,i2+2]) + Phi2_3*(coefs[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+2,i1+3,i2+0]) + Phi2_1*(coefs[i0+2,i1+3,i2+1]) + Phi2_2*(coefs[i0+2,i1+3,i2+2]) + Phi2_3*(coefs[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(coefs[i0+3,i1+0,i2+0]) + Phi2_1*(coefs[i0+3,i1+0,i2+1]) + Phi2_2*(coefs[i0+3,i1+0,i2+2]) + Phi2_3*(coefs[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+3,i1+1,i2+0]) + Phi2_1*(coefs[i0+3,i1+1,i2+1]) + Phi2_2*(coefs[i0+3,i1+1,i2+2]) + Phi2_3*(coefs[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+3,i1+2,i2+0]) + Phi2_1*(coefs[i0+3,i1+2,i2+1]) + Phi2_2*(coefs[i0+3,i1+2,i2+2]) + Phi2_3*(coefs[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+3,i1+3,i2+0]) + Phi2_1*(coefs[i0+3,i1+3,i2+1]) + Phi2_2*(coefs[i0+3,i1+3,i2+2]) + Phi2_3*(coefs[i0+3,i1+3,i2+3])))

    return t


@njit(cache=True)
def vec_eval_cubic_spline_3(a, b, orders, coefs, points, values):
    """
    Evaluate a 2d polynomial of 2d polynomial.

    Args:
        a: (todo): write your description
        b: (todo): write your description
        orders: (int): write your description
        coefs: (todo): write your description
        points: (array): write your description
        values: (str): write your description
    """

    N = points.shape[0]

    for n in range(N):
        point = points[n, :]
        values[n] = eval_cubic_spline_3(a, b, orders, coefs, point, Ad, dAd)

@njit(cache=True)
def vec_eval_cubic_spline_3_inlined(a, b, orders, coefs, points, values):
    """
    Vec_eval_evalicic regularization.

    Args:
        a: (todo): write your description
        b: (todo): write your description
        orders: (int): write your description
        coefs: (todo): write your description
        points: (array): write your description
        values: (array): write your description
    """

    N = points.shape[0]

    for n in range(N):

        x0 = points[n,0]
        x1 = points[n,1]
        x2 = points[n,2]

        # # a bit faster ?
#        x0 = points[0,n]
#        x1 = points[1,n]
#        x2 = points[2,n]

        M0 = orders[0]
        start0 = a[0]
        dinv0 = (orders[0]-1.0)/(b[0]-a[0])
        M1 = orders[1]
        start1 = a[1]
        dinv1 = (orders[1]-1.0)/(b[1]-a[1])
        M2 = orders[2]
        start2 = a[2]
        dinv2 = (orders[2]-1.0)/(b[2]-a[2])

        u0 = (x0 - start0)*dinv0
        i0 = int( floor( u0 ) )
        i0 = max( min(i0,M0-2), 0 )
        t0 = u0-i0
        u1 = (x1 - start1)*dinv1
        i1 = int( floor( u1 ) )
        i1 = max( min(i1,M1-2), 0 )
        t1 = u1-i1
        u2 = (x2 - start2)*dinv2
        i2 = int( floor( u2 ) )
        i2 = max( min(i2,M2-2), 0 )
        t2 = u2-i2
        tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
        tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
        tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
        Phi0_0 = 0
        Phi0_1 = 0
        Phi0_2 = 0
        Phi0_3 = 0
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

        Phi1_0 = 0
        Phi1_1 = 0
        Phi1_2 = 0
        Phi1_3 = 0
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

        Phi2_0 = 0
        Phi2_1 = 0
        Phi2_2 = 0
        Phi2_3 = 0
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


        values[n] = Phi0_0*(Phi1_0*(Phi2_0*(coefs[i0+0,i1+0,i2+0]) + Phi2_1*(coefs[i0+0,i1+0,i2+1]) + Phi2_2*(coefs[i0+0,i1+0,i2+2]) + Phi2_3*(coefs[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+0,i1+1,i2+0]) + Phi2_1*(coefs[i0+0,i1+1,i2+1]) + Phi2_2*(coefs[i0+0,i1+1,i2+2]) + Phi2_3*(coefs[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+0,i1+2,i2+0]) + Phi2_1*(coefs[i0+0,i1+2,i2+1]) + Phi2_2*(coefs[i0+0,i1+2,i2+2]) + Phi2_3*(coefs[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+0,i1+3,i2+0]) + Phi2_1*(coefs[i0+0,i1+3,i2+1]) + Phi2_2*(coefs[i0+0,i1+3,i2+2]) + Phi2_3*(coefs[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(coefs[i0+1,i1+0,i2+0]) + Phi2_1*(coefs[i0+1,i1+0,i2+1]) + Phi2_2*(coefs[i0+1,i1+0,i2+2]) + Phi2_3*(coefs[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+1,i1+1,i2+0]) + Phi2_1*(coefs[i0+1,i1+1,i2+1]) + Phi2_2*(coefs[i0+1,i1+1,i2+2]) + Phi2_3*(coefs[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+1,i1+2,i2+0]) + Phi2_1*(coefs[i0+1,i1+2,i2+1]) + Phi2_2*(coefs[i0+1,i1+2,i2+2]) + Phi2_3*(coefs[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+1,i1+3,i2+0]) + Phi2_1*(coefs[i0+1,i1+3,i2+1]) + Phi2_2*(coefs[i0+1,i1+3,i2+2]) + Phi2_3*(coefs[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(coefs[i0+2,i1+0,i2+0]) + Phi2_1*(coefs[i0+2,i1+0,i2+1]) + Phi2_2*(coefs[i0+2,i1+0,i2+2]) + Phi2_3*(coefs[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+2,i1+1,i2+0]) + Phi2_1*(coefs[i0+2,i1+1,i2+1]) + Phi2_2*(coefs[i0+2,i1+1,i2+2]) + Phi2_3*(coefs[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+2,i1+2,i2+0]) + Phi2_1*(coefs[i0+2,i1+2,i2+1]) + Phi2_2*(coefs[i0+2,i1+2,i2+2]) + Phi2_3*(coefs[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+2,i1+3,i2+0]) + Phi2_1*(coefs[i0+2,i1+3,i2+1]) + Phi2_2*(coefs[i0+2,i1+3,i2+2]) + Phi2_3*(coefs[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(coefs[i0+3,i1+0,i2+0]) + Phi2_1*(coefs[i0+3,i1+0,i2+1]) + Phi2_2*(coefs[i0+3,i1+0,i2+2]) + Phi2_3*(coefs[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+3,i1+1,i2+0]) + Phi2_1*(coefs[i0+3,i1+1,i2+1]) + Phi2_2*(coefs[i0+3,i1+1,i2+2]) + Phi2_3*(coefs[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+3,i1+2,i2+0]) + Phi2_1*(coefs[i0+3,i1+2,i2+1]) + Phi2_2*(coefs[i0+3,i1+2,i2+2]) + Phi2_3*(coefs[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+3,i1+3,i2+0]) + Phi2_1*(coefs[i0+3,i1+3,i2+1]) + Phi2_2*(coefs[i0+3,i1+3,i2+2]) + Phi2_3*(coefs[i0+3,i1+3,i2+3])))

@njit(cache=True)
def vec_eval_cubic_spline_3_inlined_columns(a, b, orders, coefs, points, values):
    """
    R evaluate_cub_columnic_columns ( a b c dinv ).

    Args:
        a: (todo): write your description
        b: (todo): write your description
        orders: (int): write your description
        coefs: (todo): write your description
        points: (array): write your description
        values: (array): write your description
    """

    # N = points.shape[0]
    N = points.shape[1]

    M0 = orders[0]
    start0 = a[0]
    dinv0 = (orders[0]-1.0)/(b[0]-a[0])
    M1 = orders[1]
    start1 = a[1]
    dinv1 = (orders[1]-1.0)/(b[1]-a[1])
    M2 = orders[2]
    start2 = a[2]
    dinv2 = (orders[2]-1.0)/(b[2]-a[2])


    for n in range(N):

#        x0 = points[n,0]
#        x1 = points[n,1]
#        x2 = points[n,2]

        # # a bit faster ?
        x0 = points[0,n]
        x1 = points[1,n]
        x2 = points[2,n]

        u0 = (x0 - start0)*dinv0
        i0 = int( floor( u0 ) )
        i0 = max( min(i0,M0-2), 0 )
        t0 = u0-i0
        u1 = (x1 - start1)*dinv1
        i1 = int( floor( u1 ) )
        i1 = max( min(i1,M1-2), 0 )
        t1 = u1-i1
        u2 = (x2 - start2)*dinv2
        i2 = int( floor( u2 ) )
        i2 = max( min(i2,M2-2), 0 )
        t2 = u2-i2
        tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
        tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
        tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
        Phi0_0 = 0
        Phi0_1 = 0
        Phi0_2 = 0
        Phi0_3 = 0
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

        Phi1_0 = 0
        Phi1_1 = 0
        Phi1_2 = 0
        Phi1_3 = 0
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

        Phi2_0 = 0
        Phi2_1 = 0
        Phi2_2 = 0
        Phi2_3 = 0
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


        values[n] = Phi0_0*(Phi1_0*(Phi2_0*(coefs[i0+0,i1+0,i2+0]) + Phi2_1*(coefs[i0+0,i1+0,i2+1]) + Phi2_2*(coefs[i0+0,i1+0,i2+2]) + Phi2_3*(coefs[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+0,i1+1,i2+0]) + Phi2_1*(coefs[i0+0,i1+1,i2+1]) + Phi2_2*(coefs[i0+0,i1+1,i2+2]) + Phi2_3*(coefs[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+0,i1+2,i2+0]) + Phi2_1*(coefs[i0+0,i1+2,i2+1]) + Phi2_2*(coefs[i0+0,i1+2,i2+2]) + Phi2_3*(coefs[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+0,i1+3,i2+0]) + Phi2_1*(coefs[i0+0,i1+3,i2+1]) + Phi2_2*(coefs[i0+0,i1+3,i2+2]) + Phi2_3*(coefs[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(coefs[i0+1,i1+0,i2+0]) + Phi2_1*(coefs[i0+1,i1+0,i2+1]) + Phi2_2*(coefs[i0+1,i1+0,i2+2]) + Phi2_3*(coefs[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+1,i1+1,i2+0]) + Phi2_1*(coefs[i0+1,i1+1,i2+1]) + Phi2_2*(coefs[i0+1,i1+1,i2+2]) + Phi2_3*(coefs[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+1,i1+2,i2+0]) + Phi2_1*(coefs[i0+1,i1+2,i2+1]) + Phi2_2*(coefs[i0+1,i1+2,i2+2]) + Phi2_3*(coefs[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+1,i1+3,i2+0]) + Phi2_1*(coefs[i0+1,i1+3,i2+1]) + Phi2_2*(coefs[i0+1,i1+3,i2+2]) + Phi2_3*(coefs[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(coefs[i0+2,i1+0,i2+0]) + Phi2_1*(coefs[i0+2,i1+0,i2+1]) + Phi2_2*(coefs[i0+2,i1+0,i2+2]) + Phi2_3*(coefs[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+2,i1+1,i2+0]) + Phi2_1*(coefs[i0+2,i1+1,i2+1]) + Phi2_2*(coefs[i0+2,i1+1,i2+2]) + Phi2_3*(coefs[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+2,i1+2,i2+0]) + Phi2_1*(coefs[i0+2,i1+2,i2+1]) + Phi2_2*(coefs[i0+2,i1+2,i2+2]) + Phi2_3*(coefs[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+2,i1+3,i2+0]) + Phi2_1*(coefs[i0+2,i1+3,i2+1]) + Phi2_2*(coefs[i0+2,i1+3,i2+2]) + Phi2_3*(coefs[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(coefs[i0+3,i1+0,i2+0]) + Phi2_1*(coefs[i0+3,i1+0,i2+1]) + Phi2_2*(coefs[i0+3,i1+0,i2+2]) + Phi2_3*(coefs[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+3,i1+1,i2+0]) + Phi2_1*(coefs[i0+3,i1+1,i2+1]) + Phi2_2*(coefs[i0+3,i1+1,i2+2]) + Phi2_3*(coefs[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+3,i1+2,i2+0]) + Phi2_1*(coefs[i0+3,i1+2,i2+1]) + Phi2_2*(coefs[i0+3,i1+2,i2+2]) + Phi2_3*(coefs[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+3,i1+3,i2+0]) + Phi2_1*(coefs[i0+3,i1+3,i2+1]) + Phi2_2*(coefs[i0+3,i1+3,i2+2]) + Phi2_3*(coefs[i0+3,i1+3,i2+3])))

@njit(cache=True)
def vec_eval_cubic_spline_3_inlined_lesswork(orders, coefs, points, values, Ad, dAd):
    """
    Evaluate the cubic regularization.

    Args:
        orders: (int): write your description
        coefs: (todo): write your description
        points: (array): write your description
        values: (str): write your description
        Ad: (todo): write your description
        dAd: (todo): write your description
    """

    N = points.shape[0]
    M0 = orders[0]
#    start0 = a[0]
#    dinv0 = (orders[0]-1.0)/(b[0]-a[0])
    M1 = orders[1]
#    start1 = a[1]
#    dinv1 = (orders[1]-1.0)/(b[1]-a[1])
    M2 = orders[2]
#    start2 = a[2]
#    dinv2 = (orders[2]-1.0)/(b[2]-a[2])


    for n in range(N):

        u0 = points[n,0]
        u1 = points[n,1]
        u2 = points[n,2]

        i0 = int( floor( u0 ) )
        i0 = max( min(i0,M0-2), 0 )
        t0 = u0-i0
        i1 = int( floor( u1 ) )
        i1 = max( min(i1,M1-2), 0 )
        t1 = u1-i1
        i2 = int( floor( u2 ) )
        i2 = max( min(i2,M2-2), 0 )
        t2 = u2-i2
        tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
        tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
        tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
        
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


        values[n] = Phi0_0*(Phi1_0*(Phi2_0*(coefs[i0+0,i1+0,i2+0]) + Phi2_1*(coefs[i0+0,i1+0,i2+1]) + Phi2_2*(coefs[i0+0,i1+0,i2+2]) + Phi2_3*(coefs[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+0,i1+1,i2+0]) + Phi2_1*(coefs[i0+0,i1+1,i2+1]) + Phi2_2*(coefs[i0+0,i1+1,i2+2]) + Phi2_3*(coefs[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+0,i1+2,i2+0]) + Phi2_1*(coefs[i0+0,i1+2,i2+1]) + Phi2_2*(coefs[i0+0,i1+2,i2+2]) + Phi2_3*(coefs[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+0,i1+3,i2+0]) + Phi2_1*(coefs[i0+0,i1+3,i2+1]) + Phi2_2*(coefs[i0+0,i1+3,i2+2]) + Phi2_3*(coefs[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(coefs[i0+1,i1+0,i2+0]) + Phi2_1*(coefs[i0+1,i1+0,i2+1]) + Phi2_2*(coefs[i0+1,i1+0,i2+2]) + Phi2_3*(coefs[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+1,i1+1,i2+0]) + Phi2_1*(coefs[i0+1,i1+1,i2+1]) + Phi2_2*(coefs[i0+1,i1+1,i2+2]) + Phi2_3*(coefs[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+1,i1+2,i2+0]) + Phi2_1*(coefs[i0+1,i1+2,i2+1]) + Phi2_2*(coefs[i0+1,i1+2,i2+2]) + Phi2_3*(coefs[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+1,i1+3,i2+0]) + Phi2_1*(coefs[i0+1,i1+3,i2+1]) + Phi2_2*(coefs[i0+1,i1+3,i2+2]) + Phi2_3*(coefs[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(coefs[i0+2,i1+0,i2+0]) + Phi2_1*(coefs[i0+2,i1+0,i2+1]) + Phi2_2*(coefs[i0+2,i1+0,i2+2]) + Phi2_3*(coefs[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+2,i1+1,i2+0]) + Phi2_1*(coefs[i0+2,i1+1,i2+1]) + Phi2_2*(coefs[i0+2,i1+1,i2+2]) + Phi2_3*(coefs[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+2,i1+2,i2+0]) + Phi2_1*(coefs[i0+2,i1+2,i2+1]) + Phi2_2*(coefs[i0+2,i1+2,i2+2]) + Phi2_3*(coefs[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+2,i1+3,i2+0]) + Phi2_1*(coefs[i0+2,i1+3,i2+1]) + Phi2_2*(coefs[i0+2,i1+3,i2+2]) + Phi2_3*(coefs[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(coefs[i0+3,i1+0,i2+0]) + Phi2_1*(coefs[i0+3,i1+0,i2+1]) + Phi2_2*(coefs[i0+3,i1+0,i2+2]) + Phi2_3*(coefs[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+3,i1+1,i2+0]) + Phi2_1*(coefs[i0+3,i1+1,i2+1]) + Phi2_2*(coefs[i0+3,i1+1,i2+2]) + Phi2_3*(coefs[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+3,i1+2,i2+0]) + Phi2_1*(coefs[i0+3,i1+2,i2+1]) + Phi2_2*(coefs[i0+3,i1+2,i2+2]) + Phi2_3*(coefs[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+3,i1+3,i2+0]) + Phi2_1*(coefs[i0+3,i1+3,i2+1]) + Phi2_2*(coefs[i0+3,i1+3,i2+2]) + Phi2_3*(coefs[i0+3,i1+3,i2+3])))



@njit(cache=True)
def kernel(n, a, b, orders, coefs, points, values):
    """
    Create a kernel of a and b

    Args:
        n: (array): write your description
        a: (array): write your description
        b: (array): write your description
        orders: (int): write your description
        coefs: (array): write your description
        points: (array): write your description
        values: (str): write your description
    """

    x0 = points[n,0]
    x1 = points[n,1]
    x2 = points[n,2]

    # common to all units
    M0 = orders[0]
    start0 = a[0]
    dinv0 = (orders[0]-1.0)/(b[0]-a[0])
    M1 = orders[1]
    start1 = a[1]
    dinv1 = (orders[1]-1.0)/(b[1]-a[1])
    M2 = orders[2]
    start2 = a[2]
    dinv2 = (orders[2]-1.0)/(b[2]-a[2])

    # locate the point
    u0 = (x0 - start0)*dinv0
    i0 = int( floor( u0 ) )
    i0 = max( min(i0,M0-2), 0 )
    t0 = u0-i0
    u1 = (x1 - start1)*dinv1
    i1 = int( floor( u1 ) )
    i1 = max( min(i1,M1-2), 0 )
    t1 = u1-i1
    u2 = (x2 - start2)*dinv2
    i2 = int( floor( u2 ) )
    i2 = max( min(i2,M2-2), 0 )
    t2 = u2-i2


    tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
    tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
    tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;
    Phi0_0 = 0
    Phi0_1 = 0
    Phi0_2 = 0
    Phi0_3 = 0
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

    Phi1_0 = 0
    Phi1_1 = 0
    Phi1_2 = 0
    Phi1_3 = 0
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

    Phi2_0 = 0
    Phi2_1 = 0
    Phi2_2 = 0
    Phi2_3 = 0
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


    t = Phi0_0*(Phi1_0*(Phi2_0*(coefs[i0+0,i1+0,i2+0]) + Phi2_1*(coefs[i0+0,i1+0,i2+1]) + Phi2_2*(coefs[i0+0,i1+0,i2+2]) + Phi2_3*(coefs[i0+0,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+0,i1+1,i2+0]) + Phi2_1*(coefs[i0+0,i1+1,i2+1]) + Phi2_2*(coefs[i0+0,i1+1,i2+2]) + Phi2_3*(coefs[i0+0,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+0,i1+2,i2+0]) + Phi2_1*(coefs[i0+0,i1+2,i2+1]) + Phi2_2*(coefs[i0+0,i1+2,i2+2]) + Phi2_3*(coefs[i0+0,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+0,i1+3,i2+0]) + Phi2_1*(coefs[i0+0,i1+3,i2+1]) + Phi2_2*(coefs[i0+0,i1+3,i2+2]) + Phi2_3*(coefs[i0+0,i1+3,i2+3]))) + Phi0_1*(Phi1_0*(Phi2_0*(coefs[i0+1,i1+0,i2+0]) + Phi2_1*(coefs[i0+1,i1+0,i2+1]) + Phi2_2*(coefs[i0+1,i1+0,i2+2]) + Phi2_3*(coefs[i0+1,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+1,i1+1,i2+0]) + Phi2_1*(coefs[i0+1,i1+1,i2+1]) + Phi2_2*(coefs[i0+1,i1+1,i2+2]) + Phi2_3*(coefs[i0+1,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+1,i1+2,i2+0]) + Phi2_1*(coefs[i0+1,i1+2,i2+1]) + Phi2_2*(coefs[i0+1,i1+2,i2+2]) + Phi2_3*(coefs[i0+1,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+1,i1+3,i2+0]) + Phi2_1*(coefs[i0+1,i1+3,i2+1]) + Phi2_2*(coefs[i0+1,i1+3,i2+2]) + Phi2_3*(coefs[i0+1,i1+3,i2+3]))) + Phi0_2*(Phi1_0*(Phi2_0*(coefs[i0+2,i1+0,i2+0]) + Phi2_1*(coefs[i0+2,i1+0,i2+1]) + Phi2_2*(coefs[i0+2,i1+0,i2+2]) + Phi2_3*(coefs[i0+2,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+2,i1+1,i2+0]) + Phi2_1*(coefs[i0+2,i1+1,i2+1]) + Phi2_2*(coefs[i0+2,i1+1,i2+2]) + Phi2_3*(coefs[i0+2,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+2,i1+2,i2+0]) + Phi2_1*(coefs[i0+2,i1+2,i2+1]) + Phi2_2*(coefs[i0+2,i1+2,i2+2]) + Phi2_3*(coefs[i0+2,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+2,i1+3,i2+0]) + Phi2_1*(coefs[i0+2,i1+3,i2+1]) + Phi2_2*(coefs[i0+2,i1+3,i2+2]) + Phi2_3*(coefs[i0+2,i1+3,i2+3]))) + Phi0_3*(Phi1_0*(Phi2_0*(coefs[i0+3,i1+0,i2+0]) + Phi2_1*(coefs[i0+3,i1+0,i2+1]) + Phi2_2*(coefs[i0+3,i1+0,i2+2]) + Phi2_3*(coefs[i0+3,i1+0,i2+3])) + Phi1_1*(Phi2_0*(coefs[i0+3,i1+1,i2+0]) + Phi2_1*(coefs[i0+3,i1+1,i2+1]) + Phi2_2*(coefs[i0+3,i1+1,i2+2]) + Phi2_3*(coefs[i0+3,i1+1,i2+3])) + Phi1_2*(Phi2_0*(coefs[i0+3,i1+2,i2+0]) + Phi2_1*(coefs[i0+3,i1+2,i2+1]) + Phi2_2*(coefs[i0+3,i1+2,i2+2]) + Phi2_3*(coefs[i0+3,i1+2,i2+3])) + Phi1_3*(Phi2_0*(coefs[i0+3,i1+3,i2+0]) + Phi2_1*(coefs[i0+3,i1+3,i2+1]) + Phi2_2*(coefs[i0+3,i1+3,i2+2]) + Phi2_3*(coefs[i0+3,i1+3,i2+3])))

    values[n] = t

@njit(cache=True)
def vec_eval_cubic_spline_3_kernel(a, b, orders, coefs, points, values):
    """
    Evaluates_eval_kernel_3_3 ).

    Args:
        a: (todo): write your description
        b: (todo): write your description
        orders: (int): write your description
        coefs: (todo): write your description
        points: (array): write your description
        values: (str): write your description
    """

    N = points.shape[0]
    for n in range(N):
        kernel(n, a, b, orders, coefs, points, values)

if __name__ == '__main__':
    import numpy as np
    d = 3
    K = 50
    N = 10**6
    a = np.zeros(3)
    b = np.ones(3)
    orders = np.array([K for i in range(d)])
    coeffs = np.random.random([k+2 for k in orders])
    points = np.random.random((N,d))  # each line is a vector
    points_c = points.T.copy() # each column is a vector
    vals = np.zeros(N)

    print("Interpolation comparison (d={}, K={}, N={})" .format(d,K,N))


    import time

    vec_eval_cubic_spline_3        (a,b,orders,coeffs,points,vals)  # warmup
    vec_eval_cubic_spline_3_inlined(a,b,orders,coeffs,points,vals)  # warmup
    vec_eval_cubic_spline_3_inlined_columns(a,b,orders,coeffs,points_c,vals)  # warmup
    vec_eval_cubic_spline_3_kernel (a,b,orders,coeffs,points,vals)  # warmup
    vec_eval_cubic_spline_3_inlined_lesswork(orders,coeffs,points,vals,Ad,dAd)

    t1 = time.time()
    vec_eval_cubic_spline_3(a,b,orders,coeffs,points,vals)
    t2 = time.time()
    vec_eval_cubic_spline_3_inlined(a,b,orders,coeffs,points,vals)
    t3 = time.time()
    vec_eval_cubic_spline_3_inlined_columns(a,b,orders,coeffs,points_c,vals)
    t4 = time.time()
    vec_eval_cubic_spline_3_kernel(a,b,orders,coeffs,points,vals)
    t5 = time.time()
    vec_eval_cubic_spline_3_inlined_lesswork(orders,coeffs,points,vals,Ad,dAd)
    t6 = time.time()
    print("one function call per point: {}".format(t2-t1))
    print("inlined (points in rows): {}".format(t3-t2))
    print("inlined (points in columns): {}".format(t4-t3))
    print("kernel: {}".format(t5-t4))
    print("less work: {}".format(t6-t5))


