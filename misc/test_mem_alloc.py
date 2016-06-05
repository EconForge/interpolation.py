from interpolation.splines.eval_cubic import vec_eval_cubic_spline_3, Ad, dAd

vec_eval_cubic_spline_3
d = 3
import numpy
a = numpy.array([0.0,0.0,0.0])
b = numpy.array([1.0,1.0,1.0])
orders = numpy.array([30,30,30])

coefs = numpy.random.random(orders+2)

N = 100000

points = numpy.random.random((N,d))
out = numpy.random.random(N)
vec_eval_cubic_spline_3(a,b,orders,coefs,points,out)


from  numba import njit
from numpy import zeros
from math import floor

@njit
def reduce_tensor(Phi,C):
    return Phi[0,0]*(Phi[1,0]*(Phi[2,0]*(C[0,0,0]) + Phi[2,1]*(C[0,0,1]) + Phi[2,2]*(C[0,0,2]) + Phi[2,3]*(C[0,0,3])) + Phi[1,1]*(Phi[2,0]*(C[0,1,0]) + Phi[2,1]*(C[0,1,1]) + Phi[2,2]*(C[0,1,2]) + Phi[2,3]*(C[0,1,3])) + Phi[1,2]*(Phi[2,0]*(C[0,2,0]) + Phi[2,1]*(C[0,2,1]) + Phi[2,2]*(C[0,2,2]) + Phi[2,3]*(C[0,2,3])) + Phi[1,3]*(Phi[2,0]*(C[0,3,0]) + Phi[2,1]*(C[0,3,1]) + Phi[2,2]*(C[0,3,2]) + Phi[2,3]*(C[0,3,3]))) + Phi[0,1]*(Phi[1,0]*(Phi[2,0]*(C[1,0,0]) + Phi[2,1]*(C[1,0,1]) + Phi[2,2]*(C[1,0,2]) + Phi[2,3]*(C[1,0,3])) + Phi[1,1]*(Phi[2,0]*(C[1,1,0]) + Phi[2,1]*(C[1,1,1]) + Phi[2,2]*(C[1,1,2]) + Phi[2,3]*(C[1,1,3])) + Phi[1,2]*(Phi[2,0]*(C[1,2,0]) + Phi[2,1]*(C[1,2,1]) + Phi[2,2]*(C[1,2,2]) + Phi[2,3]*(C[1,2,3])) + Phi[1,3]*(Phi[2,0]*(C[1,3,0]) + Phi[2,1]*(C[1,3,1]) + Phi[2,2]*(C[1,3,2]) + Phi[2,3]*(C[1,3,3]))) + Phi[0,2]*(Phi[1,0]*(Phi[2,0]*(C[2,0,0]) + Phi[2,1]*(C[2,0,1]) + Phi[2,2]*(C[2,0,2]) + Phi[2,3]*(C[2,0,3])) + Phi[1,1]*(Phi[2,0]*(C[2,1,0]) + Phi[2,1]*(C[2,1,1]) + Phi[2,2]*(C[2,1,2]) + Phi[2,3]*(C[2,1,3])) + Phi[1,2]*(Phi[2,0]*(C[2,2,0]) + Phi[2,1]*(C[2,2,1]) + Phi[2,2]*(C[2,2,2]) + Phi[2,3]*(C[2,2,3])) + Phi[1,3]*(Phi[2,0]*(C[2,3,0]) + Phi[2,1]*(C[2,3,1]) + Phi[2,2]*(C[2,3,2]) + Phi[2,3]*(C[2,3,3]))) + Phi[0,3]*(Phi[1,0]*(Phi[2,0]*(C[3,0,0]) + Phi[2,1]*(C[3,0,1]) + Phi[2,2]*(C[3,0,2]) + Phi[2,3]*(C[3,0,3])) + Phi[1,1]*(Phi[2,0]*(C[3,1,0]) + Phi[2,1]*(C[3,1,1]) + Phi[2,2]*(C[3,1,2]) + Phi[2,3]*(C[3,1,3])) + Phi[1,2]*(Phi[2,0]*(C[3,2,0]) + Phi[2,1]*(C[3,2,1]) + Phi[2,2]*(C[3,2,2]) + Phi[2,3]*(C[3,2,3])) + Phi[1,3]*(Phi[2,0]*(C[3,3,0]) + Phi[2,1]*(C[3,3,1]) + Phi[2,2]*(C[3,3,2]) + Phi[2,3]*(C[3,3,3])))

@njit
def get_data(coefs,inds,C):
    i0 = inds[0]
    i1 = inds[1]
    i2 = inds[2]
    C[0,0,0] = coefs[i0+0,i1+0,i2+0]
    C[0,0,1] = coefs[i0+0,i1+0,i2+1]
    C[0,0,2] = coefs[i0+0,i1+0,i2+2]
    C[0,0,3] = coefs[i0+0,i1+0,i2+3]
    C[0,1,0] = coefs[i0+0,i1+1,i2+0]
    C[0,1,1] = coefs[i0+0,i1+1,i2+1]
    C[0,1,2] = coefs[i0+0,i1+1,i2+2]
    C[0,1,3] = coefs[i0+0,i1+1,i2+3]
    C[0,2,0] = coefs[i0+0,i1+2,i2+0]
    C[0,2,1] = coefs[i0+0,i1+2,i2+1]
    C[0,2,2] = coefs[i0+0,i1+2,i2+2]
    C[0,2,3] = coefs[i0+0,i1+2,i2+3]
    C[0,3,0] = coefs[i0+0,i1+3,i2+0]
    C[0,3,1] = coefs[i0+0,i1+3,i2+1]
    C[0,3,2] = coefs[i0+0,i1+3,i2+2]
    C[0,3,3] = coefs[i0+0,i1+3,i2+3]
    C[1,0,0] = coefs[i0+0,i1+0,i2+0]
    C[1,0,1] = coefs[i0+0,i1+0,i2+1]
    C[1,0,2] = coefs[i0+0,i1+0,i2+2]
    C[1,0,3] = coefs[i0+0,i1+0,i2+3]
    C[1,1,0] = coefs[i0+0,i1+1,i2+0]
    C[1,1,1] = coefs[i0+0,i1+1,i2+1]
    C[1,1,2] = coefs[i0+0,i1+1,i2+2]
    C[1,1,3] = coefs[i0+0,i1+1,i2+3]
    C[1,2,0] = coefs[i0+0,i1+2,i2+0]
    C[1,2,1] = coefs[i0+0,i1+2,i2+1]
    C[1,2,2] = coefs[i0+0,i1+2,i2+2]
    C[1,2,3] = coefs[i0+0,i1+2,i2+3]
    C[1,3,0] = coefs[i0+0,i1+3,i2+0]
    C[1,3,1] = coefs[i0+0,i1+3,i2+1]
    C[1,3,2] = coefs[i0+0,i1+3,i2+2]
    C[1,3,3] = coefs[i0+0,i1+3,i2+3]
    C[2,0,0] = coefs[i0+0,i1+0,i2+0]
    C[2,0,1] = coefs[i0+0,i1+0,i2+1]
    C[2,0,2] = coefs[i0+0,i1+0,i2+2]
    C[2,0,3] = coefs[i0+0,i1+0,i2+3]
    C[2,1,0] = coefs[i0+0,i1+1,i2+0]
    C[2,1,1] = coefs[i0+0,i1+1,i2+1]
    C[2,1,2] = coefs[i0+0,i1+1,i2+2]
    C[2,1,3] = coefs[i0+0,i1+1,i2+3]
    C[2,2,0] = coefs[i0+0,i1+2,i2+0]
    C[2,2,1] = coefs[i0+0,i1+2,i2+1]
    C[2,2,2] = coefs[i0+0,i1+2,i2+2]
    C[2,2,3] = coefs[i0+0,i1+2,i2+3]
    C[2,3,0] = coefs[i0+0,i1+3,i2+0]
    C[2,3,1] = coefs[i0+0,i1+3,i2+1]
    C[2,3,2] = coefs[i0+0,i1+3,i2+2]
    C[2,3,3] = coefs[i0+0,i1+3,i2+3]
    C[3,0,0] = coefs[i0+0,i1+0,i2+0]
    C[3,0,1] = coefs[i0+0,i1+0,i2+1]
    C[3,0,2] = coefs[i0+0,i1+0,i2+2]
    C[3,0,3] = coefs[i0+0,i1+0,i2+3]
    C[3,1,0] = coefs[i0+0,i1+1,i2+0]
    C[3,1,1] = coefs[i0+0,i1+1,i2+1]
    C[3,1,2] = coefs[i0+0,i1+1,i2+2]
    C[3,1,3] = coefs[i0+0,i1+1,i2+3]
    C[3,2,0] = coefs[i0+0,i1+2,i2+0]
    C[3,2,1] = coefs[i0+0,i1+2,i2+1]
    C[3,2,2] = coefs[i0+0,i1+2,i2+2]
    C[3,2,3] = coefs[i0+0,i1+2,i2+3]
    C[3,3,0] = coefs[i0+0,i1+3,i2+0]
    C[3,3,1] = coefs[i0+0,i1+3,i2+1]
    C[3,3,2] = coefs[i0+0,i1+3,i2+2]
    C[3,3,3] = coefs[i0+0,i1+3,i2+3]

from numba import int8

@njit
def construct_Phi(loc, Phi):

    t0 = loc[0]
    t1 = loc[1]
    t2 = loc[2]

    tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
    tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;
    tp2_0 = t2*t2*t2;  tp2_1 = t2*t2;  tp2_2 = t2;  tp2_3 = 1.0;

    Phi[0,0] = 0
    Phi[0,1] = 0
    Phi[0,2] = 0
    Phi[0,3] = 0
    if t0 < 0:
        Phi[0,0] = dAd[0,3]*t0 + Ad[0,3]
        Phi[0,1] = dAd[1,3]*t0 + Ad[1,3]
        Phi[0,2] = dAd[2,3]*t0 + Ad[2,3]
        Phi[0,3] = dAd[3,3]*t0 + Ad[3,3]
    elif t0 > 1:
        Phi[0,0] = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
        Phi[0,1] = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
        Phi[0,2] = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
        Phi[0,3] = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
    else:
        Phi[0,0] = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
        Phi[0,1] = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
        Phi[0,2] = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
        Phi[0,3] = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

    Phi[1,0] = 0
    Phi[1,1] = 0
    Phi[1,2] = 0
    Phi[1,3] = 0
    if t1 < 0:
        Phi[1,0] = dAd[0,3]*t1 + Ad[0,3]
        Phi[1,1] = dAd[1,3]*t1 + Ad[1,3]
        Phi[1,2] = dAd[2,3]*t1 + Ad[2,3]
        Phi[1,3] = dAd[3,3]*t1 + Ad[3,3]
    elif t1 > 1:
        Phi[1,0] = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t1-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
        Phi[1,1] = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t1-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
        Phi[1,2] = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t1-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
        Phi[1,3] = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t1-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
    else:
        Phi[1,0] = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
        Phi[1,1] = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
        Phi[1,2] = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
        Phi[1,3] = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)

    Phi[2,0] = 0
    Phi[2,1] = 0
    Phi[2,2] = 0
    Phi[2,3] = 0
    if t2 < 0:
        Phi[2,0] = dAd[0,3]*t2 + Ad[0,3]
        Phi[2,1] = dAd[1,3]*t2 + Ad[1,3]
        Phi[2,2] = dAd[2,3]*t2 + Ad[2,3]
        Phi[2,3] = dAd[3,3]*t2 + Ad[3,3]
    elif t2 > 1:
        Phi[2,0] = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t2-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
        Phi[2,1] = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t2-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
        Phi[2,2] = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t2-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
        Phi[2,3] = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t2-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
    else:
        Phi[2,0] = (Ad[0,0]*tp2_0 + Ad[0,1]*tp2_1 + Ad[0,2]*tp2_2 + Ad[0,3]*tp2_3)
        Phi[2,1] = (Ad[1,0]*tp2_0 + Ad[1,1]*tp2_1 + Ad[1,2]*tp2_2 + Ad[1,3]*tp2_3)
        Phi[2,2] = (Ad[2,0]*tp2_0 + Ad[2,1]*tp2_1 + Ad[2,2]*tp2_2 + Ad[2,3]*tp2_3)
        Phi[2,3] = (Ad[3,0]*tp2_0 + Ad[3,1]*tp2_1 + Ad[3,2]*tp2_2 + Ad[3,3]*tp2_3)

@njit
def vec_eval_cubic_spline_3_new(a, b, orders, coefs, points,out):

    Phi = zeros((4,4))
    C = zeros((4,4,4))
    inds = zeros(3, dtype=int32)
    loc = zeros(3)

    M0 = orders[0]
    start0 = a[0]
    dinv0 = (orders[0]-1.0)/(b[0]-a[0])
    M1 = orders[1]
    start1 = a[1]
    dinv1 = (orders[1]-1.0)/(b[1]-a[1])
    M2 = orders[2]
    start2 = a[2]
    dinv2 = (orders[2]-1.0)/(b[2]-a[2])
    N = points.shape[0]

    for n in range(N):
        x0 = points[n,0]
        x1 = points[n,1]
        x2 = points[n,2]
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

        inds[0] = i0
        inds[1] = i1
        loc[1] = t1
        loc[0] = t0
        inds[2] = i2
        loc[2] = t2
        #
        construct_Phi(loc, Phi)
        get_data(coefs, inds, C)
        t = reduce_tensor(Phi, C)

        out[n] = t


vec_eval_cubic_spline_3_new(a,b,orders,coefs,points,out)

def test_1():
    for i in range(100):
        vec_eval_cubic_spline_3(a,b,orders,coefs,points,out)
    return out

def test_2():
    for i in range(100):
        vec_eval_cubic_spline_3_new(a,b,orders,coefs,points,out)
    return out



%time test_2()
%time test_1()
