import numpy as np
from interpolation.linear_bases.basis import LinearBasis, CompactLinearBasis

import numpy as np
from numpy import zeros, array, zeros


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

from interpolation.linear_bases.compact_matrices import CompactBasisMatrix, CompactBasisArray

class UniformSplineBasis(CompactLinearBasis):

    def __init__(self, a, b, num, k=3):

        self.nodes = np.linspace(a,b,num)
        dx =  self.nodes[1]-self.nodes[0]
        self.knots = np.concatenate([[a-dx], self.nodes, [b+dx]])
        self.n = num
        self.min = a
        self.max = b

        if k != 3:
            raise Exception("Not implemented")

    def eval(self, x, orders=None):

        if orders is None:
            orders = 0
        elif not isinstance(orders, int):
            l = [self.eval(x, orders=o) for o in orders]
            return tuple(l)

        m = self.n

        M0 = m
        start0 = self.min
        dinv0 = (self.n-1.0)/(self.max-self.min)

        # x0 = point[0]
        x0 = x  ###
        u0 = (x0 - start0)*dinv0
        i0 = np.array( np.floor( u0 ), dtype=int )
        i0 = np.maximum( np.minimum(i0,M0-2), 0 )
        t0 = u0-i0
        tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
        def cat(*l):
            nd = l[0].ndim
            return np.concatenate([e[...,None] for e in l], axis=nd)
        if orders==0:
            Phi0_0 = (dAd[0,3]*t0 + Ad[0,3])*(t0<0) + ((3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3]))*(t0>1) + (t0>=0)*(t0<=1)*((Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3))
            Phi0_1 = (dAd[1,3]*t0 + Ad[1,3])*(t0<0) + ((3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3]))*(t0>1) + (t0>=0)*(t0<=1)*((Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3))
            Phi0_2 = (dAd[2,3]*t0 + Ad[2,3])*(t0<0) + ((3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3]))*(t0>1) + (t0>=0)*(t0<=1)*((Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3))
            Phi0_3 = (dAd[3,3]*t0 + Ad[3,3])*(t0<0) + ((3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3]))*(t0>1) + (t0>=0)*(t0<=1)*((Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3))
            return CompactBasisMatrix(i0, cat(Phi0_0, Phi0_1, Phi0_2, Phi0_3), self.n+2)
        elif orders==1:
            dPhi0_0 = (dAd[0,0]*tp0_0 + dAd[0,1]*tp0_1 + dAd[0,2]*tp0_2 + dAd[0,3]*tp0_3)*dinv0
            dPhi0_1 = (dAd[1,0]*tp0_0 + dAd[1,1]*tp0_1 + dAd[1,2]*tp0_2 + dAd[1,3]*tp0_3)*dinv0
            dPhi0_2 = (dAd[2,0]*tp0_0 + dAd[2,1]*tp0_1 + dAd[2,2]*tp0_2 + dAd[2,3]*tp0_3)*dinv0
            dPhi0_3 = (dAd[3,0]*tp0_0 + dAd[3,1]*tp0_1 + dAd[3,2]*tp0_2 + dAd[3,3]*tp0_3)*dinv0
            return CompactBasisMatrix(i0, cat(dPhi0_0, dPhi0_1, dPhi0_2, dPhi0_3), self.n+2)
        elif orders==2:
            d2Phi0_0 = (d2Ad[0,0]*tp0_0 + d2Ad[0,1]*tp0_1 + d2Ad[0,2]*tp0_2 + d2Ad[0,3]*tp0_3)*dinv0**2
            d2Phi0_1 = (d2Ad[1,0]*tp0_0 + d2Ad[1,1]*tp0_1 + d2Ad[1,2]*tp0_2 + d2Ad[1,3]*tp0_3)*dinv0**2
            d2Phi0_2 = (d2Ad[2,0]*tp0_0 + d2Ad[2,1]*tp0_1 + d2Ad[2,2]*tp0_2 + d2Ad[2,3]*tp0_3)*dinv0**2
            d2Phi0_3 = (d2Ad[3,0]*tp0_0 + d2Ad[3,1]*tp0_1 + d2Ad[3,2]*tp0_2 + d2Ad[3,3]*tp0_3)*dinv0**2
            return CompactBasisMatrix(i0, cat(d2Phi0_0, d2Phi0_1, d2Phi0_2, d2Phi0_3), self.n+2)
        else:
            raise Exception("Not implemented")

    def filter(self, x):

        from interpolation.splines.filter_cubic import find_coefs_1d

        dx =  self.nodes[1]-self.nodes[0]
        dinv = 1/dx
        C = np.zeros(self.n+2)
        find_coefs_1d(dinv, self.n, x, C)

        return C
