import numpy.linalg
from typing import List

import numpy as np

class LinearBasis:
    pass

class ChebychevBasis(LinearBasis):

    def __init__(self, min=0, max=1.0, n=10):
        self.min = min
        self.max = max
        self.n = n      # number of nodes
        self.m = n      # number of basis functions
        self.Phi = self.eval(self.nodes)

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def __str__(self):
        return "ChebychevBasis(min={}, max={}, n={})".format(self.min, self.max, self.n)

    def eval(self, x, orders=None):

        m = self.m
        x = np.asarray(x)
        if x.ndim == 0:
            return self.eval(x[None], x, orders=orders)[0, ...]
        N = x.shape[0]
        # rescale x
        x = -1 + (x-self.min)/(self.max-self.min)*2
        if orders is None:
            orders = 0
        if orders == 0:
            return chebychev(x,m)
        elif orders == 1:
            # should be replaced by direct call to d_chebychev
            U = chebychev(x,m,kind=2.0)
            dT = np.zeros((N,m))
            dT[:,1:] = U[:,:-1]
            for t in range(dT.shape[-1]):
                dT[:,t] *= t
            # scale
            dT = dT/(self.max-self.min)*2
            return dT
        elif isinstance(orders, int) and orders >= 2:
            raise Exception("Not implemented")
        else:
            res = [chebbase(x,m, orders=o) for o in orders]
            res = np.concatenate([r[:,None] for r in res], axis=-1)
            return res

    @property
    def nodes(self):
        m = self.m
        comp_vals = np.arange(1., m + 1.)
        vals = -1. * np.cos(np.pi*(comp_vals - 1.)/(m-1.))
        vals[np.where(np.abs(vals) < 1e-14)] = 0.0
        vals = self.min + (vals+1)/2*(self.max-self.min)
        return vals

    def filter(self,x):

        from numpy.linalg import solve

        x = np.asarray(x)
        return solve(self.Phi, x)


def chebbase(x, m, orders=None):

    N = x.shape[0]

    if orders is None:
        orders = 0

    if orders == 0:
        return chebychev(x,m)

    elif orders == 1:
        # should be replaced by direct call to d_chebychev
        U = chebychev(x,m,kind=2.0)
        dT = np.zeros((N,m))
        dT[:,1:] = U[:,:-1]
        for t in range(dT.shape[-1]):
            dT[:,t] *= t
        return dT
    else:
        res = [chebbase(x,m, orders=o) for o in orders]
        res = np.concatenate([r[:,None] for r in res], axis=-1)
        return res


def chebychev(x,m,kind=1.0):
    N = x.shape[0]
    if isinstance(x, (float, int)):
        T = np.zeros(m)
    else:
        assert(x.ndim==1)
    T = np.zeros((N,m))
    T[...,0] = 1
    T[...,1] = kind*x
    for i in range(2,m):
        T[...,i] = 2*x*T[...,i-1] - T[...,i-2]
    return T

def chebychev2(x,m):
    N = x.shape[0]
    if isinstance(x, (float, int)):
        T = np.zeros(m)
    else:
        assert(x.ndim==1)
    T = np.zeros((N,m))
    T[...,0] = 1
    T[...,1] = 2*x
    for i in range(2,m):
        T[...,i] = 2*x*T[...,i-1] - T[...,i-2]
    return T



if __name__ == '__main__':

    def fun(x): return np.sin(x**2/5) + 2*x
    def dfun(x): return np.cos(x**2/5)*2*x/5 + 2

    xvec = np.linspace(-0.5, 5,200)
    yvec = fun(xvec)



    cb = ChebychevBasis(min=-0.5, max=5, n=5)

    coeffs = cb.filter(fun(cb.nodes))

    from matplotlib import pyplot as plt

    plt.figure()
    plt.subplot(121)
    Phi = cb.eval(xvec)
    ivec = Phi @ coeffs
    plt.plot(xvec, yvec)
    plt.plot(xvec,ivec)
    plt.plot(cb.nodes, fun(cb.nodes),'x')


    plt.subplot(122)
    d_yvec = dfun(xvec)
    d_Phi = cb.eval(xvec, orders=1)
    d_ivec = d_Phi @ coeffs
    plt.plot(xvec, d_yvec)
    plt.plot(xvec, d_ivec)
    plt.plot(cb.nodes, dfun(cb.nodes),'x')



    ff = np.concatenate( [fun(cb.nodes)[:,None] for i in range(2)], axis=1 )

    ff.shape
    coeffs = cb.filter(ff)
    coeffs.shape
    plt.show()
