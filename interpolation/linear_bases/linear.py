import numpy as np
# from interpolation.splines.util import lookup
from interpolation.linear_bases.util import lookup
from interpolation.linear_bases.basis import LinearBasis

import scipy.sparse as spa

class UniformLinearSpline:

    def __init__(self, min=0.0, max=1.0, n=10):
        self.nodes = np.linspace(min,max,n)
        self.min = min
        self.max = max
        self.n = n
        self.m = n

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def eval(self, x, orders=None):

        x = np.array(x)

        if x.ndim == 0:
            return self.eval(x[None], orders=orders)[0,:]

        if orders is None:
            orders = 0

        N = x.shape[0]
        m = self.m

        ind = lookup(self.nodes, x, 3)

        z = np.empty(N)
        if orders == 0:
            for i in range(N):
                ii = ind[i]
                z[i] = ((x[i] - self.nodes[ii]) / (self.nodes[ii + 1] - self.nodes[ii]))
        elif orders == 1:
            for i in range(N):
                ii = ind[i]
                z[i] = (1 / (self.nodes[ii + 1] - self.nodes[ii]))
        else:
            pass # matrix is empty


        rows = np.arange(N)
        i = np.concatenate((rows, rows))
        j = np.concatenate((ind, ind+1))
        v = np.concatenate((1-z, z))
        mat = spa.coo_matrix((v, (i, j)), shape=(N, m))

        return mat.todense().A



    def filter(self,x):
        return x

us = UniformLinearSpline(0,1,10)

x = np.linspace(-0.2,1.2, 100)



us.eval(0.1)

us.eval([0.1,0.2])

mat = us.eval(x)

mat

# from matplotlib import pyplot as plt
# %matplotlib inline
#
# for i in range(10):
#     plt.plot(x, dmat[:,i])
