import numpy as np
from interpolation.linear_bases.util import lookup
from interpolation.linear_bases.basis import LinearBasis

import scipy.sparse as spa

class LinearSplineBasis(LinearBasis):

    def __init__(self, nodes):

        nodes = np.asarray(nodes)
        n = len(nodes)
        self.nodes = nodes
        self.min = nodes.min()
        self.max = nodes.max()
        self.n = n
        self.m = n

    def __call__(self, *args, **kwargs):
        return self.Phi(*args, **kwargs)

    def Phi(self, x, orders=None):

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


class UniformLinearSplineBasis(LinearSplineBasis):

    def __init__(self, *args, **kwargs):

        nodes = np.linspace(*args, **kwargs)
        n = len(nodes)
        self.nodes = nodes
        self.min = nodes.min()
        self.max = nodes.max()
        self.n = n
        self.m = n
