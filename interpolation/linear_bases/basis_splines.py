from interpolation.linear_bases.basis import LinearBasis
from interpolation.linear_bases.util import lookup
import scipy.sparse as spa
import numpy as np


class SplineBasis(LinearBasis):

    def __init__(self, nodes, evennum=0, k=3):

        breaks = nodes

        if k <= 0:
            raise ValueError("spline order must be positive")

        # check to see if breaks is a list or tuple
        if isinstance(breaks, (list, tuple)):
            if len(breaks) == 3:
                # assume it is of the form (lb, ub, n)
                breaks = np.linspace(*breaks)
            else:
                # assume the breaks are given directly.
                breaks = np.asarray(breaks)

        if isinstance(breaks, np.ndarray):
            if not all(breaks == np.sort(breaks)):
                raise ValueError("Breaks must be sorted")

            if breaks.size < 2:
                raise ValueError("must have at least 2 breakpoints")

            if evennum == 0:
                if breaks.size == 2:
                    evennum = 2

            else:
                if breaks.size == 2:
                    breaks = np.linspace(breaks[0], breaks[2], evennum)
                else:
                    m = "Breakpoint sequence must contain 2 values when evennum > 0"
                    raise ValueError(m)

            # self.nodes = breaks
            self.nodes = breaks
            self.evennum = evennum
            self.k = k
            self.m = len(breaks) + k - 1
            self.n = len(breaks)


        else:
            raise ValueError("Couldn't interpret input form of breaks")

    @property
    def knots(self):
        nodes, k = self.nodes, self.k
        a = nodes[0]  # 20
        b = nodes[-1]  # 21
        n = self.m
        x = np.cumsum(np.concatenate((np.full(k, a), nodes, np.full(k, b))))  # 23
        x = (x[k:n+k] - x[:n]) / k  # 24
        x[0] = a  # 25
        x[-1] = b  # 26
        return x

    @property
    def B(self):
        raise Exception("Not implemented")

    def derivative_op(self, order=1):
        raise NotImplementedError()

    def Phi(self, x, orders=None):

        x = np.atleast_1d(x)

        if isinstance(orders, list):
            return np.concatenate([self.eval(x, orders=o)[...,None] for o in orders],axis=-1)

        if orders is None:
            orders=0

        if orders>1:
            raise Exception("Not implemented")

        order = orders

        breaks, evennum, k = self.nodes, self.evennum, self.k
        order = np.atleast_1d(order)

        # error handling
        if any(order >= k):
            raise ValueError("Order of diff must be < k")

        if x.ndim > 1:
            if any(np.asarray(x.shape[1:]) > 1):
                raise ValueError("x must be a vector")
            else:
                # flatten
                x = x.reshape(-1)

        m = len(x)
        minorder = order.min()

        # Augment the breakpoint sequence 57-59
        n = self.m
        a = breaks[0]
        b = breaks[-1]
        augbreaks = np.concatenate((np.full(k-minorder, a),
                                    breaks,
                                    np.full(k-minorder, b)))

        ind = lookup(augbreaks, x, 3)  # 69

        bas = np.zeros((m, k - minorder + 1))
        bas[:, 0] = 1.0

        B = []

        if order.max() > 0:
            D = self.derivative_op(order.max())

        if minorder < 0:
            I = self.derivative_op(minorder)

        for j in range(1, k-minorder+1):
            for jj in range(j, 0, -1):
                b0 = augbreaks[ind+jj-j]
                b1 = augbreaks[ind+jj]
                temp = bas[:, jj-1] / (b1 - b0)
                bas[:, jj] = (x-b0) * temp + bas[:, jj]
                bas[:, jj-1] = (b1-x) * temp

            # bas now contains the `j` order spline basis
            ii = np.argwhere(order == k-j)
            if len(ii) > 0:
                ii = ii[0][0]

                # put the values into appropriate rows/columns of sparse
                # matrix
                r = np.tile(np.arange(m), k-order[ii]+1)
                c = np.arange(-k+order[ii], 1) - (order[ii] - minorder)
                c = (c[:, None] + ind[None]).reshape(-1)
                v = np.ravel(bas[:, :k-order[ii]+1], order="F")
                B_ii = spa.coo_matrix((v, (r, c)), shape=(m, n-order[ii]))

                if order[ii] > 0:
                    B.append(B_ii.dot(D[order[ii]]))
                elif order[ii] < 0:
                    B.append(B_ii.dot(I[order[ii]]))
                else:
                    B.append(B_ii)

        if isinstance(orders, list):
            # return B
            return np.concatenate([b[...,None] for b in B], axis=-1)
        else:
            return np.array(B[0].todense())

    def filter(self,x):
        from numpy.linalg import solve
        Phi = self.eval(self.knots)
        xx = np.concatenate([[1],x,[1]])
        return solve(Phi, xx)


class UniformSplineBasis(SplineBasis):

    def __init__(self, a, b, num, evennum=0, k=3):

        nodes = np.linspace(a, b, num)
        super().__init__(nodes, evennum=evennum, k=k)
