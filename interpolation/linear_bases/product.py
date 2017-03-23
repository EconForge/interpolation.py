from dolo import dprint

import numpy.linalg
import numpy as np

from interpolation.cartesian import cartesian
from interpolation.linear_bases.basis import LinearBasis, CompactLinearBasis

from interpolation.linear_bases.basis_chebychev import ChebychevBasis
from interpolation.linear_bases.compact_matrices import CompactBasisMatrix, CompactBasisArray, CompactKroneckerProduct

from functools import reduce
import operator


class BasisMatrix:
    pass

class BasisArray:
    pass

class KroneckerProduct:

    def __init__(self, arrays):

        # right now, arrays is a list of vectors or matrices
        # TODO: allow for 3d-array
        self.arrays = [np.asarray(a) for a in arrays]
        self.d = len(arrays)
        self.tensor_type = self.arrays[0].ndim
        # 1: tensor product of vectors
        # 2: vectorized tensor product
        # 3: vectorized tensor product with derivative informations

    def as_matrix(self):
        if self.tensor_type < 3:
            return gu_tensor_product(self.arrays)
        else:
            raise Exception('Not supported. Use as_array instead.')

    def as_array(self, enum='complete'):
        if enum == 'complete':
            # TODO: check that all arrays have the same first and last dimensions
            rr = np.arange(self.arrays[0].shape[2])
            enum = [tuple(el.astype(int).tolist()) for el in cartesian([rr]*self.d, order='F') if sum(el)<=1]
        last_dim = len(enum)
        N = self.arrays[0].shape[0]
        K = reduce(operator.mul, [a.shape[1] for a in self.arrays])
        res = np.zeros((N, K, last_dim))
        for k in range(last_dim):
            arrs = [a[:, :, 0] for a in self.arrays]
            rhs = gu_tensor_product(arrs)
            res[:, :, k] = rhs
        return res

    def __mul__(self, c):
        c = np.asarray(c)
        # placeholder algo
        if self.tensor_type <3:
            mat = self.as_matrix()
            res = mat @ c.reshape((mat.shape[1],-1))
            if c.ndim == len(self.arrays):
                res = res.ravel()
        else:
            mat = self.as_array()
            tmat = mat.swapaxes(1,2)

            res = np.dot( tmat , c.reshape((mat.shape[1],-1)) )
            if c.ndim == 1:
                res = res[:,:,0]
            elif c.ndim==2:
                res = res.swapaxes(1,2)
        return res

def gu_tensor_product(arrays):
    # this is actually a "kronecker" product
    # placeholder algo
    d = len(arrays)
    if d == 1:
        return arrays[0]
    tensor_type = arrays[0].ndim
    c = tensor_type - 1
    # enum_res = (np.expand_dims(a, axis=d-i-1+c) for i, a in enumerate(arrays))
    enum_res = []
    for i,a in enumerate(arrays):
        ind = [None]*d
        ind[i] = slice(None,None,None)
        if tensor_type == 2:
            ind = [slice(None,None,None)] + ind
        enum_res.append(a[ind])
    res = reduce(operator.mul, enum_res)
    if tensor_type ==1:
        return res.ravel()
    else:
        return  res.reshape((res.shape[0],-1))

class TensorBase:

    def __init__(self, bases):
        self.bases = bases
        is_compact_base = [isinstance(b, CompactLinearBasis) for b in bases]
        if len(set(is_compact_base))>1:
            raise Exception("One cannot mix (for now) compact support bases with non-compact ones.")
        self.__compact_bases__ = is_compact_base[0]
        self.d = len(bases)

    @property
    def grid(self):
        return cartesian([b.nodes for b in self.bases])

    def Phi(self, x, orders=None):
        x = np.asarray(x)
        if orders is None:
            orders = [None]*len(self.bases)

        ll = list( b.eval(x[..., i], orders=orders[i]) for i,b in enumerate(self.bases) )
        if self.__compact_bases__:
            return CompactKroneckerProduct(ll)
        else:
            return KroneckerProduct(ll)


    def B(self, x):
        Phi = self.Phi(x)

        print(Phi)
        return Phi.as_matrix()
        # return self.Phi(x).as_matrix()

    def __str__(self):
        return str.join(" ⊗ ", [str(e) for e in self.bases])

    def filter(self, x, filter=True):

        x = np.asarray(x)
        d = self.d

        # c = np.zeros([b.m for b in self.bases])
        c = np.zeros_like(x) # here we should know he required sizes
        c[...] = x

        if d == 1:
            return self.bases[0].filter(x)

        if (d<=2) and x.ndim == d and filter:
            # need to generalize that
            # filter lines first
            b = self.bases[1]
            for n in range(self.bases[0].m):
                c[n,:] = b.filter(c[n,:])
            # filter columns now
            b = self.bases[0]
            for n in range(self.bases[1].m):
                c[:,n] = b.filter(c[:,n])
            return c

        else:
            B = self.B(self.grid)
            B = B.reshape((B.shape[0],-1))
            from numpy.linalg import solve
            xx = x.reshape((B.shape[-1],-1))
            cc = solve(B, xx)
            # cc = cc.reshape((-1,x.shape[-1]))
            if len(x.ravel())>B.shape[1]:
                cc = cc.reshape([b.m for b in self.bases] + [-1])
            else:
                cc = cc.reshape([b.m for b in self.bases])
            return cc


    # def eval(coeffs, x)

def test_product_dense():

    n = 10
    n_1 = 10
    n_2 = 20

    from interpolation.linear_bases.basis_linear import UniformLinearSplineBasis as UniformLinearSpline
    from interpolation.linear_bases.basis_chebychev import ChebychevBasis

    cb = ChebychevBasis(min=0,max=1,n=n_1)
    lb = UniformLinearSpline(0,1,n_2)

    tp = TensorBase([cb, lb])

    print( tp.B(tp.grid).shape )

    xvec = numpy.linspace(0,1,10)
    yvec = numpy.linspace(0,1,20)

    def f(x,y):
        return x**2 + y**3/(1+y)

    values = f(tp.grid[:,0], tp.grid[:,1]).reshape((n_1,n_2))

    coeffs = tp.filter(values)
    coeffs_2 = tp.filter(values, filter=False)


    values.shape

    from matplotlib import pyplot as plt
    plt.plot(values[:,0])

    plt.plot(tp.bases[1].filter(values[:,0]))

    c = tp.bases[1].filter(values[:,0])
    Phi = tp.bases[1].eval(tp.bases[0].nodes)


    print( abs(coeffs-coeffs_2).max() )
    assert(abs(coeffs-coeffs_2).max()<1e-8)

    coeffs
    coeffs_2
    coeffs - coeffs_2



    # try to evaluate the interpolant

    tvec = numpy.linspace(0,1,100)
    s = np.concatenate([tvec[:,None],tvec[:,None]],axis=1)
    Phi = tp.Phi(s)

    vv = Phi*coeffs
    true_vals = [f(v,v) for v in tvec]


    # take derivative w.r.t. various coordinates
    Phi_0 = tp.Phi(s,orders=[1,0,0]).as_matrix()
    Phi_1 = tp.Phi(s,orders=[0,1,0]).as_matrix()
    Phi_2 = tp.Phi(s,orders=[0,0,1]).as_matrix()

    # does not work yet: should compute all necessary derivatives
    # Phi_diff = tp.Phi(s, orders=[[0,1],[0,1],[0,1]])


    # multivalued functions

    vvalues = np.concatenate([values[:,:,None],values[:,:,None]],axis=2)
    ccoeffs = tp.filter(vvalues)
    abs(ccoeffs[:,:,0] - coeffs).max()<1e-8
    abs(ccoeffs[:,:,1] - coeffs).max()<1e-8

    # assert( abs( vv - true_vals ).max() < 1e-6 )
    from matplotlib import pyplot as plt
    plt.plot(tvec, vv)
    plt.plot(tvec, true_vals)
    # plt.plot(tvec, true_vals-vv)
    plt.show()


def test_product_compact():

    n = 10
    n_1 = 10
    n_2 = 20

    # from interpolation.linear_bases.basis_linear import UniformLinearSplineBasis as UniformLinearSpline
    from interpolation.linear_bases.basis_uniform_cubic_splines import UniformSpline

    # cb = ChebychevBasis(min=0,max=1,n=n_1)
    lb1 = UniformSpline(0,1,n_1)
    lb2 = UniformSpline(0,1,n_2)

    tb = TensorBase([lb1, lb2])

    print(tb)
    print( tb.B(tb.grid).shape )




    tp = (tb.Phi(tb.grid))

    xvec = numpy.linspace(0,1,10)
    yvec = numpy.linspace(0,1,20)




if __name__ == '__main__':



    test_product_dense()
    test_product_compact()
