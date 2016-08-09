from interpolation.linear_bases.chebychev import LinearBasis, ChebychevBasis
from interpolation.cartesian import cartesian
import numpy.linalg
import numpy as np

class TensorProduct:

    def __init__(self, arrays):
        # right now, arrays is a list of vectors or matrices
        # TODO: allow for 3d-array
        self.arrays = [np.asarray(a) for a in arrays]
        self.d = len(arrays)
        self.tensor_type = self.arrays[0].ndim
        # 1: tensor product of vectors
        # 2: vectorized tensor product
        # 3: vectorized tensor product with derivative informations
        if self.tensor_type==3:
            raise Exception('Not supported yet.')


    def as_matrix(self):
        if self.d == 1:
            return self.arrays[0]
        arrays = self.arrays
        # placeholder algo (should be optimized/generic)
        if self.tensor_type==1:
            if self.d == 2:
                res = arrays[0][:,None]*arrays[1][None,:]
            if self.d == 3:
                res = arrays[0][:,:,None]*arrays[1][:,None,:]*arrays[1][None,:,:]
            return res.ravel()
        elif self.tensor_type==2:
            if self.d == 2:
                res = arrays[0][:,:,None]*arrays[1][:,None,:]
            if self.d == 3:
                res = arrays[0][:,:,:,None]*arrays[1][:,:,None,:]*arrays[1][:,None,:,:]
            return res.reshape((res.shape[0],-1))
        else: # self.tensor_type==3:
            raise Exception('Not supported yet.')

    def __mul__(self, c):
        c = np.asarray(c)
        # placeholder algo
        mat = self.as_matrix()
        res = mat @ c.reshape((mat.shape[1],-1))
        return res

class TensorBase:

    def __init__(self, bases):
        self.bases = bases
        self.d = len(bases)

    @property
    def grid(self):
        return cartesian([b.nodes for b in self.bases])

    def Phi(self, x, orders=None):
        x = np.asarray(x)
        if orders is None:
            orders = [None]*len(self.bases)
        return TensorProduct(
            list( b.eval(x[..., i], orders=orders[i]) for i,b in enumerate(self.bases) )
                )

    def B(self, x):
        return self.Phi(x).as_matrix()

    def __str__(self):
        return str.join(" ⊗ ", [str(e) for e in self.bases])

    def filter(self, x, filter=True):

        x = np.asarray(x)
        d = self.d
        # c = np.zeros(tuple([b.m for b in self.bases]))
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

if __name__ == '__main__':

    n = 10
    n_1 = 10
    n_2 = 20
    from interpolation.linear_bases.linear import UniformLinearSpline
    from interpolation.linear_bases.chebychev import ChebychevBasis
    cb = ChebychevBasis(min=0,max=1,n=n_1)
    lb = UniformLinearSpline(min=0,max=1,n=n_2)

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

    plt.plot(values[:,0])

    plt.plot(tp.bases[1].filter(values[:,0]))

    c = tp.bases[1].filter(values[:,0])
    Phi = tp.bases[1].eval(tp.bases[0].nodes)



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
