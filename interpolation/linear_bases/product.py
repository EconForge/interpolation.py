# from interpolation.linear_bases.chebychev import LinearBasis, ChebychevBasis
from interpolation.cartesian import cartesian
import numpy.linalg

class TensorProduct:

    def __init__(self, arrays):
        self.arrays = arrays
        self.d = len(arrays)

    def as_matrix(self):
        if self.d == 1:
            return self.arrays[0]
        arrays = self.arrays
        if arrays[0].ndim == 1:
            return arrays[0][:,None]*arrays[1][None,:]
        else:
            return arrays[0][:,:,None]*arrays[1][:,None,:]

    def __mul__(self, c):
        c = np.asarray(c)
        assert(self.d<=2)
        if self.d == 1:
            return self.arrays[0] @ c
        assert (self.d == 2)
        arrays = self.arrays
        N = arrays[0].shape[0]
        # I = arrays[0].shape[1]
        # J = arrays[1].shape[1]
        if len(c.shape) == self.d:
            res = np.zeros((N))
            for n in range(N):
                hh = arrays[0][n,:][:,None]*arrays[1][n,:][None,:]
                val = hh *c
                res[n] = val.sum(axis=(0,1))
            return res

class TensorBase:

    def __init__(self, bases):
        self.bases = bases
        self.d = len(bases)

    @property
    def grid(self):
        return cartesian([b.nodes for b in self.bases])

    def Phi(self, x):
        x = np.asarray(x)
        return TensorProduct(
            list( b.eval(x[..., i]) for i,b in enumerate(self.bases) )
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
            for n in range(self.bases[1].m):
                c[:,n] = b.filter(c[:,n])
            return c

        else:
            B = tp.B(tp.grid)
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

    cb = ChebychevBasis()

    tp = TensorBase([cb, cb])

    tp.B(tp.grid)

    xvec = numpy.linspace(0,1,10)
    yvec = numpy.linspace(0,1,20)
    def f(x,y):
        return x**2 + y**3/(1+y)

    values = f(tp.grid[:,0], tp.grid[:,1]).reshape((10,10))

    coeffs = tp.filter(values)
    coeffs_2 = tp.filter(values, filter=False)

    assert(abs(coeffs-coeffs_2).max()<1e-8)





    # try to evaluate the interpolant

    tvec = numpy.linspace(0,1,100)
    s = np.concatenate([tvec[:,None],tvec[:,None]],axis=1)
    Phi = tp.Phi(s)

    vv = Phi*coeffs
    true_vals = [f(v,v) for v in tvec]

    plt.plot(tvec, vv)
    plt.plot(tvec, true_vals)

    assert( abs( vv - true_vals ).max() < 1e-6 )


    # multivalued functions

    vvalues = np.concatenate([values[:,:,None],values[:,:,None]],axis=2)
    ccoeffs = tp.filter(vvalues)
    abs(ccoeffs[:,:,0] - coeffs).max()<1e-8
    abs(ccoeffs[:,:,1] - coeffs).max()<1e-8
