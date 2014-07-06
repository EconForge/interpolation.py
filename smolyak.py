"""

We demonstrate the use of the ```SmolyakGrid``` object using the function
:math:`x,y \\rightarrow \\exp \\left( -x^2 - y^2 \\right)`. We define it on a subset of :math:`[-1,1]^2` and use a
smolyak product of chebychev polynomials to approximate it at other points.

Let define the state space :math:`[-1,1]^2` and the interpolation object.


.. code-block:: python

    from dolo.numeric.interpolation import SmolyakGrid
    import numpy


    bounds = numpy.array([
        [ -1, -1 ],    # lower bounds
        [ 1, 1 ]     # upper bounds
    ])

    sg = SmolyakGrid(bounds, 3)  # 3 is the smolyak parameter l

    print(sg.grid.shape)

The points selected by the smolyak algorithm are accessible in ```sg.grid```  which is a ```2x13``` matrix. They can by plotted with:

.. code-block:: python

    sg.plot_grid()

Let define a two-variables function and evaluate it on the grid and we initialize the interpolator with these values.

.. code-block:: python

    fun = lambda x,y: numpy.exp( -numpy.power(x,2) - numpy.power(y,2) )
    values_on_the_grid = fun( sg.grid[0,:], sg.grid[1,:] )

    sg.set_values(values_on_the_grid)

Now we construct a random matrix ```s``` of size```2xN``` where each column is a point of the state space. On these
states, we compute interpolated values and compare it with the true ones.

.. code-block:: python

    from  numpy.random import multivariate_normal as  mvtnorm
    mean = numpy.zeros(2)
    cov = numpy.eye(2)
    s = mvtnorm(mean,cov,50).T

    interpolated_values = sg.interpolate( s )

    true_values = fun(s[0,:], s[1,:])

    max_error = abs( true_values - interpolated_values ).max()

    print( 'Maximum error : {}'.format(max_error) )



"""

from __future__ import division

import numpy
import numpy.linalg

from functools import reduce

from numpy import array

from operator import mul
from itertools import product

def cheb_extrema(n):
    jj = numpy.arange(1.0,n+1.0)
    zeta =  numpy.cos( numpy.pi * (jj-1) / (n-1 ) )
    return zeta

def chebychev(x, n):
    # computes the chebychev polynomials of the first kind
    dim = x.shape
    results = numpy.zeros((n+1,) + dim)
    results[0,...] = numpy.ones(dim)
    results[1,...] = x
    for i in range(2,n+1):
        results[i,...] = 2 * x * results[i-1,...] - results[i-2,...]
    return results

def chebychev2(x, n):
    # computes the chebychev polynomials of the second kind
    dim = x.shape
    results = numpy.zeros((n+1,) + dim)
    results[0,...] = numpy.ones(dim)
    results[1,...] = 2*x
    for i in range(2,n+1):
        results[i,...] = 2 * x * results[i-1,...] - results[i-2,...]
    return results

def enum(d,l):
    r = range(l)
    b = l - 1
    #stupid :
    res = []
    for maximum in range(b+1):
        res.extend( [e for e in product(r, repeat=d ) if sum(e)==maximum ] )
    return res

def build_indices_levels(l):
    return [(0,)] + [(1,2)] + [ tuple(range(2**(i-1)+1, 2**(i)+1)) for i in range(2,l) ]

def build_basic_grids(l):
    ll = [ numpy.array( [0.5] ) ]
    ll.extend( [ numpy.linspace(0.0,1.0,2**(i)+1) for i in range(1,l) ]  )
    ll = [ - numpy.cos( e * numpy.pi ) for e in ll]
    incr = [[0.0],[-1.0,1.0]]
    for i in range(2,len(ll)):
        t = ll[i]
        n = (len(t)-1)/2
        tt =  [ t[2*n+1] for n in range( int(n) ) ]
        incr.append( tt )
    incr = [numpy.array(i) for i in incr]
    return [ll,incr]

def smolyak_grids(d,l):

    ret,incr = build_basic_grids(l)
    tab =  build_indices_levels(l)

    eee =  [ [ tab[i] for i in e] for e in enum( d, l) ]
    smolyak_indices = []
    for ee in eee:
        smolyak_indices.extend( [e for e in product( *ee ) ] )

    fff =  [ [ incr[i] for i in e] for e in enum( d, l) ]
    smolyak_points = []
    for ff in fff:
        smolyak_points.extend( [f for f in product( *ff ) ] )

    smolyak_points = numpy.c_[smolyak_points]

    return [smolyak_points, smolyak_indices]

class SmolyakBasic(object):
    '''Smolyak interpolation on [-1,1]^d'''

    def __init__(self,d,l, dtype=None):

        self.d = d
        self.l = l

        [self.smolyak_points, self.smolyak_indices] = smolyak_grids(d,l)

        self.u_grid = array( self.smolyak_points.T, order='C')

        self.isup = max(max(self.smolyak_indices))
        self.n_points = len(self.smolyak_points)
        #self.grid = self.real_gri

        Ts = chebychev( self.smolyak_points.T, self.n_points - 1 )
        C = []
        for comb in self.smolyak_indices:
            p = reduce( mul, [Ts[comb[i],i,:] for i in range(self.d)] )
            C.append(p)
        C = numpy.row_stack( C )

        # C is such that :  X = theta * C
        self.__C__ = C
        self.__C_inv__ = numpy.linalg.inv(C)  # would be better to store preconditioned matrix

        self.bounds = numpy.row_stack([(0,1)]*d)

    def __call__(self,s):

        if s.ndim == 1:
            res = self.__call__( numpy.atleast_2d(s).T )
            return res.ravel()


        return self.interpolate(s)

    def interpolate(self, s, with_derivative=True, with_theta_deriv=False, with_X_deriv=False):

        # points in x must be stacked horizontally

        theta = self.theta

        [n_v, n_t] = theta.shape  # (n_v, n_t) -> (number of variables?, ?)
        assert( n_t == self.n_points )

        [n_d, n_p] = s.shape  # (n_d, n_p) -> (number of dimensions, number of points)
        n_obs = n_p # by def
        assert( n_d == self.d )

        Ts = chebychev( s, self.n_points - 1 )

        coeffs = []
        for comb in self.smolyak_indices:
            p = reduce( mul, [Ts[comb[i],i,:] for i in range(self.d)] )
            coeffs.append(p)
        coeffs = numpy.row_stack( coeffs )

        val = numpy.dot(theta,coeffs)
#
        if with_derivative:

            # derivative w.r.t. arguments
            Us = chebychev2( s, self.n_points - 2 )
            Us = numpy.concatenate([numpy.zeros( (1,n_d,n_obs) ), Us],axis=0)
            for i in range(Us.shape[0]):
                Us[i,:,:] = Us[i,:,:] * i

            der_s = numpy.zeros( ( n_t, n_d, n_obs ) )
            for i in range(n_d):
                #BB = Ts.copy()
                #BB[:,i,:] = Us[:,i,:]
                el = []
                for comb in self.smolyak_indices:
                    #p = reduce( mul, [BB[comb[j],j,:] for j in range(self.d)] )
                    p = reduce( mul, [ (Ts[comb[j],j,:] if i!=j else Us[comb[j],j,:]) for j in range(self.d)] )
                    el.append(p)
                el = numpy.row_stack(el)
                der_s[:,i,:] =  el
            dder = numpy.tensordot( theta, der_s, (1,0) )

            if with_theta_deriv:
                # derivative w.r.t. to theta
                l = []
                for i in range(n_v):
                    block = numpy.zeros( (n_v,n_t,n_obs) )
                    block[i,:,:] = coeffs
                    l.append(block)
                dval = numpy.concatenate( l, axis = 1 )

                if with_X_deriv:

                    dd = dval.reshape( (dval.shape[0],) + theta.shape + (dval.shape[2],) )

                    C_inv = self.__C_inv__

                    d_x = numpy.tensordot( dd, C_inv.T, axes=(2,0))
                    d_x = d_x.swapaxes(2,3)
                    pp = theta.size
                    d_x = d_x.reshape( (dval.shape[0],) + (pp, ) + (dval.shape[2],) )
                    return [val,dder,dval,d_x]

                else:
                    return [val,dder,dval]

            else:
                return [val,dder]

        else:
            return val


    def set_values(self,x):
        """ Updates self.theta parameter. No returns values"""

        x = numpy.atleast_2d(x)

        x = x.real # ahem

        C_inv = self.__C_inv__
        theta = numpy.dot( x, C_inv )
        self.theta = theta

        return theta

    def plot_grid(self):
        import matplotlib.pyplot as plt
        grid = self.smolyak_points
        if grid.shape[1] == 2:
            xs = grid[0,:]
            ys = grid[1,:]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        elif grid.shape[1] == 3:
            xs = grid[0,:]
            ys = grid[1,:]
            zs = grid[2,:]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        else:
            raise ValueError('Can only plot 2 or 3 dimensional problems')

class SmolyakGridRows(SmolyakBasic):

    '''Smolyak interpolation'''

    def __init__(self, smin, smax, l, axes=None, dtype=None):
        """
        @param bounds: matrix of bounds
        @param l:
        @param axes:
        @return: a smolyak interpolator
        """

        bounds = numpy.row_stack([smin,smax])

        d = bounds.shape[1]

        super(SmolyakGridRows, self).__init__( d, l)

        self.bounds = bounds
        self.smin = numpy.array(smin)
        self.smax = numpy.array(smax)

        self.center = [b[0]+(b[1]-b[0])/2 for b in bounds.T]
        self.radius =  [(b[1]-b[0])/2 for b in bounds.T]

        if not axes == None:
            self.P = numpy.dot( axes, numpy.diag(self.radius))
            self.Pinv = numpy.linalg.inv(axes)
        else:
            self.P = numpy.diag(self.radius)
            self.Pinv = numpy.linalg.inv(self.P)

        base = numpy.eye(d)
        image_of_base = numpy.dot( self.P , base)

        self.grid = self.A( self.u_grid )

    # A goes from [0,1] to bounds
    def A(self,x):
#        '''A is the inverse of B'''
        N = x.shape[1]
        c = numpy.tile(self.center, (N,1) ).T
        P = self.P
        return c + numpy.dot(P, x)

    # B returns from bounds to [0,1]
    def B(self,y):
#        '''B is the inverse of A'''
        N = y.shape[1]
        c = numpy.tile(self.center, (N,1) ).T
        Pinv = self.Pinv
        return numpy.dot(Pinv,y-c)

    def interpolate(self, y, with_derivative=False, with_theta_deriv=False, with_X_deriv=False):

        x = self.B(y)  # Transform back to [0,1]
        res = super(SmolyakGridRows, self).interpolate( x, with_derivative=with_derivative, with_theta_deriv=with_theta_deriv, with_X_deriv=with_X_deriv)  # Call super class' (SmolyakGrid) interpolate func
        if with_derivative or with_theta_deriv or with_X_deriv:
            dder = res[1]
            dder = numpy.tensordot(dder, self.Pinv, axes=(1,0)).swapaxes(1,2)
            res[1] = dder
        return res
#    return res
#        if with_derivative:
#            if with_theta_deriv:
#                [val,dder,dval] = res
#                dder = numpy.tensordot(dder, self.Pinv, axes=(1,0)).swapaxes(1,2)
#                return [val,dder,dval]
#            else:
#                [val,dder] = res
#                dder = numpy.tensordot(dder, self.Pinv, axes=(1,0)).swapaxes(1,2)
#                return [val,dder]
#        else:
#            return res


    def plot_grid(self):
        import matplotlib.pyplot as plt
        grid = self.grid
        if grid.shape[1] == 2:
            xs = grid[:, 0]
            ys = grid[:, 1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        elif grid.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            xs = grid[:, 0]
            ys = grid[:, 1]
            zs = grid[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        else:
            raise ValueError('Can only plot 2 or 3 dimensional problems')


class SmolyakGrid(SmolyakGridRows):

    def __init__(self, smin, smax, l, axes=None, dtype=None):

        super(SmolyakGrid, self).__init__( smin, smax, l, axes=None, dtype=None)

        self.grid = numpy.ascontiguousarray(self.grid.T)


    def set_values(self, x):

        super(SmolyakGrid, self).set_values(x.T)

    def __call__(self, y):

        res = self.interpolate(y)
        return res

    #
    def interpolate(self, y):

        if y.ndim == 1:
            y = numpy.atleast_2d(y)
            res = self.interpolate(y)
            return res.ravel()

        res = super(SmolyakGrid, self).interpolate(y.T).T
        return res


if __name__ == '__main__':

    from numpy import column_stack, array

    smin = [-2,-1]
    smax = [0.5,2]

    sg = SmolyakGrid(smin,smax,2)
    sg2 = SmolyakGrid(smin,smax,3)

#    print(sg.u_grid)
    grid = sg.grid
    grid2 = sg2.grid

    print(grid2.shape)

    values = column_stack( [grid[:,0] * (1-grid[:,1])/2.0, grid[:,1], grid[:,0]] )
    print("Values")
    print(values.shape)

    sg.set_values(values)

    print('sizes of the grids')
    print(sg.u_grid.shape)
    print(sg2.u_grid.shape)

    sh = values.shape
    print('values')
    print(sh)

    print('theta')
    print(sg.theta.shape)

    def fun(x):

        sg.set_values(x.reshape(sh))
        res = sg.interpolate(grid2)
        return res
#
#    def fun(theta):
#
#        sg.theta = theta.reshape(sh).copy()
#        res = sg.interpolate(grid2, with_derivative=False)
#        return res
#

    from dolo.numeric.serial_operations import numdiff1, numdiff2

#    x0 = sg.theta.flatten()
    x0 = values.flatten()
    print(x0.shape)

    test = fun(x0)
    print('image')
    print( test.shape)
#    dtest = numdiff1(grid[1] fun, values)
    dtest = numdiff2( fun, x0, dv=1e-5)

#    print(values.shape)
#    print(test.shape)
#    print(dtest)

    print('numerical derivative')
    print(dtest.shape)

#    dtest = dtest.swapaxes(1,2)

#    print(test.shape)
#    print(test)
#    [res1, res2, res3] = sg.interpolate(grid2, with_theta_deriv=True)
#
    print('symbolic derivative')
    [res1, res2, res3, res4] = sg.interpolate(grid2, with_derivative=True, with_theta_deriv=True, with_X_deriv=True)
    print(res4.shape)
#
    diff = dtest - res4

#
    print( abs(diff).max() )
#
#    print(res3.shape)

