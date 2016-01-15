from __future__ import division

from numpy import *
from interpolation.cartesian import mlinspace

d = 2               # number of dimension
Ng = 100          # number of points on the grid
K = int(Ng**(1/d))  # nb of points in each dimension
N = 100           # nb of points to evaluate
a = array([0.0]*d, dtype=float)
b = array([1.0]*d, dtype=float)
orders = array([K]*d, dtype=int)

grid = mlinspace(a,b,orders)

# single valued function to interpolate
f = lambda vec: sqrt(vec.sum(axis=1))
# df

# single valued function to interpolate
vals = f(grid)

from interpolation.splines.filter_cubic import filter_coeffs
cc = filter_coeffs(a,b,orders,vals)


def test_cuda():

    mvals = concatenate([vals[:,None],vals[:,None]],axis=1)

    # many points
    points = row_stack([[0.5, 0.5]]*N)

    from numba import cuda

    from interpolation.splines.eval_cubic_cuda import vec_eval_cubic_spline_2, Ad, dAd
    jitted = cuda.jit(vec_eval_cubic_spline_2)

    out = zeros(N)
    jitted[(N,1)](a,b,orders,cc,points,out,Ad,dAd)

    # compare
    from interpolation.splines.eval_cubic import vec_eval_cubic_spline
    test = vec_eval_cubic_spline(a,b,orders,cc,points)
    assert( (abs(test - out).max() < 1e-8 ) )


    # now with multi splines and gradient evaluation
    ccc = cc[:,:,None].repeat(3, axis=2) # three times the same spline

    # same thing with gradient evaluation
    from interpolation.splines.eval_cubic_cuda import vec_eval_cubic_splines_G_2
    jitted_G = cuda.jit(vec_eval_cubic_splines_G_2)

    out = zeros((N,3))
    dout = zeros((N,2,3)) # (n,i,j) i-th derivative of the j-th spline at the n-th point
    jitted_G[(N,1)](a,b,orders,ccc,points,out,dout, Ad,dAd)

    # compare
    from interpolation.splines.eval_cubic import vec_eval_cubic_splines_G
    test, dtest = vec_eval_cubic_splines_G(a,b,orders,ccc,points)
    
    assert( (abs(test - out).max() < 1e-8 ) )
    assert( (abs(dtest - dout).max() < 1e-8 ) )


if __name__ == '__main__':

    test_cuda()
