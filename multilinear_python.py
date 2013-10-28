from __future__ import division

from numpy import array, zeros, floor, cumprod, column_stack, reshape


from itertools import product
import numpy as np

def multilinear_interpolation( smin, smax, orders, x, y):

    '''
    :param smin: dx1 array : lower bounds
    :param smax: dx1 array : upper bounds
    :param orders: dx1 array : number of points in each dimension
    :param x: Nxd array : values on the grid
    :param y: Mxd array : points where to interpolate
    :return: Mxd array : interpolated values
    '''

    dtype = x.dtype

    d = len(orders)

    n_x, N = x.shape
    n_s, M = y.shape

    assert(d == n_s)

    qq = zeros( (d,M), dtype=dtype )
    mm = zeros( (d,M), dtype=dtype )

    for i in range(n_s):
        s = (y[i,:]-smin[i])/(smax[i]-smin[i])
        n = orders[i]
        delta = 1/(n-1)
        r = s/delta
        q = floor(r)
        q = (q<0) * 0 + (q>=n-2)*(n-2) + (0<=q)*(q<n-2)*q
        m = r-q
        mm[i,:] = m
        qq[i,:] = q

    [b,g] = index_lookup( x, qq, orders )

    z = b + recursive_evaluation(d,tuple([]),mm[:,np.newaxis,:], g)

    return z

def recursive_evaluation(d,ind,mm,g):
    if len(ind) == d:
        return g[ind]
    else:
        j = len(ind)
        ind1 = ind + (0,)
        ind2 = ind + (1,)
        return (1-mm[j,:]) * recursive_evaluation(d,ind1,mm,g) + mm[j,:] * recursive_evaluation(d,ind2,mm,g)


def index_lookup(a, q, dims):
    '''

    :param a: (l1*...*ld)*n_x array
    :param q: k x M array
    :param dims: M: array
    :return: 2**k array (nx*2*...*2)
    '''

    dtype = a.dtype

    M = q.shape[1]
    n_x = a.shape[0]

    d = len(dims)

    cdims  = (cumprod(dims[::-1]))
    cdims = cdims[::-1]

    q = array(q,dtype=np.int)

    lin_q = q[d-1,:]

    for i in reversed(range(d-1)):
        lin_q += q[i,:] * cdims[i+1]

    cart_prod = column_stack( [e for e in product(*[(0,1)]*d)] )

    lin_cp = cart_prod[d-1,:]
    for i in reversed(range(d-1)):
        lin_cp += cart_prod[i,:] * cdims[i+1]

    b = a[:,lin_q]

    g = zeros( (cart_prod.shape[1], n_x, M), dtype=dtype )

    for i in range(cart_prod.shape[1]):
        t = a[:,lin_q + lin_cp[i]] - b
        g[i,:,:] = t


    g = reshape(g, (2,)*d + (n_x,M))

    return [b,g]
