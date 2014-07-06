from __future__ import print_function

import pyximport
pyximport.install()
from numba import jit, njit

import numpy

## the functions in this file work for any dimension (d<=3)
## they can optionnally allocate memory for the result

def eval_cubic_spline(a, b, orders, coefs, point, value):

    pass


def eval_cubic_spline_d(a, b, orders, coefs, point, value, dvalue):

    pass


def eval_cubic_multi_spline(a, b, orders, mcoefs, point, values):

    pass


def eval_cubic_multi_spline_d(a, b, orders, mcoefs, point, values, dvalues):

    pass

# try:

# from eval_cubic_splines_cython import vec_eval_cubic_multi_spline_1, vec_eval_cubic_multi_spline_2
#from eval_cubic_splines_cython import vec_eval_cubic_multi_spline_3, vec_eval_cubic_multi_spline_4

from eval_cubic_splines_numba import vec_eval_cubic_multi_spline_1, vec_eval_cubic_multi_spline_2
# from eval_cubic_splines_numba import vec_eval_cubic_multi_spline_3, vec_eval_cubic_multi_spline_4

def vec_eval_cubic_multi_spline(a, b, orders, mcoefs, points, values=None):
    """Evaluates multi-splines on a series of points.

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension.
    mcoefs : array of dimension d+1, and size (p, n1, ..., nd)
        Filtered coefficients. For i in 1:(mcoefs.shape[0]), mcoefs[i,...] contains
        the coefficients of splines number i.
    points : array of size N x d
        List of points where the splines must be interpolated.
    values (optional) :  array of size (N x p)
        If not None, contains the result.



    Returns
    -------
    values : array of size (N x p)
        Interpolated values. values[i,j] contains spline n-j evaluated at point points[i,:].
    """



    d = a.shape[0]

    if values is None:

        N = points.shape[0]
        n_sp = mcoefs.shape[0]
        values = numpy.empty((N, n_sp))

    if d == 1:
        vec_eval_cubic_multi_spline_1(a, b, orders, mcoefs, points, values)

    elif d == 2:
        vec_eval_cubic_multi_spline_2(a, b, orders, mcoefs, points, values)

    elif d == 3:
        vec_eval_cubic_multi_spline_3(a, b, orders, mcoefs, points, values)

    elif d == 4:
        vec_eval_cubic_multi_spline_4(a, b, orders, mcoefs, points, values)


    return values




def vec_eval_cubic_multi_spline_d(a, b, orders, mcoefs, points, values=None, dvalues=None):

    pass
