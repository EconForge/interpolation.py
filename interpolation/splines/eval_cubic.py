import numpy

from .eval_splines import eval_cubic

## the functions in this file provide backward compatibility calls
##
## they can optionnally allocate memory for the result
## they work for any dimension, except the functions which compute the gradient

#######################
# Compatibility calls #
#######################

from numba import generated_jit
from .codegen import source_to_function


@generated_jit
def get_grid(a, b, n, C):
    d = C.ndim
    s = "({},)".format(str.join(", ", [f"(a[{k}],b[{k}],n[{k}])" for k in range(d)]))
    txt = "def get_grid(a,b,n,C): return {}".format(s)
    f = source_to_function(txt)
    return f


def eval_cubic_spline(a, b, orders, coefs, point):
    """Evaluates a cubic spline at one point

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension (=(n1,...,nd) )
    coefs : array of dimension d, and size (n1+2, ..., nd+2)
        Filtered coefficients.
    point : array of size d
        Coordinate of the point where the splines must be interpolated.

    Returns
    -------
    value : float
        Interpolated value.
    """
    grid = get_grid(a, b, orders, coefs)
    return eval_cubic(grid, coefs, point)


def vec_eval_cubic_spline(a, b, orders, coefs, points, values=None):
    """Evaluates a cubic spline at many points

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension. (=(n1,...,nd))
    coefs : array of dimension d, and size (n1+2, ..., nd+2)
        Filtered coefficients.
    points : array of size N x d
        List of points where the splines must be interpolated.
    values (optional) :  array of size (N)
        If not None, contains the result.

    Returns
    -------
    values : array of size (N)
        Interpolated values. values[i] contains spline evaluated at point points[i,:].
    """

    grid = get_grid(a, b, orders, coefs)
    if values is None:
        return eval_cubic(grid, coefs, points)
    else:
        eval_cubic(grid, coefs, points, values)


def eval_cubic_splines(a, b, orders, mcoefs, point, values=None):
    """Evaluates multi-splines at one point.

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension.
    mcoefs : array of dimension d+1, and size (p, n1+2, ..., nd+2)
        Filtered coefficients. For i in 1:(mcoefs.shape[0]), mcoefs[i,...] contains
        the coefficients of spline number i.
    point : array of size d
        Point where the spline must be interpolated.
    values (optional) :  array of size (p)
        If not None, contains the result.

    Returns
    -------
    values : array of size (p)
        Interpolated values. values[j] contains spline n-j evaluated at point `point`.
    """

    grid = get_grid(a, b, orders, mcoefs[..., 0])
    if values is None:
        return eval_cubic(grid, mcoefs, point)
    else:
        eval_cubic(grid, mcoefs, point, values)


def vec_eval_cubic_splines(a, b, orders, mcoefs, points, values=None):
    """Evaluates multi-splines on a series of points.

    Parameters:
    -----------
    a : array of size d (float)
        Lower bounds of the cartesian grid.
    b : array of size d (float)
        Upper bounds of the cartesian grid.
    orders : array of size d (int)
        Number of nodes along each dimension. ( =(n1,...nd) )
    mcoefs : array of dimension d+1, and size (n1+2, ..., nd+2, p)
        Filtered coefficients. coefs[i,...] contains the coefficients of spline number i.
    points : array of size N x d
        List of points where the splines must be interpolated.
    values (optional) :  array of size (N x p)
        If not None, contains the result.

    Returns
    -------
    values : array of size (N x p)
        Interpolated values. values[i,j] contains spline n-j evaluated at point points[i,:].
    """

    grid = get_grid(a, b, orders, mcoefs[..., 0])
    if values is None:
        return eval_cubic(grid, mcoefs, points)
    else:
        eval_cubic(grid, mcoefs, points, values)


#########

from .eval_cubic_numba import (
    vec_eval_cubic_splines_G_1,
    vec_eval_cubic_splines_G_2,
    vec_eval_cubic_splines_G_3,
    vec_eval_cubic_splines_G_4,
)


def vec_eval_cubic_splines_G(a, b, orders, mcoefs, points, values=None, dvalues=None):

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = a.shape[0]
    N = points.shape[0]
    n_sp = mcoefs.shape[-1]

    if values is None:
        values = numpy.empty((N, n_sp))

    if dvalues is None:
        dvalues = numpy.empty((N, d, n_sp))

    if d == 1:
        vec_eval_cubic_splines_G_1(a, b, orders, mcoefs, points, values, dvalues)

    elif d == 2:
        vec_eval_cubic_splines_G_2(a, b, orders, mcoefs, points, values, dvalues)

    elif d == 3:
        vec_eval_cubic_splines_G_3(a, b, orders, mcoefs, points, values, dvalues)

    elif d == 4:
        vec_eval_cubic_splines_G_4(a, b, orders, mcoefs, points, values, dvalues)

    return [values, dvalues]
