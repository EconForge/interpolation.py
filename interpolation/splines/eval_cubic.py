import numpy

from .eval_cubic_numba import *

## the functions in this file work for any dimension (d<=4)
## they can optionnally allocate memory for the result


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

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = len(a)

    if d == 1:
        value = eval_cubic_spline_1(a, b, orders, coefs, point)

    elif d == 2:
        value = eval_cubic_spline_2(a, b, orders, coefs, point)

    elif d == 3:
        value = eval_cubic_spline_3(a, b, orders, coefs, point)

    elif d == 4:
        value = eval_cubic_spline_4(a, b, orders, coefs, point)

    return value


# def eval_cubic_spline_d(a, b, orders, coefs, point, value, dvalue):

# pass


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

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = a.shape[0]

    if values is None:
        N = points.shape[0]
        values = numpy.empty(N)

    if d == 1:
        vec_eval_cubic_spline_1(a, b, orders, coefs, points, values)
    elif d == 2:
        vec_eval_cubic_spline_2(a, b, orders, coefs, points, values)
    elif d == 3:
        vec_eval_cubic_spline_3(a, b, orders, coefs, points, values)
    elif d == 4:
        vec_eval_cubic_spline_4(a, b, orders, coefs, points, values)

    return values


# def eval_cubic_multi_spline_d(a, b, orders, mcoefs, point, values, dvalues):
#
#     pass


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

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = a.shape[0]

    if values is None:
        n_sp = mcoefs.shape[0]
        values = numpy.empty(n_sp)

    if d == 1:
        eval_cubic_splines_1(a, b, orders, mcoefs, point, values)

    elif d == 2:
        eval_cubic_splines_2(a, b, orders, mcoefs, point, values)

    elif d == 3:
        eval_cubic_splines_3(a, b, orders, mcoefs, point, values)

    elif d == 4:
        eval_cubic_splines_4(a, b, orders, mcoefs, point, values)

    return values


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
    mcoefs : array of dimension d+1, and size (p, n1+2, ..., nd+2)
        Filtered coefficients. For i in 1:(mcoefs.shape[0]), mcoefs[i,...] contains
        the coefficients of spline number i.
    points : array of size N x d
        List of points where the splines must be interpolated.
    values (optional) :  array of size (N x p)
        If not None, contains the result.

    Returns
    -------
    values : array of size (N x p)
        Interpolated values. values[i,j] contains spline n-j evaluated at point points[i,:].
    """

    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    orders = numpy.array(orders, dtype=int)

    d = a.shape[0]

    if values is None:

        N = points.shape[0]
        n_sp = mcoefs.shape[-1]
        values = numpy.empty((N, n_sp))

    if d == 1:
        vec_eval_cubic_splines_1(a, b, orders, mcoefs, points, values)

    elif d == 2:
        vec_eval_cubic_splines_2(a, b, orders, mcoefs, points, values)

    elif d == 3:
        vec_eval_cubic_splines_3(a, b, orders, mcoefs, points, values)

    elif d == 4:
        vec_eval_cubic_splines_4(a, b, orders, mcoefs, points, values)

    return values


def vec_eval_cubic_splines_G(a,
                             b,
                             orders,
                             mcoefs,
                             points,
                             values=None,
                             dvalues=None):

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
        vec_eval_cubic_splines_G_1(a, b, orders, mcoefs, points, values,
                                   dvalues)

    elif d == 2:
        vec_eval_cubic_splines_G_2(a, b, orders, mcoefs, points, values,
                                   dvalues)

    elif d == 3:
        vec_eval_cubic_splines_G_3(a, b, orders, mcoefs, points, values,
                                   dvalues)

    elif d == 4:
        vec_eval_cubic_splines_G_4(a, b, orders, mcoefs, points, values,
                                   dvalues)

    return [values, dvalues]
