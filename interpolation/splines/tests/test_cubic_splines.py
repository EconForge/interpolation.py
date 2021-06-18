from __future__ import division

from numpy import *
from interpolation.cartesian import mlinspace

d = 2  # number of dimension
Ng = 1000  # number of points on the grid
K = int(Ng ** (1 / d))  # nb of points in each dimension
N = 10000  # nb of points to evaluate
a = array([0.0] * d, dtype=float)
b = array([1.0] * d, dtype=float)
orders = array([K] * d, dtype=int)

grid = mlinspace(a, b, orders)

# single valued function to interpolate
f = lambda vec: sqrt(vec.sum(axis=1))
# df

# # vector valued function
# g


# single valued function to interpolate
vals = f(grid)

print(vals)
mvals = concatenate([vals[:, None], vals[:, None]], axis=1)

print(mvals.shape)

# one single point
point = array([0.5, 0.5])

# many points
points = row_stack([[0.5, 0.5]] * N)


def test_cubic_spline():

    from interpolation.splines.filter_cubic import filter_coeffs
    from interpolation.splines.eval_cubic import (
        eval_cubic_spline,
        vec_eval_cubic_spline,
    )

    cc = filter_coeffs(a, b, orders, vals)
    assert tuple(cc.shape) == tuple([o + 2 for o in orders])

    ii = eval_cubic_spline(a, b, orders, cc, point)
    iii = vec_eval_cubic_spline(a, b, orders, cc, points)

    assert isinstance(ii, float)
    assert iii.ndim == 1


def test_cubic_multi_spline():

    from interpolation.splines.filter_cubic import filter_mcoeffs
    from interpolation.splines.eval_cubic import (
        eval_cubic_splines,
        vec_eval_cubic_splines,
    )

    cc = filter_mcoeffs(a, b, orders, mvals)
    assert tuple(cc.shape) == tuple([o + 2 for o in orders] + [mvals.shape[1]])

    ii = eval_cubic_splines(a, b, orders, cc, point)
    iii = vec_eval_cubic_splines(a, b, orders, cc, points)

    assert ii.ndim == 1
    assert iii.ndim == 2


def test_cubic_spline_object():

    from interpolation.splines import CubicSpline

    cs = CubicSpline(a, b, orders, vals)
    ii = cs(point)
    iii = cs(points)

    assert ii.ndim == 0
    assert isscalar(ii)
    assert iii.ndim == 1
    assert tuple(iii.shape) == (N,)


def test_cubic_multi_spline_object():

    from interpolation.splines import CubicSplines

    cs = CubicSplines(a, b, orders, mvals)
    ii = cs(point)
    iii = cs(points)

    n_splines = mvals.shape[1]
    assert ii.ndim == 1
    assert tuple(ii.shape) == (n_splines,)
    assert iii.ndim == 2
    assert tuple(iii.shape) == (N, n_splines)


if __name__ == "__main__":

    test_cubic_spline()
    test_cubic_multi_spline()
    test_cubic_spline_object()
    test_cubic_multi_spline_object()
