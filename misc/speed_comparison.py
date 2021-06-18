from interpolation.splines.eval_cubic_numba import (
    vec_eval_cubic_spline_3,
    vec_eval_cubic_spline_2,
)
from interpolation.splines.filter_cubic import filter_coeffs
from interpolation.splines.multilinear_numba import multilinear_interpolation

from interpolation.splines.misc import mlinspace

import numpy

K = 50
d = 2
N = 10 ** 6
# N = 100

a = numpy.array([0.0] * d)
b = numpy.array([1.0] * d)
orders = numpy.array([K] * d, dtype=int)

V = numpy.random.random(orders)
C = filter_coeffs(a, b, orders, V)


X = numpy.random.random((N, d))
res = numpy.zeros(N)
res2 = res.copy()

if d == 3:
    vec_eval_cubic_spline = vec_eval_cubic_spline_3
elif d == 2:
    vec_eval_cubic_spline = vec_eval_cubic_spline_2

vec_eval_cubic_spline(a, b, orders, C, X, res)

multilinear_interpolation(a, b, orders, V, X, res)


import time

t1 = time.time()
vec_eval_cubic_spline(a, b, orders, C, X, res)
t2 = time.time()
multilinear_interpolation(a, b, orders, V, X, res2)
t3 = time.time()

print("Cubic: {}".format(t2 - t1))
print("Linear: {}".format(t3 - t2))


# assert(abs(res-res2).max()<1e-10)


# scipy
from scipy.interpolate import RegularGridInterpolator

pp = [numpy.linspace(a[i], b[i], orders[i]) for i in range(d)]
rgi = RegularGridInterpolator(pp, V)
t1 = time.time()
rgi(X)
t2 = time.time()
print("Scipy (linear): {}".format(t2 - t1))


# new multilinear
from interp_experiment import vec_interp

grid = ((a[0], b[0], orders[0]), (a[1], b[1], orders[1]))

grid = ((0.0, 1.0, 50), (0.0, 1.0, 50))
res2 = vec_interp(grid, V, X)  # warmup
t2 = time.time()
res2 = vec_interp(grid, V, X)
t3 = time.time()
print("mlinterp (linear): {}".format(t3 - t2))
