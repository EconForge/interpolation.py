from numba.extending import overload
import numba
import numpy as np
from numba import float64, int64
from numba import generated_jit, njit
import ast

from numba.types.containers import Tuple, UniTuple

# the following code implements a function
#
# interpolate(grid, c, u)
#
# where grid is a cartesian grid, but dimensions are not necessarly evenly spaced
# grid are represented by a tuple like:
# - `( (-1.0,1.0,10), (-1.0,1.0,20) )` : a `10x20` even cartesian grid on `[-1,1]^2`
# - `( linspace(0,1,10), linspace(0,1,100)**2)` : a `10x100` uneven cartesian grid on `[0,1]^2`
# - `( (0,1.0,10), linspace(0,1,100)**2)` : a `10x100` cartesian grid where first dimension is evenly distributed, the second not

# there is only one (easy-to-read?) jitted implementation of `interpolate`, line 185
# it depends on several generated jit functions which dispatch the right behaviour
# in this example this helper functions are written by hand, but for any dimension
# the code could be generated just in time too.


####################
# Dimension helper #
####################

t_coord = numba.typeof((2.3,2.4,1))           # type of an evenly spaced dimension
t_array = numba.typeof(np.array([4.0, 3.9]))  # type of an unevenly spaced dimension

# returns the index of a 1d point along a 1d dimension
@generated_jit(nopython=True)
def get_index(gc, x):
    if gc == t_coord:
        # regular coordinate
        def fun(gc,x):
            δ = (gc[1]-gc[0])/(gc[2]-1)
            d = x-gc[0]
            ii = d // δ
            r = d-ii*δ
            i = int(ii)
            λ = r/δ
            return (i, λ)
        return fun
    else:
        # irregular coordinate
        def fun(gc,x):
            i = int(np.searchsorted(gc, x))-1
            λ = (x-gc[i])/(gc[i+1]-gc[i])
            return (i, λ)
        return fun

# returns number of dimension of a dimension
@generated_jit(nopython=True)
def get_size(gc):
    if gc == t_coord:
        # regular coordinate
        def fun(gc):
            return gc[2]
        return fun
    else:
        # irregular coordinate
        def fun(gc):
            return len(gc)
        return fun

#####################
# Generator helpers #
#####################

# the next functions replace the use of generators, with the difference that the
# output is a tuple, which dimension is known by the jit guy.

# example:
# ```
# def f(x): x**2
# fmap(f, (1,2,3)) -> (1,3,9)
# def g(x,y): x**2 + y
# fmap(g, (1,2,3), 0.1) -> (1.1,3.1,9.1)  # (g(1,0.1), g(2,0.1), g(3,0.1))
# def g(x,y): x**2 + y
# fmap(g, (1,2,3), (0.1,0.2,0.3)) -> (1.1,3.0.12,9.3)
# ```


def fmap():
    pass

@overload(fmap)
def _map(*args):

    if len(args)==2 and isinstance(args[1], (Tuple, UniTuple)):
        k = len(args[1])
        s = "def __map(f, t): return ({}, )".format(str.join(', ',['f(t[{}])'.format(i) for i in range(k)]))
    elif len(args)==3 and isinstance(args[1], (Tuple, UniTuple)):
        k = len(args[1])
        if isinstance(args[2], (Tuple, UniTuple)):
            if len(args[2]) != k:
                # we don't know what to do in this case
                return None
            s = "def __map(f, t1, t2): return ({}, )".format(str.join(', ',['f(t1[{}], t2[{}])'.format(i,i) for i in range(k)]))
        else:
            s = "def __map(f, t1, x): return ({}, )".format(str.join(', ',['f(t1[{}], x)'.format(i,i) for i in range(k)]))
    else:
        return None
    d = {}
    eval(compile(ast.parse(s),'<string>','exec'), d)
    return d['__map']


# not that `fmap` does nothing in python mode...
# an alternative would be
#
# @njit
# def _fmap():
#     pass
#
# @overload(_fmap)
# ...
#
# @njit
# def fmap(*args):
#     return _fmap(*args)
#
# but this seems to come with a performance cost.
# It it is also possible to overload `map` but we would risk
# a future conflict with the map api.


#
# @njit
# def fmap(*args):
#     return _fmap(*args)
#
# funzip(((1,2), (2,3), (4,3))) -> ((1,2,4),(2,3,3))

@generated_jit(nopython=True)
def funzip(t):
    k = t.count
    assert(len(set([e.count for e in t.types]))==1)
    l = t.types[0].count
    def print_tuple(t): return "({},)".format(str.join(", ", t))
    tab =  [ [ 't[{}][{}]'.format(i,j) for i in range(k)] for j in range(l) ]
    s = "def funzip(t): return {}".format(print_tuple( [print_tuple(e) for e in tab] ))
    d = {}
    eval(compile(ast.parse(s),'<string>','exec'), d)
    return d['funzip']


#####
# array subscribing:
# when X is a 2d array and I=(i,j) a 2d index, `get_coeffs(X,I)`
# extracts `X[i:i+1,j:j+1]` but represents it as a tuple of tuple, so that
# the number of its elements can be inferred by the compiler
#####

@generated_jit(nopython=True)
def get_coeffs(X,I):
    if X.ndim>len(I):
        print("not implemented yet")
    else:
        if len(I)==1:
            def fun(X,I): return ((X[I[0]],X[I[0]+1]),)
        elif len(I)==2:
            def fun(X,I): return (
                    (X[I[0],  I[1]],X[I[0],  I[1]+1]),
                    (X[I[0]+1,I[1]],X[I[0]+1,I[1]+1]),
                )
        else:
            print("Not implemented")
        return fun

# tensor_reduction(C,l)
# (in 2d) computes the equivalent of np.einsum('ij,i,j->', C, l[1], l[2])`
# but where C and l are given as list of tuples.

@generated_jit(nopython=True)
def tensor_reduction(C,l):
    if len(l)==1:
        def fun(C,l): return (1-l[0])*C[0] + l[0]*C[1]
    elif len(l)==2:
        def fun(C,l): return (1-l[0])*((1-l[1])*C[0][0]+l[1]*C[0][1]) + l[0]*((1-l[1])*C[1][0]+l[1]*C[1][1])
    else:
        print("Not implemented")
    return fun# funzip(((1,2), (2,3), (4,3)))n

#################################
# Actual interpolation function #
#################################

from typing import Tuple


@njit
def interp(grid: Tuple, c, u: Tuple)->float:

    # get indices and barycentric coordinates
    tmp = fmap(get_index, grid, u)
    indices, barycenters = funzip(tmp)
    coeffs = get_coeffs(c, indices)
    v = tensor_reduction(coeffs, barycenters)
    return v


grid = (
    (0.0, 1.0, 11),
    (0.0, 1.0, 11)
)

vv = np.linspace(0,1,11)

grid_uneven = (
    vv,
    vv
)

C = np.random.rand(11,11)

# two equivalent calls:
v = interp(grid, C, (0.3, 0.2))
v_unevn = interp(grid_uneven, C, (0.3, 0.2))

assert(abs(v_unevn-v)<1e-10)


#
# # let's compare with interp2d
from scipy.interpolate import interp2d
intp2 = interp2d(vv,vv,C.T)
v_2d = intp2(0.3,0.2)
assert(abs(v_2d-v)<1e-10)

# and Regular Grid Interpolator
from scipy.interpolate import RegularGridInterpolator
vg = np.linspace(0,1,11)
rgi = RegularGridInterpolator((vg,vg),C)
v_rgi = rgi([0.3, 0.2])[0]
assert(abs(v_rgi-v)<1e-10)


###############################################################
# Now let's see what are the gains of jit for repeated callas #
# with some unscientific performance benchmarks               #
###############################################################
@njit
def vec_interp(grid, C, points):
    N = points.shape[0]
    out = np.zeros(N)
    for n in range(N):
        p1 = points[n,0]
        p2 = points[n,1]
        out[n] = interp(grid, C, (p1,p2))
    return out

N = 100000
points = np.random.rand(N,2)



vals = vec_interp(grid, C, points)
vals_un = vec_interp(grid_uneven, C, points)
vals_rgi = rgi(points)

# both give the same result
assert((abs(vals-vals_rgi).max()<1e-10))
assert((abs(vals-vals_un).max()<1e-10))

import time
K = 1000

t1_a = time.time()
for k in range(K):
    vals = vec_interp(grid, C, points)
t1_b = time.time()

t2_a = time.time()
for k in range(K):
    vals_un = vec_interp(grid_uneven, C, points)
t2_b = time.time()

t3_a = time.time()
for k in range(K):
    vals_rgi = rgi(points)
t3_b = time.time()

print(f"Even: {t1_b-t1_a}")
print(f"Uneven: {t2_b-t2_a}")
print(f"Scipy: {t3_b-t3_a}")
