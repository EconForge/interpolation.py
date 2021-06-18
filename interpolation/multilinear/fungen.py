import numba
import numpy as np
from numba import float64, int64
from numba import generated_jit, njit
import ast

from numba.extending import overload
from ..compat import Array, Tuple, UniTuple
from ..compat import overload_options


# from math import max, min

####################
# Dimension helper #
####################

t_coord = numba.typeof((2.3, 2.4, 1))  # type of an evenly spaced dimension
t_array = numba.typeof(np.array([4.0, 3.9]))  # type of an unevenly spaced dimension


@njit
def clamp(x, a, b):
    return min(b, max(a, x))


# returns the index of a 1d point along a 1d dimension
@generated_jit(nopython=True)
def get_index(gc, x):
    if gc == t_coord:
        # regular coordinate
        def fun(gc, x):
            δ = (gc[1] - gc[0]) / (gc[2] - 1)
            d = x - gc[0]
            ii = d // δ
            i = int(ii)
            i = clamp(i, 0, gc[2] - 2)
            r = d - i * δ
            λ = r / δ
            return (i, λ)

        return fun
    else:
        # irregular coordinate
        def fun(gc, x):
            N = gc.shape[0]
            i = int(np.searchsorted(gc, x)) - 1
            i = clamp(i, 0, N - 2)
            λ = (x - gc[i]) / (gc[i + 1] - gc[i])
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


@overload(fmap, **overload_options)
def _map(*args):

    if len(args) == 2 and isinstance(args[1], (Tuple, UniTuple)):
        k = len(args[1])
        s = "def __map(f, t): return ({}, )".format(
            str.join(", ", ["f(t[{}])".format(i) for i in range(k)])
        )
    elif len(args) == 3 and isinstance(args[1], (Tuple, UniTuple)):
        k = len(args[1])
        if isinstance(args[2], (Tuple, UniTuple)):
            if len(args[2]) != k:
                # we don't know what to do in this case
                return None
            s = "def __map(f, t1, t2): return ({}, )".format(
                str.join(", ", ["f(t1[{}], t2[{}])".format(i, i) for i in range(k)])
            )
        else:
            s = "def __map(f, t1, x): return ({}, )".format(
                str.join(", ", ["f(t1[{}], x)".format(i, i) for i in range(k)])
            )
    else:
        return None
    d = {}
    eval(compile(ast.parse(s), "<string>", "exec"), d)
    return d["__map"]


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
    assert len(set([e.count for e in t.types])) == 1
    l = t.types[0].count

    def print_tuple(t):
        return "({},)".format(str.join(", ", t))

    tab = [["t[{}][{}]".format(i, j) for i in range(k)] for j in range(l)]
    s = "def funzip(t): return {}".format(print_tuple([print_tuple(e) for e in tab]))
    d = {}
    eval(compile(ast.parse(s), "<string>", "exec"), d)
    return d["funzip"]


#####
# array subscribing:
# when X is a 2d array and I=(i,j) a 2d index, `get_coeffs(X,I)`
# extracts `X[i:i+1,j:j+1]` but represents it as a tuple of tuple, so that
# the number of its elements can be inferred by the compiler
#####


@generated_jit(nopython=True)
def get_coeffs(X, I):
    if X.ndim > len(I):
        print("not implemented yet")
    else:
        from itertools import product

        d = len(I)
        mat = np.array(
            [
                "X[{}]".format(str.join(",", e))
                for e in product(*[(f"I[{j}]", f"I[{j}]+1") for j in range(d)])
            ]
        ).reshape((2,) * d)
        mattotup = (
            lambda mat: mat
            if isinstance(mat, str)
            else "({})".format(str.join(",", [mattotup(e) for e in mat]))
        )
        t = mattotup(mat)
        s = "def get_coeffs(X,I): return {}".format(t)
        dd = {}
        eval(compile(ast.parse(s), "<string>", "exec"), dd)
        return dd["get_coeffs"]
        return fun


# tensor_reduction(C,l)
# (in 2d) computes the equivalent of np.einsum('ij,i,j->', C, [1-l[0],l[0]], [1-l[1],l[1]])`
# but where l is a tuple and C a tuple of tuples.

# this one is a temporary implementation (should be merged with old gen_splines* code)
def gen_tensor_reduction(X, symbs, inds=[]):
    if len(symbs) == 0:
        return "{}[{}]".format(X, str.join("][", [str(e) for e in inds]))
    else:
        h = symbs[0]
        q = symbs[1:]
        exprs = [
            "{}*({})".format(
                h if i == 1 else "(1-{})".format(h),
                gen_tensor_reduction(X, q, inds + [i]),
            )
            for i in range(2)
        ]
        return str.join(" + ", exprs)


@generated_jit(nopython=True)
def tensor_reduction(C, l):
    d = len(l.types)
    ex = gen_tensor_reduction("C", ["l[{}]".format(i) for i in range(d)])
    dd = dict()
    s = "def tensor_reduction(C,l): return {}".format(ex)
    eval(compile(ast.parse(s), "<string>", "exec"), dd)
    return dd["tensor_reduction"]


@generated_jit(nopython=True)
def extract_row(a, n, tup):
    d = len(tup.types)
    dd = {}
    s = "def extract_row(a, n, tup): return ({},)".format(
        str.join(", ", [f"a[n,{i}]" for i in range(d)])
    )
    eval(compile(ast.parse(s), "<string>", "exec"), dd)
    return dd["extract_row"]


# find closest point inside the grid domain
@generated_jit
def project(grid, point):
    s = "def __project(grid, point):\n"
    d = len(grid.types)
    for i in range(d):
        if isinstance(grid.types[i], Array):
            s += f"    x_{i} = min(max(point[{i}], grid[{i}][0]), grid[{i}][grid[{i}].shape[0]-1])\n"
        else:
            s += f"    x_{i} = min(max(point[{i}], grid[{i}][0]), grid[{i}][1])\n"
    s += f"    return ({str.join(', ', ['x_{}'.format(i) for i in range(d)])},)"
    d = {}
    eval(compile(ast.parse(s), "<string>", "exec"), d)
    return d["__project"]
