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


#################################
# Actual interpolation function #
#################################

from .fungen import (
    fmap,
    funzip,
    get_coeffs,
    tensor_reduction,
    get_index,
    extract_row,
    project,
)

from numba import njit
from typing import Tuple

from ..compat import UniTuple, Tuple, Float, Integer, Array

Scalar = (Float, Integer)

import numpy as np
from numba import generated_jit

# logic of multilinear interpolation


@generated_jit
def mlinterp(grid, c, u):
    if isinstance(u, UniTuple):

        def mlininterp(grid: Tuple, c: Array, u: Tuple) -> float:
            # get indices and barycentric coordinates
            tmp = fmap(get_index, grid, u)
            indices, barycenters = funzip(tmp)
            coeffs = get_coeffs(c, indices)
            v = tensor_reduction(coeffs, barycenters)
            return v

    elif isinstance(u, Array) and u.ndim == 2:

        def mlininterp(grid: Tuple, c: Array, u: Array) -> float:
            N = u.shape[0]
            res = np.zeros(N)
            for n in range(N):
                uu = extract_row(u, n, grid)
                # get indices and barycentric coordinates
                tmp = fmap(get_index, grid, uu)
                indices, barycenters = funzip(tmp)
                coeffs = get_coeffs(c, indices)
                res[n] = tensor_reduction(coeffs, barycenters)
            return res

    else:
        mlininterp = None
    return mlininterp


### The rest of this file constrcts function `interp`

from collections import namedtuple

itt = namedtuple("InterpType", ["d", "values", "eval"])


def detect_types(args):

    dims = [e.ndim if isinstance(e, Array) else -1 for e in args]

    md = max(dims)

    if len(args) == 3:
        d = 1
        i_C = 1
    else:
        i_C = dims.index(md)  # index of coeffs
        d = i_C

    if args[i_C].ndim == d:
        values_type = "scalar"
    else:
        assert args[i_C].ndim == d + 1
        values_type = "vector"

    eval_args = args[(i_C + 1) :]

    if len(eval_args) >= 2:
        if set([isinstance(e, Array) for e in eval_args]) == set([True]):
            eval_type = "cartesian"
            assert set([e.ndim for e in eval_args]) == set([1])
        elif set([isinstance(e, Scalar) for e in eval_args]) == set([True]):
            eval_type = "scalar"
        else:
            raise Exception("Undetected evaluation type.")
    else:
        if isinstance(eval_args[0], Array):
            if eval_args[0].ndim == 1:
                eval_type = "point"
            elif eval_args[0].ndim == 2:
                eval_type = "vectorized"
            else:
                raise Exception("Undetected evaluation type.")
        elif isinstance(eval_args[0], UniTuple):
            eval_type = "tuple"
        elif set([isinstance(e, Scalar) for e in eval_args]) == set([True]):
            eval_type = "scalar"
        else:
            raise Exception("Undetected evaluation type.")

    return itt(d, values_type, eval_type)


def make_mlinterp(it, funname):

    if it.values == "vector":
        return None

    if it.eval in ("scalar", "tuple") and it.values == "vector":
        # raise Exception("Non supported. (return type unknown)")
        return None

    # grid = str.join(',', ['args[{}]'.format(i) for i in range(it.d)])
    grid_s = "({},)".format(str.join(",", [f"args[{i}]" for i in range(it.d)]))
    if it.eval in ("scalar", "point", "tuple"):
        if it.eval == "scalar":
            point_s = "({},)".format(
                str.join(",", [f"args[{it.d+i+1}]" for i in range(it.d)])
            )
            # point_s = f"(args[{d+1}])"
        elif it.eval == "tuple":
            point_s = f"args[{it.d+1}]"
        else:
            point_s = "({},)".format(
                str.join(",", [f"args[{it.d+1}][{i}]" for i in range(it.d)])
            )

        source = f"""\
def {funname}(*args):
    grid = {grid_s}
    C = args[{it.d}]
    point = {point_s}
    ppoint = project(grid, point)
    res = mlinterp(grid, C, ppoint)
    return res
    """
        return source
    elif it.eval == "vectorized":
        p_s = "({},)".format(str.join(",", [f"points[n,{i}]" for i in range(it.d)]))
        source = f"""\
from numpy import zeros
def {funname}(*args):
    grid = {grid_s}
    C = args[{it.d}]
    points = args[{it.d+1}]
    N = points.shape[0]
    res = zeros(N)
    # return res
    for n in range(N):
        ppoint = project(grid, {p_s})
        res[n] = mlinterp(grid, C, ppoint)
    return res
"""
        return source

    elif it.eval == "cartesian":
        if it.d == 1:
            source = f"""
from numpy import zeros
def {funname}(*args):
    grid = {grid_s}
    C = args[{it.d}]
    points_x = args[2]
    N = points_x.shape[0]
    res = zeros(N)
    for n in range(N):
        ppoint = project(grid,(points_x[n],))
        res[n] = mlinterp(grid, C, ppoint)
    return res
"""
        elif it.d == 2:
            source = f"""
from numpy import zeros
def {funname}(*args):
    grid = {grid_s}
    C = args[{it.d}]
    points_x = args[3]
    points_y = args[4]
    N = points_x.shape[0]
    M = points_y.shape[0]
    res = zeros((N,M))
    for n in range(N):
        for m in range(M):
            ppoint = project(grid,(points_x[n], points_y[m]))
            res[n,m] = mlinterp(grid, C, ppoint)
    return res
"""
        else:
            return None
        return source


from numba import generated_jit


@generated_jit(nopython=True)
def interp(*args):

    aa = args[0].types

    it = detect_types(aa)
    if it.d == 1 and it.eval == "point":
        it = itt(it.d, it.values, "cartesian")
    source = make_mlinterp(it, "__mlinterp")
    import ast

    tree = ast.parse(source)
    code = compile(tree, "<string>", "exec")
    eval(code, globals())
    return __mlinterp
