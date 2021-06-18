# these are compatibility calls

import numpy as np
from numba import njit
from .eval_splines import eval_linear
from .eval_cubic import get_grid


def vec_multilinear_interpolation(smin, smax, orders, mvalues, s, out=None):

    grid = get_grid(smin, smax, orders, mvalues[..., 0])

    if out is None:
        return eval_linear(grid, mvalues, s)
    else:
        eval_linear(grid, values, s, out)


def multilinear_interpolation(smin, smax, orders, values, s, out=None):

    grid = get_grid(smin, smax, orders, values)

    if out is None:
        return eval_linear(grid, values, s)
    else:
        eval_linear(grid, values, s, out)
