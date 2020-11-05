# these are compatibility calls

import numpy as np
from numba import njit
from .eval_splines import eval_linear
from .eval_cubic import get_grid

def vec_multilinear_interpolation(smin, smax, orders, mvalues, s, out=None):
    """
    Vec_multipinear_interpolation ( smin mvalues.

    Args:
        smin: (float): write your description
        smax: (int): write your description
        orders: (int): write your description
        mvalues: (array): write your description
        s: (array): write your description
        out: (array): write your description
    """

    grid = get_grid(smin, smax, orders, mvalues[...,0])

    if out is None:
        return eval_linear(grid, mvalues, s)
    else:
        eval_linear(grid, values, s, out)

def multilinear_interpolation(smin, smax, orders, values, s, out=None):
    """
    Multilinear interpolation.

    Args:
        smin: (float): write your description
        smax: (int): write your description
        orders: (todo): write your description
        values: (array): write your description
        s: (array): write your description
        out: (array): write your description
    """

    grid = get_grid(smin, smax, orders, values)

    if out is None:
        return eval_linear(grid, values, s)
    else:
        eval_linear(grid, values, s, out)
