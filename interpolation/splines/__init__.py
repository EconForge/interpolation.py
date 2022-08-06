from .splines import CubicSpline, CubicSplines
from .multilinear import LinearSpline, LinearSplines
from .eval_splines import options, eval_linear, eval_cubic, eval_spline
from .prefilter_cubic import filter_cubic, prefilter
from .option_types import options as extrap_options

import numba
import numpy as np

# dummy functions
def UCGrid(*args):
    tt = numba.typeof((10.0, 1.0, 1))
    for a in args:
        assert numba.typeof(a) == tt
        min, max, n = a
        assert min < max
        assert n > 1

    return tuple(args)


def CGrid(*args):
    tt = numba.typeof((10.0, 1.0, 1))
    for a in args:
        if isinstance(a, np.ndarray):
            assert a.ndim == 1
            assert a.shape[0] >= 2
        elif numba.typeof(a) == tt:
            min, max, n = a
            assert min < max
            assert n > 1
        else:
            raise Exception(f"Unknown dimension specification: {a}")

    return tuple(args)


def nodes(grid):
    from interpolation.cartesian import mlinspace

    return mlinspace([g[0] for g in grid], [g[1] for g in grid], [g[2] for g in grid])
