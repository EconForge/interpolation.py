import numpy as np
from numba import jit, generated_jit
from numpy import zeros
from numpy import floor

from numba import prange
from .codegen import get_code_spline, source_to_function


import numba
import numpy as np
from numba import njit
from numba.extending import overload
from numba import literally
import numba.types
from numba.core.types.misc import NoneType as none
from numpy import zeros
from numpy import floor
from interpolation.splines.codegen import get_code_spline, source_to_function
from numba.types import UniTuple, float64, Array
from interpolation.splines.codegen import source_to_function
from numba import generated_jit


from ..compat import Tuple, UniTuple
from ..compat import overload_options

#

Ad = np.array(
    [
        #      t^3       t^2        t        1
        [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
        [3.0 / 6.0, -6.0 / 6.0, 0.0 / 6.0, 4.0 / 6.0],
        [-3.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 0.0 / 6.0, 0.0 / 6.0, 0.0 / 6.0],
    ]
)
dAd = zeros((4, 4))
for i in range(1, 4):
    dAd[:, i] = Ad[:, i - 1] * (4 - i)

t_array_1d = [
    Array(float64, 1, a, readonly=b) for b in [True, False] for a in ["A", "C", "F"]
]
t_array_2d = [
    Array(float64, 2, a, readonly=b) for b in [True, False] for a in ["A", "C", "F"]
]

### eval spline (main function)


# @generated_jit(inline='always', nopython=True) # doens't work
@generated_jit(nopython=True)
def allocate_output(G, C, P, O):
    if C.ndim == len(G) + 1:
        # vector valued
        if P.ndim == 2:
            # vectorized call
            if isinstance(O, none):
                return lambda G, C, P, O: np.zeros((P.shape[0], C.shape[C.ndim - 1]))
            else:
                n_o = len(O)
                s = f"lambda G,C,P,O: np.zeros( (P.shape[0], C.shape[C.ndim-1], {n_o}) )"
                return eval(s)
        else:
            if isinstance(O, none):
                return lambda G, C, P, O: np.zeros(P.shape[0])
            else:
                n_o = len(O)
                s = f"lambda G,C,P,O: np.zeros( (P.shape[0], C.shape[C.ndim-1], {n_o}) )"
                return eval(s)
            # points.ndim == 1
    else:
        if P.ndim == 2:
            # vectorized call
            if isinstance(O, none):
                return lambda G, C, P, O: np.zeros(P.shape[0])
            else:
                n_o = len(O)
                s = f"lambda G,C,P,O: np.zeros( (P.shape[0], {n_o}) )"
                return eval(s)
        else:
            if isinstance(O, none):
                raise Exception("Makes no sense")
            else:
                n_o = len(O)
                s = f"lambda G,C,P,O: np.zeros({n_o})"
                return eval(s)  # makes no sense either
            # points.ndim == 1


def _eval_spline():
    pass


@overload(_eval_spline)
def __eval_spline(
    grid, C, points, out=None, order=1, diff="None", extrap_mode="linear"
):

    if not (
        isinstance(order, numba.types.Literal)
        and isinstance(diff, numba.types.Literal)
        and isinstance(extrap_mode, numba.types.Literal)
    ):

        def ugly_workaround(
            grid, C, points, out=None, order=1, diff="None", extrap_mode="linear"
        ):
            return (literally(order), literally(diff), literally(extrap_mode))

        # def __eval_spline(grid, C, points, out=None, order=1, diff="None", extrap_mode='linear'):
        #     return __eval_spline(grid, C, points, out=out, order=literally(order), diff=literally(diff), extrap_mode=literally(extrap_mode))

    kk = (order).literal_value
    diffs = (diff).literal_value
    extrap_ = (extrap_mode).literal_value
    d = len(grid)

    vectorized = points in t_array_2d
    allocate = True
    vector_valued = C.ndim == (len(grid) + 1)

    orders = eval(diffs)

    allocate = isinstance(out, none)  ### strange...

    grid_types = [
        "nonuniform" if isinstance(tt, Array) else "uniform" for tt in grid.types
    ]

    code = get_code_spline(
        d,
        k=kk,
        vectorized=vectorized,
        allocate=allocate,
        vector_valued=vector_valued,
        orders=orders,
        extrap_mode=extrap_,
        grid_types=grid_types,
    )

    context = {
        "floor": floor,
        "zeros": zeros,
        "Cd": Ad,
        "dCd": dAd,
        "allocate_output": allocate_output,
        "np": np,
    }
    f = source_to_function(code, context)

    return f


@njit
def eval_spline(grid, C, points, out=None, order=1, diff="None", extrap_mode="linear"):
    """Do I get a docstring ?"""
    dd = numba.literally(diff)
    k = numba.literally(order)
    extrap_ = numba.literally(extrap_mode)
    return _eval_spline(grid, C, points, out=out, order=k, diff=dd, extrap_mode=extrap_)


###
### Compatibility calls:
###


def _eval_linear():
    pass


from .option_types import options, t_CONSTANT, t_LINEAR, t_NEAREST


@overload(_eval_linear, **overload_options)
def __eval_linear(grid, C, points):
    # print("We allocate with default extrapolation.")
    return lambda grid, C, points: eval_spline(
        grid, C, points, order=1, extrap_mode="linear", diff="None"
    )


@overload(_eval_linear, **overload_options)
def __eval_linear(grid, C, points, extrap_mode):

    # print(f"We are going to extrapolate in {extrap_mode} mode.")
    if extrap_mode == t_NEAREST:
        extrap_ = "nearest"
    elif extrap_mode == t_CONSTANT:
        extrap_ = "constant"
    elif extrap_mode == t_LINEAR:
        extrap_ = "linear"
    else:
        return None

    return lambda grid, C, points, extrap_mode: eval_spline(
        grid, C, points, order=1, diff="None", extrap_mode=extrap_
    )


@overload(_eval_linear, **overload_options)
def __eval_linear(grid, C, points, out, extrap_mode):

    # print(f"We are going to do inplace, with {extrap_mode} extrapolation")
    if extrap_mode == t_NEAREST:
        extrap_ = "nearest"
    elif extrap_mode == t_CONSTANT:
        extrap_ = "constant"
    elif extrap_mode == t_LINEAR:
        extrap_ = "linear"
    else:
        return None
    return lambda grid, C, points, out, extrap_mode: eval_spline(
        grid, C, points, out=out, order=1, diff="None", extrap_mode=extrap_
    )


@overload(_eval_linear, **overload_options)
def __eval_linear(grid, C, points, out):

    return lambda grid, C, points, out: eval_spline(
        grid, C, points, out=out, order=1, diff="None", extrap_mode="linear"
    )


@njit
def eval_linear(*args):
    """Do I get a docstring ?"""
    return _eval_linear(*args)


### Let's be cubic now.


def _eval_cubic():
    pass


from .option_types import options, t_CONSTANT, t_LINEAR, t_NEAREST


@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid, C, points):
    # print("We allocate with default extrapolation.")
    return lambda grid, C, points: eval_spline(
        grid,
        C,
        points,
        order=literally(3),
        extrap_mode=literally("linear"),
        diff=literally("None"),
    )


@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid, C, points, extrap_mode):

    # print(f"We are going to extrapolate in {extrap_mode} mode.")
    if extrap_mode == t_NEAREST:
        extrap_ = literally("nearest")
    elif extrap_mode == t_CONSTANT:
        extrap_ = literally("constant")
    elif extrap_mode == t_cubic:
        extrap_ = literally("cubic")
    else:
        return None

    return lambda grid, C, points, extrap_mode: eval_spline(
        grid, C, points, order=literally(3), diff=literally("None"), extrap_mode=extrap_
    )


@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid, C, points, out, extrap_mode):

    if extrap_mode == t_NEAREST:
        extrap_ = literally("nearest")
    elif extrap_mode == t_CONSTANT:
        extrap_ = literally("constant")
    elif extrap_mode == t_cubic:
        extrap_ = literally("cubic")
    else:
        return None
    return lambda grid, C, points, out, extrap_mode: eval_spline(
        grid,
        C,
        points,
        out=out,
        order=literally(3),
        diff=literally("None"),
        extrap_mode=extrap_,
    )


from numba import literally


@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid, C, points, out):

    return lambda grid, C, points, out: eval_spline(
        grid,
        C,
        points,
        out=out,
        order=literally(3),
        diff=literally("None"),
        extrap_mode=literally("linear"),
    )


@njit
def eval_cubic(*args):
    """Do I get a docstring ?"""
    return _eval_cubic(*args)
