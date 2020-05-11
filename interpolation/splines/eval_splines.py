import numpy as np
from numba import jit, generated_jit
from numpy import zeros
from numpy import floor

from numba import prange
from .codegen import get_code_linear, get_code_cubic, source_to_function

from ..compat import Tuple, UniTuple
from ..compat import overload_options

#

Ad = np.array([
#      t^3       t^2        t        1
   [-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0],
   [ 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0],
   [-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0],
   [ 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0]
])
dAd = zeros((4,4))
for i in range(1,4):
    dAd[:,i] = Ad[:,i-1]*(4-i)

# @generated_jit(nopython=True)
# def v_eval_cubic(grid,C,points,out):
#     d = len(grid.types)
#     vector_valued = (C.ndim==d+1)
#     context = {'floor': floor,'Cd': Ad, 'dCd': dAd}
#     code = ( templ_vec.substitute(d=d, vector_valued=vector_valued, get_values=get_values) )
#     f = source_to_function(code, context=context)
#     return f

from numba import njit
from numba.extending import overload

# def _eval_cubic():
#     pass
#
# @overload(_eval_cubic)
# def __eval_cubic(grid,C,points,out):
#     d = len(grid)
#     n_x = len(grid.types)
#     vector_valued = (C.ndim==d+1)
#     vec_eval = (points.ndim==2)
#     from math import floor
#     from numpy import zeros
#     context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}
#
#     print("Compiling nonallocating cubic code")
#     code = get_code_cubic(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=False)
#     print(code)
#     f = source_to_function(code, context)
#     return f
#
#
# @overload(_eval_cubic)
# def __eval_cubic(grid,C,points):
#     d = len(grid)
#     n_x = len(grid.types)
#     vector_valued = (C.ndim==d+1)
#     vec_eval = (points.ndim==2)
#     from math import floor
#     from numpy import zeros
#     context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}
#
#     print("Compiling allocating cubic code")
#     code = get_code_cubic(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=True)
#
#     f = source_to_function(code, context)
#     return f
#
# @njit
# def eval_cubic(*args):
#     return _eval_cubic(*args)

import numba
import numpy as np
from numba import njit
from numba.extending import overload
from ..compat import Array

def _eval_linear():
    pass

from .option_types import options, t_CONSTANT, t_LINEAR, t_NEAREST

@overload(_eval_linear, **overload_options)
def __eval_linear(grid,C,points):
    # print("We allocate with default extrapolation.")
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    from math import floor
    from numpy import zeros
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'np': np} #, 'Cd': Ad, 'dCd': dAd}
    code = get_code_linear(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=True, grid_types=grid_types)
    # print(code)
    f = source_to_function(code, context)
    return f

@overload(_eval_linear, **overload_options)
def __eval_linear(grid,C,points,extrap_mode):

    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    from math import floor
    from numpy import zeros
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'np': np} #, 'Cd': Ad, 'dCd': dAd}
    # print(f"We are going to extrapolate in {extrap_mode} mode.")
    if extrap_mode == t_NEAREST:
        extrap_ = 'nearest'
    elif extrap_mode == t_CONSTANT:
        extrap_ = 'constant'
    elif extrap_mode == t_LINEAR:
        extrap_ = 'linear'
    else:
        return None
    code = get_code_linear(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=True, grid_types=grid_types, extrap_mode=extrap_)
    f = source_to_function(code, context)
    return f



@overload(_eval_linear, **overload_options)
def __eval_linear(grid,C,points,out,extrap_mode):

    # print(f"We are going to do inplace, with {extrap_mode} extrapolation")
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'np': np} #, 'Cd': Ad, 'dCd': dAd}
    if extrap_mode == t_NEAREST:
        extrap_ = 'nearest'
    elif extrap_mode == t_CONSTANT:
        extrap_ = 'constant'
    elif extrap_mode == t_LINEAR:
        extrap_ = 'linear'
    else:
        return None
    code = get_code_linear(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=False, grid_types=grid_types, extrap_mode=extrap_)
    # print(code)
    f = source_to_function(code, context)
    return f


@overload(_eval_linear, **overload_options)
def __eval_linear(grid,C,points,out):

    # print("We are going to do inplace, with default extrapolation")
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'np': np} #, 'Cd': Ad, 'dCd': dAd}
    code = get_code_linear(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=False, grid_types=grid_types)
    # print(code)
    f = source_to_function(code, context)
    return f

@njit
def eval_linear(*args):
    """Do I get a docstring ?"""
    return _eval_linear(*args)


### Let's be cubic now.



def _eval_cubic():
    pass

from .option_types import options, t_CONSTANT, t_LINEAR, t_NEAREST

@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid,C,points):
    # print("We allocate with default extrapolation.")
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    from math import floor
    from numpy import zeros
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}
    code = get_code_cubic(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=True, grid_types=grid_types)
    # print(code)
    f = source_to_function(code, context)
    return f

@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid,C,points,extrap_mode):

    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    from math import floor
    from numpy import zeros
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}

    # print(f"We are going to extrapolate in {extrap_mode} mode.")
    if extrap_mode == t_NEAREST:
        extrap_ = 'nearest'
    elif extrap_mode == t_CONSTANT:
        extrap_ = 'constant'
    elif extrap_mode == t_LINEAR:
        extrap_ = 'linear'
    else:
        return None
    code = get_code_cubic(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=True, grid_types=grid_types, extrap_mode=extrap_)
    # print(code)
    f = source_to_function(code, context)
    return f



@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid,C,points,out,extrap_mode):

    # print(f"We are going to do inplace, with {extrap_mode} extrapolation")
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}
    if extrap_mode == t_NEAREST:
        extrap_ = 'nearest'
    elif extrap_mode == t_CONSTANT:
        extrap_ = 'constant'
    elif extrap_mode == t_LINEAR:
        extrap_ = 'linear'
    else:
        return None
    code = get_code_cubic(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=False, grid_types=grid_types, extrap_mode=extrap_)
    # print(code)
    f = source_to_function(code, context)
    return f


@overload(_eval_cubic, **overload_options)
def __eval_cubic(grid,C,points,out):

    # print("We are going to do inplace, with default extrapolation")
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    grid_types = ['nonuniform' if isinstance(tt, Array) else 'uniform' for tt in grid.types]
    context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}
    code = get_code_cubic(d, vector_valued=vector_valued, vectorized=vec_eval, allocate=False, grid_types=grid_types)
    # print(code)
    f = source_to_function(code, context)
    return f





@njit
def eval_cubic(*args):
    """Do I get a docstring ?"""
    return _eval_cubic(*args)
