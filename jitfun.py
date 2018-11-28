import numpy as np
import tempita
from numba import jit, generated_jit
from numpy import zeros
from numpy import floor

from codegen import txt, txt_vec, get_values, source_to_function

templ = tempita.Template(txt)
templ_vec = tempita.Template(txt_vec)

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

def _eval_cubic():
    pass

@overload(_eval_cubic)
def __eval_cubic(grid,C,points,out):
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    from math import floor
    from numpy import zeros
    context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}

    if vec_eval:
        code = ( templ_vec.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=False) )[1:]
    else:
        code = ( templ.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=False) )[1:]
    print(code)
    f = source_to_function(code, context)
    return f


@overload(_eval_cubic)
def __eval_cubic(grid,C,points):
    d = len(grid)
    n_x = len(grid.types)
    vector_valued = (C.ndim==d+1)
    vec_eval = (points.ndim==2)
    from math import floor
    from numpy import zeros
    context = {'floor': floor, 'zeros': zeros, 'Cd': Ad, 'dCd': dAd}

    if vec_eval:
        code = ( templ_vec.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=True) )[1:]
    else:
        code = ( templ.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=True) )[1:]
    print(code)
    f = source_to_function(code, context)
    return f

@njit
def eval_cubic(*args):
    return _eval_cubic(*args)

# print(templ.substitute(d=2, vector_valued=2, get_values=get_values, allocate=True))
# print(templ.substitute(d=2, vector_valued=2, get_values=get_values, allocate=False))
# print(templ.substitute(d=2, vector_valued=False, get_values=get_values, allocate=False))
#
#
# print(templ_vec.substitute(d=2, vector_valued=False, get_values=get_values, allocate=False))
# print(templ_vec.substitute(d=2, vector_valued=False, get_values=get_values, allocate=True))
#
# print(templ_vec.substitute(d=2, vector_valued=True, get_values=get_values, allocate=True))


N = 1000
grid = (
    (0.0, 1.0, 100),
    (0.0, 1.0, 100)
)
C = np.random.random((102,102))
CC = np.random.random((102,102,2))
points = np.random.random((N,2))
out = np.random.random((N))
Cout = np.random.random((N,2))





eval_cubic(grid, C, points, out)
# eval_cubic(grid, C, points[0,:], out[0,0])
eval_cubic(grid, CC, points, Cout)
eval_cubic(grid, CC, points[0,:], Cout[0,:])


eval_cubic(grid, C, points)
eval_cubic(grid, C, points[0,:])
eval_cubic(grid, CC, points)
eval_cubic(grid, CC, points[0,:])




####
###
###


N=100000
K = 100
d = 3
a = np.array([0.0]*d)
b = np.array([1.0]*d)
n = np.array([K]*d)
grid = ((0.0,1.0,K),)*d

CC = np.random.random((K+2,K+2,K+2,5))
point = np.random.random(3)
points = np.random.random((N,3))
vval = np.zeros(5)
vval1 = np.zeros(5)
vals = np.zeros((N,5))
vals1 = np.zeros((N,5))
from interpolation.splines.eval_cubic import eval_cubic_splines_3, vec_eval_cubic_splines_3

eval_cubic_splines_3(a,b,n,CC,point,vval)
eval_cubic(grid, CC, point, vval1)
abs(vval-vval1).max()

vec_eval_cubic_splines_3(a,b,n,CC,points,vals)
eval_cubic(grid, CC, points, vals1)
dd = vals1-vals
abs(dd[~np.isnan(dd)]).max()


vvals2 = eval_cubic(grid,CC,point)
vals2 = eval_cubic(grid,CC,points)

dd2 = vals2-vals
abs(dd2).max()
