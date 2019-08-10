import numba

from numba import jitclass


import numpy as np
from numba import njit
from interpolation.splines import eval_linear, UCGrid, nodes

f = lambda x,y: np.sin(x**3+y**2+0.00001)/np.sqrt(x**2+y**2+0.00001)
g = lambda x,y: np.sin(x**3+y**2+0.00001)/np.sqrt(x**2+y**2+0.00001)

grid = UCGrid((-1.0, 1.0, 10), (-1.0, 1.0, 10))
gp = nodes(grid)   # 100x2 matrix

mvalues = np.concatenate([
   f(gp[:,0], gp[:,1]).reshape((10,10))[:,:,None],
   g(gp[:,0], gp[:,1]).reshape((10,10))[:,:,None]
],axis=2) # 10x10x2 array

points = np.random.random((1000,2))


from interpolation.splines.option_types import options as xto

@njit
def fun():
    eval_linear(grid, mvalues, points)

@njit
def no_fun():
    eval_linear(grid, mvalues, points, xto.LINEAR)

fun()       # works happily
no_fun()    # does not :'(
