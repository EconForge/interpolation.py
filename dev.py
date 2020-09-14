from interpolation.splines.codegen import get_code_spline

from interpolation.splines.eval_splines import eval_spline


# print( get_code_spline(1, vectorized=False, vector_valued=False, allocate=False, orders=None) )
# print( get_code_spline(1, vectorized=False, vector_valued=False, allocate=True, orders=None) )# orders=((0,),(1,)))

# print( get_code_spline(1, vectorized=False, vector_valued=True, allocate=False, orders=None) ) # this makes no sense
# print( get_code_spline(1, vectorized=False, vector_valued=True, allocate=True, orders=None) )




orders = ((0,), (1,))
# print( get_code_spline(1, vectorized=False, vector_valued=False, allocate=False, orders=orders) )
# print( get_code_spline(1, vectorized=False, vector_valued=False, allocate=True, orders=orders) )# orders=((0,),(1,)))

# print( get_code_spline(1, vectorized=False, vector_valued=True, allocate=False, orders=orders) ) # this makes no sense
# print( get_code_spline(1, vectorized=False, vector_valued=True, allocate=True, orders=orders) )

import numpy as np

grid = ((0.0, 1.0, 50),)
f = np.sin
vals = f(np.linspace(0,1,50))

fgrid = np.linspace(0,1,50)[:,None]

out = fgrid.ravel()*0

# print( eval_linear(grid, vals, fgrid, out) - vals )
# print( eval_linear(grid, vals, fgrid) - vals )


print( eval_spline(1, grid, vals, fgrid) - vals )
