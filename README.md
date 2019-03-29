# Optimized interpolation routines in Python / numba

[![Build Status](https://travis-ci.org/EconForge/interpolation.py.svg?branch=master)](https://travis-ci.org/EconForge/interpolation.py)

The library contains:
- splines.*: fast numba-compatible multilinear and cubic interpolation
- multilinear.*: fast numba-compatible multilinear interpolation (alternative implementation)
- smolyak.*: smolyak polynomials
- complete polynomials

## install

Latest development version from git:

`pip install git+https://github.com/econforge/interpolation.py.git/`

Or latest stable version:

- from conda: `conda install -c conda-forge interpolation`
- from PyPI: `pip install interpolation`

## multilinear and cubic interpolation

Fast numba-accelerated interpolation routines
for multilinear and cubic interpolation, with any number of dimensions.
Several interfaces are provided.

### eval_linear

Preferred interface for multilinear interpolation. It can interpolate on uniform
and nonuniform cartesian grids. Several extrapolation options are available.


```python
import numpy as np

from interpolation.splines import UCGrid, CGrid, nodes

# we interpolate function
f = lambda x,y: np.sin(np.sqrt(x**2+y**2+0.00001))/np.sqrt(x**2+y**2+0.00001)

# uniform cartesian grid
grid = UCGrid((-1.0, 1.0, 10), (-1.0, 1.0, 10))

# get grid points
gp = nodes(grid)   # 100x2 matrix

# compute values on grid points
values = f(gp[:,0], gp[:,1]).reshape((10,10))

from interpolation.splines import eval_linear
# interpolate at one point
point = np.array([0.1,0.45]) # 1d array
val = eval_linear(grid, values, point)  # float

# interpolate at many points:
points = np.random.random((10000,2))
eval_linear(grid, values, points) # 10000 vector

# output can be preallocated
out = np.zeros(10000)
eval_linear(grid, values, points, out) # 10000 vector

## jitted, non-uniform multilinear interpolation

# default calls extrapolate data by using the nearest value inside the grid
# other extrapolation options can be chosen among NEAREST, LINEAR, CONSTANT

from interpolation.splines import extrap_options as xto
points = np.random.random((100,2))*3-1
eval_linear(grid, values, points, xto.NEAREST)  # 100
eval_linear(grid, values, points, xto.LINEAR)   # 10000 vector
eval_linear(grid, values, points, xto.CONSTANT) # 10000 vector


# one can also approximate on nonuniform cartesian grids

grid = CGrid(np.linspace(-1,1,100)**3, (-1.0, 1.0, 10))

points = np.random.random((10000,2))
eval_linear(grid, values, points) # 10000 vector


# it is also possible to interpolate vector-valued functions in the following way

f = lambda x,y: np.sin(x**3+y**2+0.00001)/np.sqrt(x**2+y**2+0.00001)
g = lambda x,y: np.sin(x**3+y**2+0.00001)/np.sqrt(x**2+y**2+0.00001)
grid = UCGrid((-1.0, 1.0, 10), (-1.0, 1.0, 10))
gp = nodes(grid)   # 100x2 matrix
mvalues = np.concatenate([
   f(gp[:,0], gp[:,1]).reshape((10,10))[:,:,None],
   g(gp[:,0], gp[:,1]).reshape((10,10))[:,:,None]
],axis=2) # 10x10x2 array
points = np.random.random((1000,2))
eval_linear(grid, mvalues, points[:,1]) # 2 elements vector
eval_linear(grid, mvalues, points)      # 1000x2 matrix      
out = np.zeros((1000,2))
eval_linear(grid, mvalues, points, out) # 1000x2 matrix



# finally, the same syntax can be used to interpolate using cubic splines
# one just needs to prefilter the coefficients first
# the same set of options apply but nonuniform grids are not supported (yet)

f = lambda x,y: np.sin(x**3+y**2+0.00001)/np.sqrt(x**2+y**2+0.00001)
grid = UCGrid((-1.0, 1.0, 10), (-1.0, 1.0, 10))
gp = nodes(grid)   # 100x2 matrix
values = f(gp[:,0], gp[:,1]).reshape((10,10))

# filter values
from interpolation.splines import filter_cubic
coeffs = filter_cubic(grid, values) # a 12x12 array

from interpolation.splines import eval_cubic
points = np.random.random((1000,2))
eval_cubic(grid, coeffs, points[:,1]) # 2 elements vector
eval_cubic(grid, coeffs, points)      # 1000x2 matrix      
out = np.zeros((1000,2))
eval_cubic(grid, coeffs, points, out) # 1000x2 matrix

```

*Remark*: the arguably strange syntax for the extapolation option comes from the fact the actualy method called must be determined by type inference. So `eval_linear(..., extrap_method='linear')` would not work because the type of the last argument is always a string. Instead, we use opts.CONSTANT and opts.LINEAR for instance which have different numba types.

Despite what it looks `UCGrid` and `CGRid` are not objects but functions which return very simple python structures that is a tuple of its arguments. For instance, `((0.0,1.0, 10), (0.0,1.0,20))` represents a 2d square discretized with 10 points along the first dimension and 20 along the second dimension. Similarly `(np.array([0.0, 0.1, 0.3, 1.0]), (0.0, 1.0, 20))` represents a square nonuniformly discretized along the first dimension (with 3 points) but uniformly along the second one. Now type dispatch is very sensitive to the exact types (floats vs ints), (tuple vs lists) which is potentially error-prone. Eventually, the functions `UCGrid` and `CGrid` will provide some type check and sensible conversions where it applies. This may change when if a parameterized structure-like object appear in numba.

### interp

Simpler interface. Mimmicks default `scipy.interp`: mutlilinear interpolation with constant extrapolation.


```
### 1d grid
from interpolation import interp

x = np.linspace(0,1,100)**2 # non-uniform points
y = np.linspace(0,1,100)    # values

# interpolate at one point:
interp(x,y,0.5)

# or at many points:
u = np.linspace(0,1,1000)   # points
interp(x,y,u)

```


### object interface

This is for compatibility purpose only, until a new jittable model object is found.

```python
from interpolation.splines import LinearSpline, CubicSpline
a = np.array([0.0,0.0,0.0])         # lower boundaries
b = np.array([1.0,1.0,1.0])         # upper boundaries
orders = np.array([50,50,50])       # 50 points along each dimension
values = np.random.random(orders)   # values at each node of the grid
S = np.random.random((10**6,3))    # coordinates at which to evaluate the splines

# multilinear
lin = LinearSpline(a,b,orders,values)
V = lin(S)
# cubic
spline = CubicSpline(a,b,orders,values) # filter the coefficients
V = spline(S)                       # interpolates -> (100000,) array

```

### development notes

Old, unfair timings: (from `misc/speed_comparison.py`)

```
# interpolate 10^6 points on a 50x50x50 grid.
Cubic: 0.11488723754882812
Linear: 0.03426337242126465
Scipy (linear): 0.6502540111541748
```

More details are available as an example [notebook](https://github.com/EconForge/interpolation.py/blob/master/examples/cubic_splines_python.ipynb) (outdated)

Missing but available soon:
- splines at any order
- derivative

Feasible (some experiments)
- evaluation on the GPU (with numba.cuda)
- parallel evaluation (with guvectorize)



## smolyak interpolation

See [testfile](https://github.com/EconForge/interpolation.py/blob/master/interpolation/smolyak/tests/test_interp.py) for examples.
