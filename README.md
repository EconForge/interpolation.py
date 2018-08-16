# Optimized interpolation routines in Python / numba

[![Build Status](https://travis-ci.org/EconForge/interpolation.py.svg?branch=master)](https://travis-ci.org/EconForge/interpolation.py)

This library provides fast numba-accelerated interpolation routines
for the following cases:

## multilinear and cubic splines in any dimension

Here is an example in 3 dimensions:

```python
from interpolation.splines import LinearSpline, CubicSpline
a = np.array([0.0,0.0,0.0])         # lower boundaries
b = np.array([1.0,1.0,1.0])         # upper boundaries
orders = np.array([50,50,50])       # 10 points along each dimension
values = np.random.random(orders)   # values at each node of the grid
S = np.random.random((10^6,3))    # coordinates at which to evaluate the splines

# multilinear
lin = LinearSpline(a,b,orders,values)
V = lin(S)
# cubic
spline = CubicSpline(a,b,orders,values) # filter the coefficients
V = spline(S)                       # interpolates -> (100000,) array

```

Unfair timings: (from `misc/speed_comparison.py`)
```
# interpolate 10^6 points on a 50x50x50 grid.
Cubic: 0.11488723754882812
Linear: 0.03426337242126465
Scipy (linear): 0.6502540111541748
```

More details are available as an example [notebook](https://github.com/EconForge/interpolation.py/blob/master/examples/cubic_splines_python.ipynb)

Other available features are:
- linear extrapolation
- computation of derivatives
- interpolate many functions at once (multi-splines or vector valued splines)

Experimental
- evaluation on the GPU (with numba.cuda)
- parallel evaluation (with guvectorize)

In the near future:
- JIT classes for all interpolation objects


## jitted, non-uniform multilinear interpolation

There is a simple `interp` function with a flexible API which does multinear on uniform or non uniform cartesian grids.

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






## smolyak interpolation

See [testfile](https://github.com/EconForge/interpolation.py/blob/master/interpolation/smolyak/tests/test_interp.py) for examples.
