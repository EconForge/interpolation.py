# Optimized interpolation routines in Python

This library provides fast interpolation routines (with Numba and optional Cython)
for the following cases:

## multilinear and cubic splines in any dimension

Here is an example in 3 dimensions:

```python
from interpolation.splines import CubicSpline
a = np.array([0.0,0.0,0.0])         # lower boundaries
b = np.array([1.0,1.0,1.0])         # upper boundaries
orders = np.array([10,10,10])       # 10 points along each dimension
values = np.random.random(orders)   # values at each node of the grid
spline = CubicSpline(a,b,orders,values) # filter the coefficients
S = np.random.random((100000,3))    # coordinates at which to evaluate the spline
V = spline(S)                       # interpolates -> (100000,) array
```

More details are available as an example [notebook](https://github.com/EconForge/interpolation.py/blob/master/examples/cubic_splines_python.ipynb)

Other available features are:
- linear extrapolation
- computation of derivatives
- interpolate many functions at once (multi-splines or vector valued splines)
- evaluation on the GPU, with a few adjustments




## smolyak interpolation

See [testfile](https://github.com/EconForge/interpolation.py/blob/master/interpolation/smolyak/tests/test_interp.py) for examples.
