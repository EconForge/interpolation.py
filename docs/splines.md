# Splines interpolation

Spline interpolation can be performed using two functions `prefilter` and `eval_splines`.

```from interpolation.splines import prefilter, eval_splines```

These two functions are jit compiled using numba, and should be fast, without big performance loss when used in a loop provided the loop is compiled. 

The behaviour of these two functions is determined by the type of their argument. In the text below the types provided are numba types obtained using `numba.typeof` function.

## grids

Splines are defined by their values on a cartesian product of 1-dimensional grids. 

*Regular one dimensional grids*, are represented by a `
Tuple(float64, float64, int64)` where first element is the lower bound, the second the upper bound and the latest the number of points.

*An irregular one dimensional grid*, is represented by a numpy array  `array(float64, 1d, C)`. It is assumed to be in increasing order. These are currently supported only for multilinear splines (`k=1`)

A multidimensional grid, is represented as a tuple of one-dimensional grids. For instance, `((0.0, 1.0, 10), (0.0, 1.0, 10))`. Pay attention to one element tuples: `(np.array([0.0, 0.1, 0.5]))` is not a correct grid (it is one-dimensional). It should be `(np.array([0.0, 0.1, 0.5]),)`


## values

Given a d-dimensional grid of dimensions $n_1 \times ... \times n_d$. Values to be interpolated can be specified as:

- a $n_1 \times ... \times n_d$  numpy array: *scalar values*
- a $n_1 \times ... \times n_d \times n_x$  numpy array: *vector values* where $n_x$ variables are defined on each node of the grid

## prefiltering variables

For splines of order greater than 1, the coefficients used for interpolation must be prefiltered, for the resulting interpolant to inteporpolate exactly at grid points:

```python
C = prefilter(G, V, k=3)
```

where `G` is a grid and `V` the values as described above.Currently, `prefilter` is implemented only for `k=3` and `k=1`. In the `k=1` it does nothing as prefiltering is not required.

The coefficient array is of size $(n_1+k-1)\times ... \times (n_d+k-1)$ for scalar values and $(n_1+k-1)\times ... \times (n_d+k-1) \times n_x$ for vector values.

Inplace calculations can be performed as:

```python
prefilter(G, V, k=3, out=C)
```

Currently, prefiltering cubic splines, always used natural boundaries conditions (f''=0). 

## interpolating the function

To interpolate values:

```python
eval_spline(G, C, P, out=None, k=3, diff="None", extrap_mode="linear")
```

where:

- `G` is a multi-dimensional grid as specified above
- `C` an array of coefficients
- `P` denotes the locations at which to interpolate:
    - tuple or array of size `d`: point at which to interpolate
    - 2d array with column size `d`: *list* of points at which to interpolate
- `out`: 
    - if `None`, interpolated values are returned
    - if array: where to store the result inplace. The dimensions must be exactly equal to the array that would be returned by the function (see below)
- `k: int`: spline order (currently (1 or 3))
- `extrap_mode: str`: how to extrapolate outside of the grid. Either:
    - '"constant"': 0 outside of the grid
    - '"nearest"': takes value from nearest point on the grid
    - '"linear"' (default): projects to nearest points and use derivative at this point to extrapolate linearly
- `diff: str`: specifies which derivatives to compute
    - '"None"': no derivative
    - string representing a tuple of tuple (see below)

### just in time compilation and literal values

In the current  `eval_spline` specification, keyword arguments `k`, `extrap_mode`, and `diff`, are used to control the generation of just in time code and must therefore be known at compile-time. They are ultimately treated as literal values (1 and 3 are of different types for instance.)
This currently implies a few limitations:

- `diff` must be passed as strings even though it represents a tuple of tuples
- keyword arguments cannot be ommited. Error message is especially confusing, when they are.
- there might be a penalty cost in running this function outside of a numba jitted context. It is a known limitation in numba ~0.50, which will ultimately go away


### specifying partial derivatives

Provided the interpolated function $f(x)$ is defined on a `d` dimensional space, with arguments `$x_1, ... x_d$`. A partial derivative of any order $(\partial^{k_1} ... \partial^{k_d})$ is denoted by d-element tuple `(k_1, ..., k_d)`.
The partial derivatives to be computed  by `eval_spline` can be specified as a tuple of tuples.

For instance, in a two dimensional space, to compute the value and the jacobian, one can pass:

```( (0,0), (1,0), (0,1) )```

Note that we don't pass the tuple directly to `eval_spline` but a string. For the same example we would then pass:

```"( (0,0), (1,0), (0,1) )"```


## dimensions of the output

Depending on the the type of arguments, the output (or the supplied array to perform inplace operations will have different dimensions).

Here is a summary. When relevant `n_x` denotes the number of approximated variable (for vector valued interpolation), `N` the number of points where the interpolant is approximated (for vectorized operations), `n_j` the number of partial derivatives to evaluate.

| Vectorized    | Vector Valued  | Derivatives      | Output                     |
|---------------|----------------|-------------- ---|----------------------------|
|      no       |    no          |      no          |   `float` (no inplace)     |
|      no       |    `n_x`       |      no          |   `n_x`  array             |
|      `N`      |    no          |      no          |   `N` array                |
|      `N`      |    `n_x`       |      no          |   `N.n_x` array            |
|      no       |     no         |      `n_j`       |   `n_j` tuple (no inplace) |
|      no       |    `n_x`       |      `n_j`       |   `n_x.n_j`  array         |
|      `N`      |    no         |      `n_j`       |   `N . n_j` array          |
|      `N`      |    `n_x`       |      `n_j`       |   `N.n_x.n_j` array        |
    