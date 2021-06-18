from __future__ import division

import numpy
import numpy as np


def mlinspace(smin, smax, orders):
    if len(orders) == 1:
        res = np.atleast_2d(
            np.linspace(np.array(smin), np.array(smax), np.array(orders))
        )
        return res.copy()  ## workaround for strange bug
    else:
        meshes = np.meshgrid(
            *[numpy.linspace(smin[i], smax[i], orders[i]) for i in range(len(orders))],
            indexing="ij"
        )
        return np.row_stack([l.flatten() for l in meshes])


class LinearSpline:
    """Multilinear interpolation

    Methods
    -------
    smin, smax, orders : iterable objects
        Specifies the boundaries of a cartesian grid, with the number of points along each dimension.
    values : array_like (2D), optional
        Each line enumerates values taken by a function on the cartesian grid, with last index varying faster.


    Attributes
    ----------
    smin, smax, orders :
        Boundaries and number of points along each dimension.
    d : number of dimensions
    grid : array_like (2D)
        Enumerate the approximation grid. Each column contains coordinates of a point in R^d.
    values :
        Values on the grid to be interpolated.

    Example
    -------

    smin = [-1,-1]
    smax = [1,1]
    orders = [5,5]

    f: lambda x: np.row_stack([
        np.sqrt( x[0,:]**2 + x[1,:]**2 )
        np.pow( x[0,:]**3 + x[1,:]**3, 1.0/3.0 )
    ])

    interp = MultilinearInterpolator(smin,smax,orders)
    interp.set_values( f(interp.grid) )

    random_points = np.random.random( (2, 1000) )

    interpolated_values = interp(random_points)
    exact_values = f(random_points)
    """

    __grid__ = None

    def __init__(self, smin, smax, orders, values=None, dtype=numpy.float64):
        self.smin = numpy.array(smin, dtype=dtype)
        self.smax = numpy.array(smax, dtype=dtype)
        self.orders = numpy.array(orders, dtype=numpy.int)
        self.d = len(orders)
        self.dtype = dtype
        if values is not None:
            self.set_values(values)

    @property
    def grid(self):
        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.smin, self.smax, self.orders)
        return self.__grid__

    def set_values(self, values):
        values = values.reshape(self.orders)
        self.values = numpy.ascontiguousarray(values, dtype=self.dtype)

    def interpolate(self, s):

        from .eval_splines import eval_linear

        s = numpy.ascontiguousarray(s, dtype=self.dtype)
        grid = tuple(
            (self.smin[i], self.smax[i], self.orders[i]) for i in range(len(self.smin))
        )
        a = eval_linear(grid, self.values, s)
        return a

    def __call__(self, s):

        if s.ndim == 1:
            res = self.__call__(numpy.atleast_2d(s))
            return res[0]
        return self.interpolate(s)


class LinearSplines:
    """Multilinear interpolation

    Methods
    -------
    smin, smax, orders : iterable objects
        Specifies the boundaries of a cartesian grid, with the number of points along each dimension.
    values : array_like (2D), optional
        Each line enumerates values taken by a function on the cartesian grid, with last index varying faster.


    Attributes
    ----------
    smin, smax, orders :
        Boundaries and number of points along each dimension.
    d : number of dimensions
    grid : array_like (2D)
        Enumerate the approximation grid. Each column contains coordinates of a point in R^d.
    values :
        Values on the grid to be interpolated.

    Example
    -------

    smin = [-1,-1]
    smax = [1,1]
    orders = [5,5]

    f: lambda x: np.row_stack([
        np.sqrt( x[0,:]**2 + x[1,:]**2 )
        np.pow( x[0,:]**3 + x[1,:]**3, 1.0/3.0 )
    ])

    interp = MultilinearInterpolator(smin,smax,orders)
    interp.set_values( f(interp.grid) )

    random_points = np.random.random( (2, 1000) )

    interpolated_values = interp(random_points)
    exact_values = f(random_points)
    """

    __grid__ = None

    def __init__(self, smin, smax, orders, mvalues=None, dtype=numpy.float64):
        self.smin = numpy.array(smin, dtype=dtype)
        self.smax = numpy.array(smax, dtype=dtype)
        self.orders = numpy.array(orders, dtype=numpy.int)
        self.d = len(orders)
        self.dtype = dtype
        if mvalues is not None:
            self.set_values(mvalues)

    @property
    def grid(self):
        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.smin, self.smax, self.orders)
        return self.__grid__

    def set_values(self, mvalues):

        n_x = mvalues.shape[-1]
        new_orders = list(self.orders) + [n_x]
        mvalues = mvalues.reshape(new_orders)
        self.mvalues = numpy.ascontiguousarray(mvalues, dtype=self.dtype)

    def interpolate(self, s):

        from .multilinear_numba import eval_linear

        grid = tuple(
            (self.smin[i], self.smax[i], self.orders[i]) for i in range(len(self.smin))
        )
        s = numpy.ascontiguousarray(s, dtype=self.dtype)
        a = eval_linear(grid, self.mvalues, s)
        return a

    def __call__(self, s):

        if s.ndim == 1:
            res = self.__call__(numpy.atleast_2d(s))
            return res.ravel()

        return self.interpolate(s)
