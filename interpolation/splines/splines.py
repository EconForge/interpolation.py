"""High-level API for cubic splines"""

import numpy
import numpy as np

from misc import mlinspace


class CubicSpline:

    """Class representing a cubic spline interpolator on a regular cartesian grid.."""

    __grid__ = None
    __values__ = None
    __coeffs__ = None

    def __init__(self, a, b, orders, values=None):
        """Creates a cubic spline interpolator on a regular cartesian grid.

        Parameters:
        -----------
        a : array of size d (float)
            Lower bounds of the cartesian grid.
        b : array of size d (float)
            Upper bounds of the cartesian grid.
        orders : array of size d (int)
            Number of nodes along each dimension (=(n1,...,nd) )
        values : (optional) array isomorphic to size (n1 x ... x nd)
            Values on the nodes of the function to interpolate.

        Returns
        -------
        spline : CubicSpline
            Cubic spline interpolator. Can be evaluated at point(s) `y` with
            `spline(y)`
        """


        self.d = len(a)
        assert(len(b) == self.d)
        assert(len(orders) == self.d)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.orders = np.array(orders, dtype=int)
        self.dtype =  self.a.dtype
        self.__coeffs__ = None

        if values is not None:
            self.set_values(values)
    #
    # def set_values(self, values):
    #
    #     assert(values.ndim <= len(self.orders))
    #     self.set_mvalues(values.T)
    #

    def set_values(self, values):
        '''Set values on the nodes for the function to interpolate.'''

        values = np.array(values, dtype=float)

        from filter_cubic_splines import filter_coeffs

        if not np.all( np.isfinite(values)):
            raise Exception('Trying to interpolate non-finite values')

        sh = self.orders.tolist()
        sh2 = [ e+2 for e in self.orders]

        values = values.reshape(sh)

        self.__values__ = values

        # this should be done without temporary memory allocation
        self.__coeffs__ = filter_coeffs(self.a, self.b, self.orders, values)



    def interpolate(self, points, values=None, with_derivatives=False):
        '''Interpolate spline at a list of points.

        Parameters
        ----------
        points : (array-like) list of point where the spline is evaluated.
        values : (optional) container inplace compuation

        Returns
        -------
        values : (array-like) list of point where the spline is evaluated.
        '''

        import time

        from eval_cubic_splines import vec_eval_cubic_spline, eval_cubic_spline

        if not np.all( np.isfinite(points)):
            raise Exception('Spline interpolator evaluated at non-finite points.')

        if not with_derivatives:
            if points.ndim == 1:
                # evaluate only on one point
                return eval_cubic_spline(self.a, self.b, self.orders, self.__coeffs__, points)
            else:

                N, d = points.shape
                assert(d==self.d)
                if values is None:
                    values = np.empty(N, dtype=self.dtype)
                vec_eval_cubic_spline(self.a, self.b, self.orders, self.__coeffs__, points, values)
                return values
        else:
            raise Exception("Not implemented.")


    @property
    def grid(self):
        """Cartesian enumeration of all nodes."""

        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.a, self.b, self.orders)
        return self.__grid__

    def __call__(self, s):
        """Interpolate the spline at one or many points"""

        if s.ndim == 1:
            res = self.__call__( numpy.atleast_2d(s) )
            return res[0]

        return self.interpolate(s)


class MultiCubicSpline:

    __grid__ = None
    __values__ = None
    __coeffs__ = None
    __n_splines__ = None

    def __init__(self, a, b, orders, values=None):
        """Creates a cubic spline interpolator for many functions on a regular cartesian grid."""


        self.d = len(a)
        assert(len(b) == self.d)
        assert(len(orders) == self.d)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.orders = np.array(orders, dtype=int)
        self.__coeffs__ = None
        if values is not None:
            self.set_values(values)


    def set_values(self, values):
        """Change values on the nodes of the functions to approximate."""


        values = np.array(values, dtype=float)

        from filter_cubic_splines import filter_coeffs

        if not np.all( np.isfinite(values)):
            raise Exception('Trying to interpolate non-finite values')

        # number of splines
        n_sp = values.shape[0]

        if self.__n_splines__ is None:
            self.__n_splines = n_sp
        else:
            assert(n_sp == self.__n_splines__)

        sh = [n_sp] + self.orders.tolist()
        sh2 = [ e+2 for e in self.orders]

        values = values.reshape(sh)

        self.__values__ = values

        if self.__coeffs__ is None:
            self.__coeffs__ = numpy.zeros( [n_sp] + sh2)

        # this should be done without temporary memory allocation
        for i in range(n_sp):
            data = values[i,...]
            self.__coeffs__[i,...] = filter_coeffs(self.a, self.b, self.orders,data)



    def interpolate(self, points, with_derivatives=False):
        """Interpolate splines at manu points."""

        import time

        from eval_cubic_splines import vec_eval_cubic_multi_spline

        if points.ndim == 1:
            raise Exception('Expected 2d array. Received {}d array'.format(points.ndim))
        if points.shape[1] != self.d:
            raise Exception('Second dimension should be {}. Received : {}.'.format(self.d, points.shape[0]))
        if not np.all( np.isfinite(points)):
            raise Exception('Spline interpolator evaluated at non-finite points.')

        n_sp = self.__coeffs__.shape[0]

        N = points.shape[0]
        d = points.shape[1]

        if not with_derivatives:
            values = np.empty((N,n_sp), dtype=float)
            vec_eval_cubic_multi_spline(self.a, self.b, self.orders, self.__coeffs__, points, values)

            return values
        else:
            raise Exception("Not implemented.")


    @property
    def grid(self):
        """Cartesian enumeration of all nodes."""
        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.a, self.b, self.orders)
        return self.__grid__

    def __call__(self, s):
        """Interpolate the splines at one or many points.

        Parameters
        ----------
        s : (array-like with 1 or 2 dimensions)
            Coordinates of one point, or list of coordinates, at which the splines
            are interpolated.

        Returns:
        --------
        res : (array-like with 1 or 2 dimensions)
            Vector or list of vectors containing the interpolator evaluated at `s`.
        """

        if s.ndim == 1:
            res = self.__call__( numpy.atleast_2d(s) )
            return res.ravel()

        return self.interpolate(s)


# Aliases

MultivariateSplines = MultiCubicSpline
