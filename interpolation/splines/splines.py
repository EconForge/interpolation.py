"""High-level API for cubic splines"""

import numpy
import numpy as np

from ..cartesian import mlinspace


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
        values : (optional, (n1 x ... x nd) array)
            Values on the nodes of the function to interpolate.

        Returns
        -------
        spline : CubicSpline
            Cubic spline interpolator. Can be evaluated at point(s) `y` with
            `spline(y)`
        """

        self.d = len(a)
        assert len(b) == self.d
        assert len(orders) == self.d
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.orders = np.array(orders, dtype=int)
        self.dtype = self.a.dtype
        self.__coeffs__ = None

        if values is not None:
            self.set_values(values)

    def set_values(self, values):
        """Set values on the nodes for the function to interpolate."""

        values = np.array(values, dtype=float)

        from .filter_cubic import filter_coeffs

        if not np.all(np.isfinite(values)):
            raise Exception("Trying to interpolate non-finite values")

        sh = self.orders.tolist()
        sh2 = [e + 2 for e in self.orders]

        values = values.reshape(sh)

        self.__values__ = values

        # this should be done without temporary memory allocation
        self.__coeffs__ = filter_coeffs(self.a, self.b, self.orders, values)

    def interpolate(self, points, values=None, with_derivatives=False):
        """Interpolate spline at a list of points.

        Parameters
        ----------
        points : (array-like) list of point where the spline is evaluated.
        values : (optional) container for inplace computation

        Returns
        -------
        values : (array-like) list of point where the spline is evaluated.
        """

        import time

        from .eval_splines import eval_cubic

        grid = tuple((self.a[i], self.b[i], self.orders[i]) for i in range(len(self.a)))

        if not np.all(np.isfinite(points)):
            raise Exception("Spline interpolator evaluated at non-finite points.")

        if not with_derivatives:
            if points.ndim == 1:
                # evaluate only on one point
                return eval_cubic(grid, self.__coeffs__, points)
            else:

                N, d = points.shape
                assert d == self.d
                if values is None:
                    values = np.empty(N, dtype=self.dtype)
                eval_cubic(grid, self.__coeffs__, points, values)
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
            res = self.__call__(numpy.atleast_2d(s))
            return res[0]

        return self.interpolate(s)


class CubicSplines:

    __grid__ = None
    __values__ = None
    __coeffs__ = None
    __n_splines__ = None

    def __init__(self, a, b, orders, values=None):
        """Creates a cubic multi-spline interpolator for many functions on a regular cartesian grid."""

        self.d = len(a)
        assert len(b) == self.d
        assert len(orders) == self.d
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.orders = np.array(orders, dtype=int)
        self.__mcoeffs__ = None
        if values is not None:
            self.set_values(values)

    def set_values(self, mvalues):
        """Change values on the nodes of the functions to approximate."""

        mvalues = np.array(mvalues, dtype=float)
        n_sp = mvalues.shape[-1]

        mvalues = mvalues.reshape(list(self.orders) + [n_sp])

        if not np.all(np.isfinite(mvalues)):
            raise Exception("Trying to interpolate non-finite values")

        from .filter_cubic import filter_mcoeffs

        # number of splines
        self.__mcoeffs__ = filter_mcoeffs(self.a, self.b, self.orders, mvalues)
        self.__mvalues__ = mvalues

    def interpolate(self, points, diff=False):
        """Interpolate splines at manu points."""

        import time

        if points.ndim == 1:
            raise Exception("Expected 2d array. Received {}d array".format(points.ndim))
        if points.shape[1] != self.d:
            raise Exception(
                "Second dimension should be {}. Received : {}.".format(
                    self.d, points.shape[0]
                )
            )
        if not np.all(np.isfinite(points)):
            raise Exception("Spline interpolator evaluated at non-finite points.")

        n_sp = self.__mcoeffs__.shape[-1]

        N = points.shape[0]
        d = points.shape[1]

        from .eval_splines import eval_cubic

        if not diff:
            grid = tuple(
                (self.a[i], self.b[i], self.orders[i]) for i in range(len(self.a))
            )
            from .eval_splines import eval_cubic

            values = np.empty((N, n_sp), dtype=float)
            eval_cubic(grid, self.__mcoeffs__, points, values)
            return values
        else:
            from .eval_cubic import vec_eval_cubic_splines_G

            values = np.empty((N, n_sp), dtype=float)
            dvalues = np.empty((N, d, n_sp), dtype=float)
            vec_eval_cubic_splines_G(
                self.a, self.b, self.orders, self.__mcoeffs__, points, values, dvalues
            )
            return [values, dvalues]

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
            res = self.__call__(numpy.atleast_2d(s))
            return res.ravel()

        return self.interpolate(s)
