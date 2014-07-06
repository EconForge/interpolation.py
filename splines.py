"""High-level API for cubic splines"""

import numpy
import numpy as np

from .misc import mlinspace


class MultivariateCubicSplines:

    __grid__ = None
    __values__ = None
    __coeffs__ = None
    __n_splines__ = None

    def __init__(self, a, b, orders, dtype=np.float64):

        self.d = len(a)
        assert(len(b) == self.d)
        assert(len(orders) == self.d)
        self.a = np.array(a, dtype=dtype)
        self.b = np.array(b, dtype=dtype)
        self.orders = np.array(orders, dtype=np.int)
        self.dtype =  dtype
        self.__coeffs__ = None

    def set_values(self, values):

        assert(values.ndim <= len(self.orders))
        self.set_mvalues(values.T)


    def set_mvalues(self, values):

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
            values = np.empty((N,n_sp), dtype=self.dtype)
            vec_eval_cubic_multi_spline(self.a, self.b, self.orders, self.__coeffs__, points, values)

            return values
        else:
            raise Exception("Not implemented.")


    @property
    def grid(self):
        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.a, self.b, self.orders)
        return self.__grid__

    def __call__(self, s):

        if s.ndim == 1:
            res = self.__call__( numpy.atleast_2d(s).T )
            return res.ravel()

        return self.interpolate(s)


# Aliases

MultivariateSplines = MultivariateCubicSplines
