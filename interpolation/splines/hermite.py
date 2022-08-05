import numpy as np
from numba import jit
import numpy.typing as npt
from typing import Tuple


class Vector:
    pass


@jit(nopython=True)
def hermite_splines(lambda0: float) -> Tuple[float, float, float, float]:
    """Computes the cubic Hermite splines in lambda0
    Inputs: - float: lambda0
    Output: - tuple: cubic Hermite splines evaluated in lambda0"""
    h00 = 2 * (lambda0**3) - 3 * (lambda0**2) + 1
    h10 = (lambda0**3) - 2 * (lambda0**2) + lambda0
    h01 = -2 * (lambda0**3) + 3 * (lambda0**2)
    h11 = (lambda0**3) - (lambda0**2)
    return (h00, h10, h01, h11)


@jit(nopython=True)
def hermite_interp(
    x0: float, xk: float, xkn: float, pk: float, pkn: float, mk: float, mkn: float
) -> float:
    """Returns the interpolated value for x0.
    Inputs: - float: x0, abscissa of the point to interpolate
            - float: xk, abscissa of the nearest lowest point to x0 on the grid
            - float: xkn, abscissa of the nearest largest point to x0 on the grid
            - float: pk, value associated to xk
            - float: pkn, value associated to xkn
            - float: mk, tangent in xk
            - float: mkn, tangent in xkn
    Output: - float: interpolated value for x0
    """
    t = (x0 - xk) / (xkn - xk)
    hsplines = hermite_splines(t)
    return (
        pk * hsplines[0]
        + mk * (xkn - xk) * hsplines[1]
        + pkn * hsplines[2]
        + mkn * (xkn - xk) * hsplines[3]
    )


@jit(nopython=True)
def HermiteInterpolation(x0: float, x, y, yp):
    """Returns the interpolated value for x0
    Inputs: - float: x0, abscissa of the point to interpolate
            - np.ndarray: x, x-axis grid
            - np.ndarray: y, values of elements in x
            - np.ndarray: yp, tangents of the x elements
    Output: - float: interpolated value"""
    ###### Extrapolation case ######
    if x0 <= np.min(x):
        return y[0]
    elif x0 >= np.max(x):
        return y[-1]

    ###### Interpolation case ######
    indx = np.searchsorted(x, x0)
    xk, xkn = x[indx - 1], x[indx]
    pk, pkn = y[indx - 1], y[indx]
    mk, mkn = yp[indx - 1], yp[indx]
    return hermite_interp(x0, xk, xkn, pk, pkn, mk, mkn)


@jit(nopython=True)
def HermiteInterpolationVect(xvect, x: Vector, y: Vector, yp: Vector):
    """Returns the interpolated value for all elements in xvect
    Inputs: - np.ndarray: xvect, vector of abscissa of the point to interpolate
            - np.ndarray: x, x-axis grid
            - np.ndarray: y, values of elements in x
            - np.ndarray: tang, tangents of the x elements
    Output: - np.ndarray: interpolated values"""
    N = len(xvect)
    out = np.zeros(N)
    for i in range(N):
        x0 = xvect[i]
        out[i] = HermiteInterpolation(x0, x, y, yp)
    return out


from numba import njit, types
from numba.extending import overload, register_jitable
from numba import generated_jit


def _hermite(x0, x, y, yp, out=None):
    pass


@overload(_hermite)
def _hermite(x0, x, y, yp, out=None):
    def _hermite(x0, x, y, yp, out=None):
        return HermiteInterpolation(x0, x, y, yp)

    return _hermite


from numba.core.types.misc import NoneType as none


@generated_jit
def hermite(x0, x, y, yp, out=None):
    try:
        n = x0.ndim
        if n == 1:
            input_type = "vector"
        elif n == 2:
            input_type = "matrix"
        else:
            raise Exception("Invalid input type")
    except:
        # n must be a scalar
        input_type = "scalar"

    if input_type == "scalar":

        def _hermite(x0, x, y, yp, out=None):
            return HermiteInterpolation(x0, x, y, yp)

    elif input_type == "vector":

        def _hermite(x0, x, y, yp, out=None):
            return HermiteInterpolationVect(x0, x, y, yp)

    elif input_type == "matrix":

        def _hermite(x0, x, y, yp, out=None):
            return HermiteInterpolationVect(x0[:, 0], x, y, yp)

    return _hermite
