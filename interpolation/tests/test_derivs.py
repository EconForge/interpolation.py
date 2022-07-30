import unittest
from interpolation.splines import CGrid, eval_spline
import numpy as np
import numba


class Check1DDerivatives(unittest.TestCase):
    """ 
    Checks derivatives in a 1D interpolator
    """

    def setUp(self):

        pass

    def test_linear(self):

        # A linear function on a non-uniform grid
        x = np.exp(np.linspace(0, 2, 6))
        y0 = 2
        slope = 1.0
        y = y0 + slope * x

        eval_points = np.array([1.5, 2.5, 3.5, 4.5])

        grad = eval_spline(
            CGrid(x),
            y,
            eval_points[..., None],
            out=None,
            order=1,
            diff=str(((0,), (1,), (2,))),
            extrap_mode="linear",
        )

        print(grad)

        # 0-order must be the function
        # 1-order must be the slope
        # 2-order must be 0
        result = np.vstack(
            [
                y0 + slope * eval_points,
                np.ones_like(eval_points) * slope,
                np.zeros_like(eval_points),
            ]
        ).T

        self.assertTrue(np.allclose(grad, result))

    def test_nonlinear(self):

        # A non linear function on uniform grid
        x = np.linspace(-10, 10, 21) * (1 / 2) * np.pi
        y = np.sin(x)

        eval_points = np.array([-1, -0.5, 0, 0.5, 1]) * np.pi

        grad = eval_spline(
            CGrid(x),
            y,
            eval_points[..., None],
            out=None,
            order=1,
            diff=str(((0,), (1,), (2,))),
            extrap_mode="linear",
        )

        # 0-order must be the function
        # 1-order must be the + or - pi/2
        # 2-order must be 0
        result = np.vstack(
            [
                np.array([0, -1, 0, 1, 0]),
                np.array([-1, -1, 1, 1, -1]) * 2 / np.pi,
                np.array([0, 0, 0, 0, 0]),
            ]
        ).T

        self.assertTrue(np.allclose(grad, result))
