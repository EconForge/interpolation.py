import unittest
from interpolation.splines import CGrid, eval_spline
import numpy as np


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
            diff=str(((0,), (1,))),
            extrap_mode="linear",
        )

        # 0-order must be the function
        # 1-order must be the slope
        result = np.vstack(
            [
                y0 + slope * eval_points,
                np.ones_like(eval_points) * slope,
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
            diff=str(((0,), (1,))),
            extrap_mode="linear",
        )

        # 0-order must be the function
        # 1-order must be + or - pi/2
        result = np.vstack(
            [
                np.array([0, -1, 0, 1, 0]),
                np.array([-1, -1, 1, 1, -1]) * 2 / np.pi,
            ]
        ).T

        self.assertTrue(np.allclose(grad, result))

    def test_nonlinear_approx(self):

        # A non linear function on uniform grid
        x = np.linspace(-10, 10, 10000)
        y = np.power(x, 3)

        eval_points = np.linspace(-5, 5, 10)

        grad = eval_spline(
            CGrid(x),
            y,
            eval_points[..., None],
            out=None,
            order=1,
            diff=str(((0,), (1,))),
            extrap_mode="linear",
        )

        # 0-order must be x^3
        # 1-order must be close to 3x^2
        result = np.vstack(
            [
                np.power(eval_points, 3),
                np.power(eval_points, 2) * 3.0,
            ]
        ).T

        self.assertTrue(np.allclose(grad, result, atol=0.02))


class Check2DDerivatives(unittest.TestCase):
    """
    Checks derivatives in a 2D interpolator
    """

    def setUp(self):

        pass

    def test_linear(self):

        # A linear function on a non-uniform grid
        x = np.exp(np.linspace(0, 2, 6))
        y = np.power(np.linspace(0, 5, 10), 2)

        inter = 1
        slope_x = 2
        slope_y = -3
        z = inter + slope_x * x[..., None] + slope_y * y[None, ...]

        # Evaluation points
        n_eval = 15
        eval_points = np.vstack(
            [np.linspace(-20, 20, n_eval), np.linspace(5, -5, n_eval)]
        ).T

        # Get function and 1st derivatives
        grad = eval_spline(
            CGrid(x, y),
            z,
            eval_points,
            out=None,
            order=1,
            diff=str(((0, 0), (1, 0), (0, 1), (1, 1))),
            extrap_mode="linear",
        )

        # 0-order must be the function
        # (1,0) must be x slope
        # (0,1) must be y slope
        # (1,1) must be 0
        result = np.hstack(
            [
                inter + slope_x * eval_points[:, 0:1] + slope_y * eval_points[:, 1:2],
                np.ones((n_eval, 1)) * slope_x,
                np.ones((n_eval, 1)) * slope_y,
                np.zeros((n_eval, 1)),
            ]
        )

        self.assertTrue(np.allclose(grad, result))

    def test_nonlinear_approx(self):

        # A non linear function on uniform grid
        n_grid = 100
        x = np.linspace(1, 5, n_grid)
        y = np.linspace(1, 5, n_grid)
        z = np.sin(x[..., None]) * np.log(y[None, ...])

        # Evaluation points
        n_eval = 15
        eval_points = np.vstack(
            [np.linspace(2, 4, n_eval), np.linspace(4, 2, n_eval)]
        ).T

        # Get function and 1st derivatives
        grad = eval_spline(
            CGrid(x, y),
            z,
            eval_points,
            out=None,
            order=1,
            diff=str(((0, 0), (1, 0), (0, 1), (1, 1))),
            extrap_mode="linear",
        )

        # 0-order must be sin(x)*ln(y)
        # (1,0) must be cos(x) * ln(y)
        # (0,1) must be sin(x) / y
        # (1,1) must be cos(x) / y
        result = np.hstack(
            [
                np.sin(eval_points[:, 0:1]) * np.log(eval_points[:, 1:2]),
                np.cos(eval_points[:, 0:1]) * np.log(eval_points[:, 1:2]),
                np.sin(eval_points[:, 0:1]) * (1 / eval_points[:, 1:2]),
                np.cos(eval_points[:, 0:1]) * (1 / eval_points[:, 1:2]),
            ]
        )

        self.assertTrue(np.allclose(grad, result, atol=0.02))
