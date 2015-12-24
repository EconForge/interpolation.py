"""
This file contains the interpolation routines for the grids that are
built using the grid.py file in the smolyak package...  Write more doc
soon.
"""

from __future__ import division
import numpy as np
import numpy.linalg as la
from .grid import build_B

__all__ = ['find_theta', 'SmolyakInterp']


def find_theta(sg, f_on_grid):
    """
    Given a SmolyakGrid object and the value of the function on the
    points of the grid, this function will return the coefficients theta
    """
    return la.solve(sg.B_U, la.solve(sg.B_L, f_on_grid))


class SmolyakInterp(object):
    """
    This class is going to take several inputs.  It will need a
    SmolyakGrid object to be passed in and the values of the function
    evaluated at the grid points
    """
    def __init__(self, sg, f_on_grid):
        self.sg = sg
        self.f_on_grid = f_on_grid
        self.theta = find_theta(sg, self.f_on_grid)

    def update_theta(self, f_on_grid):
        self.f_on_grid = f_on_grid
        self.theta = find_theta(self.sg, self.f_on_grid)

    def interpolate(self, pts, interp=True, deriv=False, deriv_th=False,
                    deriv_X=False):
        """
        Basic Lagrange interpolation, with optional first derivatives
        (gradient)

        Parameters
        ----------
        pts : array (float, ndim=2)
            A 2d array of points on which to evaluate the function. Each
            row is assumed to be a new d-dimensional point. Therefore, pts
            must have the same number of columns as ``si.SGrid.d``

        interp : bool, optional(default=false)
            Whether or not to compute the actual interpolation values at pts

        deriv : bool, optional(default=false)
            Whether or not to compute the gradient of the function at each
            of the points. This will have the same dimensions as pts, where
            each column represents the partial derivative with respect to
            a new dimension.

        deriv_th : bool, optional(default=false)
            Whether or not to compute the ???? derivative with respect to the
            Smolyak polynomial coefficients (maybe?)

        deriv_X : bool, optional(default=false)
            Whether or not to compute the ???? derivative with respect to grid
            points


        Returns
        -------
        rets : list (array(float))
            A list of arrays containing the objects asked for. There are 4
            possible objects that can be computed in this function. They will,
            if they are called for, always be in the following order:

            1. Interpolation values at pts
            2. Gradient at pts
            3. ???? at pts
            4. ???? at pts

            If the user only asks for one of these objects, it is returned
            directly as an array and not in a list.


        Notes
        -----
        This is a stripped down port of ``dolo.SmolyakBasic.interpolate``

        TODO: There may be a better way to do this

        TODO: finish the docstring for the 2nd and 3rd type of derivatives

        """
        d = pts.shape[1]
        sg = self.sg

        theta = self.theta
        trans_points = sg.dom2cube(pts)  # Move points to correct domain

        rets = []

        if deriv:
            new_B, der_B = build_B(d, sg.mu, trans_points, sg.pinds, True)
            vals = new_B.dot(theta)
            d_vals = np.tensordot(theta, der_B, (0, 0)).T

            if interp:
                rets.append(vals)
            rets.append(sg.dom2cube(d_vals))

        elif not deriv and interp:  # No derivs in build_B. Just do vals
            new_B = build_B(d, sg.mu, trans_points, sg.pinds)
            vals = new_B.dot(theta)
            rets.append(vals)

        if deriv_th:  # The derivative wrt the coeffs is just new_B
            if not interp and not deriv:  # we  haven't found this  yet
                new_B = build_B(d, sg.mu, trans_points, sg.pinds)
            rets.append(new_B)

        if deriv_X:
            if not interp and not deriv and not deriv_th:
                new_B = build_B(d, sg.mu, trans_points, sg.pinds)
            d_X = la.solve(sg.B_U, la.solve(sg.B_L, new_B.T))
            rets.append(d_X)

        if len(rets) == 1:
            rets = rets[0]

        return rets


# if __name__ == '__main__':
    # from grid import SmolyakGrid
    # d = 2
    # mu = 3
    # f = lambda x: np.sum(x ** 2, axis=1)
    # f_prime = lambda x: 2 * x
    # sg = SmolyakGrid(d, mu, np.array([-1, -1.]), np.array([1., 1.]))

    # f_on_grid = f(sg.grid)

    # si = SmolyakInterp(sg, f_on_grid)

    # np.random.seed(42)
    # test_points = np.random.randn(100, 2)
    # # Make sure it is bounded by -2, 2
    # test_points = test_points/np.max(np.abs(test_points))

    # true_vals = f(test_points)
    # interp_vals = si.interpolate(test_points)

    # mean_ad = np.mean(np.abs(interp_vals - true_vals))
    # max_ad = np.max(np.abs(interp_vals - true_vals))
    # min_ad = np.min(np.abs(interp_vals - true_vals))

    # msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
    # msg += "The min abs diff is {}"

    # print(msg.format(mean_ad, max_ad, min_ad))
