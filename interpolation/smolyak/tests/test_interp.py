# TESTS: Leaving these here for now so that I don't have to do anything
# fancy
from ..grid import SmolyakGrid
from ..interp import SmolyakInterp
import numpy as np

# func = lambda x, y: np.exp(x**2 - y**2)
func = lambda x, y: x**2 - y**2


func1 = lambda points: func(points[:, 0], points[:, 1])
func1_prime = lambda x: np.column_stack([2 * x[:, 0], -2 * x[:, 1]])

func2 = lambda x: np.sum(x**2, axis=1)
func2_prime = lambda x: 2 * x


func3 = lambda x: np.sin(x[:, 0]) + np.cos(x[:, 1])
func3_prime = lambda x: np.column_stack([np.cos(x[:, 0]), -np.sin(x[:, 1])])

msg = "mean abs diff is {}\nmax abs diff is {}\nmin abs diff is {}"


def interp_2d(d, mu, f):
    lb = -2 * np.ones(d)
    ub = 2 * np.ones(d)
    sg = SmolyakGrid(d, mu, lb, ub)

    f_on_grid = f(sg.grid)

    si = SmolyakInterp(sg, f_on_grid)

    np.random.seed(42)
    test_points = np.random.randn(100, d)
    # Make sure it is bounded by -2, 2
    test_points = 2 * test_points / np.max(np.abs(test_points))

    true_vals = f(test_points)
    interp_vals = si.interpolate(test_points)

    mean_ad = np.mean(np.abs(interp_vals - true_vals))
    max_ad = np.max(np.abs(interp_vals - true_vals))
    min_ad = np.min(np.abs(interp_vals - true_vals))

    print(msg.format(mean_ad, max_ad, min_ad))
    return


def interp_2d1(d, mu, f):
    sg = SmolyakGrid(d, mu, np.array([-1, -1.0]), np.array([1.0, 1.0]))

    f_on_grid = f(sg.grid)

    si = SmolyakInterp(sg, f_on_grid)

    np.random.seed(42)
    test_points = np.random.randn(100, 2)
    # Make sure it is bounded by -2, 2
    test_points = test_points / np.max(np.abs(test_points))

    true_vals = f(test_points)
    interp_vals = si.interpolate(test_points)

    mean_ad = np.mean(np.abs(interp_vals - true_vals))
    max_ad = np.max(np.abs(interp_vals - true_vals))
    min_ad = np.min(np.abs(interp_vals - true_vals))

    print(msg.format(mean_ad, max_ad, min_ad))
    return


def interp2d_derivs(d, mu, f, f_prime, bds=1, verbose=True):
    sg = SmolyakGrid(d, mu, -bds, bds)

    f_on_grid = f(sg.grid)
    si = SmolyakInterp(sg, f_on_grid)

    np.random.seed(42)
    test_points = np.random.randn(100, d)
    # Make sure it is bounded by -2, 2
    test_points = (bds - 0.05) * test_points / np.max(np.abs(test_points))

    true_vals = f(test_points)
    true_vals_prime = f_prime(test_points)
    i_vals = si.interpolate(test_points)
    i_vals, i_vals_prime = si.interpolate(test_points, deriv=True)

    mean_ad = np.mean(np.abs(i_vals - true_vals))
    max_ad = np.max(np.abs(i_vals - true_vals))
    min_ad = np.min(np.abs(i_vals - true_vals))

    mean_ad_prime = np.mean(np.abs(i_vals_prime - true_vals_prime))
    max_ad_prime = np.max(np.abs(i_vals_prime - true_vals_prime))
    min_ad_prime = np.min(np.abs(i_vals_prime - true_vals_prime))

    if verbose:
        msg = "mean abs diff is {}\nmax abs diff is {}\nmin abs diff is {}"
        print("Interpolation results\n" + "#" * 21)
        print(msg.format(mean_ad, max_ad, min_ad))

        print("Derivative results\n" + "#" * 18)
        print(msg.format(mean_ad_prime, max_ad_prime, min_ad_prime))

    # return i_vals_prime


# functions starting with test_ are
# discovered and automatically run by nosetests


def test_interp_2d():

    interp_2d(2, 3, func1)


def test_interp_2d1():

    interp_2d1(2, 3, func1)
