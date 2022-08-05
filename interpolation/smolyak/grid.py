"""
This file contains a class that builds a Smolyak Grid.  The hope is that
it will eventually contain the interpolation routines necessary so that
the given some data, this class can build a grid and use the Chebychev
polynomials to interpolate and approximate the data.

Method based on Judd, Maliar, Maliar, Valero 2013 (W.P)

Authors
=======

- Chase Coleman (ccoleman@stern.nyu.edu)
- Spencer Lyon (slyon@stern.nyu.edu)

References
==========
Judd, Kenneth L, Lilia Maliar, Serguei Maliar, and Rafael Valero. 2013.
    "Smolyak Method for Solving Dynamic Economic Models: Lagrange
    Interpolation, Anisotropic Grid and Adaptive Domain".

Krueger, Dirk, and Felix Kubler. 2004. "Computing Equilibrium in OLG
    Models with Stochastic Production." Journal of Economic Dynamics and
    Control 28 (7) (April): 1411-1436.

"""
from __future__ import division
from operator import mul
from itertools import product, combinations_with_replacement
from itertools import chain
import numpy as np
from scipy.linalg import lu
from functools import reduce
from .util import *

## --------------- ##
# - Building Blocks -#
## --------------- ##

__all__ = [
    "num_grid_points",
    "m_i",
    "cheby2n",
    "s_n",
    "a_chain",
    "phi_chain",
    "smol_inds",
    "build_grid",
    "build_B",
    "SmolyakGrid",
]


def num_grid_points(d, mu):
    """
    Checks the number of grid points for a given d, mu combination.

    Parameters
    ----------
    d, mu : int
        The parameters d and mu that specify the grid

    Returns
    -------
    num : int
        The number of points that would be in a grid with params d, mu

    Notes
    -----
    This function is only defined for mu = 1, 2, or 3

    """
    if mu == 1:
        return 2 * d + 1

    if mu == 2:
        return 1 + 4 * d + 4 * d * (d - 1) / 2.0

    if mu == 3:
        return 1 + 8 * d + 12 * d * (d - 1) / 2.0 + 8 * d * (d - 1) * (d - 2) / 6.0


def m_i(i):
    r"""
    Compute one plus the "total degree of the interpolating
    polynoimals" (Kruger & Kubler, 2004). This shows up many times in
    Smolyak's algorithm. It is defined as:

    .. math::

        m_i = \begin{cases}
        1 \quad & \text{if } i = 1 \\
        2^{i-1} + 1 \quad & \text{if } i \geq 2
        \end{cases}

    Parameters
    ----------
    i : int
        The integer i which the total degree should be evaluated

    Returns
    -------
    num : int
        Return the value given by the expression above

    """
    if i < 0:
        raise ValueError("i must be positive")
    elif i == 0:
        return 0
    elif i == 1:
        return 1
    else:
        return 2 ** (i - 1) + 1


def chebyvalto(x, n, kind=1.0):
    """
    Computes first :math:`n` Chebychev polynomials of the first kind
    evaluated at each point in :math:`x` and places them side by side
    in a matrix.  NOTE: Not including the first Chebychev polynomial
    because it is simply a set of ones

    Parameters
    ----------
    x : float or array(float)
        A single point (float) or an array of points where each
        polynomial should be evaluated

    n : int
        The integer specifying which Chebychev polynomial is the last
        to be computed

    kind : float, optional(default=1.0)
        The "kind" of Chebychev polynomial to compute. Only accepts
        values 1 for first kind or 2 for second kind

    Returns
    -------
    results : array (float, ndim=x.ndim+1)
        The results of computation. This will be an :math:`(n+1 \\times
        dim \\dots)` where :math:`(dim \\dots)` is the shape of x. Each
        slice along the first dimension represents a new Chebychev
        polynomial. This dimension has length :math:`n+1` because it
        includes :math:`\\phi_0` which is equal to 1 :math:`\\forall x`
    """
    x = np.asarray(x)
    row, col = x.shape

    ret_matrix = np.zeros((row, col * (n - 1)))

    init = np.ones((row, col))
    ret_matrix[:, :col] = x * kind
    ret_matrix[:, col : 2 * col] = 2 * x * ret_matrix[:, :col] - init

    for i in range(3, n):
        ret_matrix[:, col * (i - 1) : col * (i)] = (
            2 * x * ret_matrix[:, col * (i - 2) : col * (i - 1)]
            - ret_matrix[:, col * (i - 3) : col * (i - 2)]
        )

    return ret_matrix


def cheby2n(x, n, kind=1.0):
    """
    Computes the first :math:`n+1` Chebychev polynomials of the first
    kind evaluated at each point in :math:`x` .

    Parameters
    ----------
    x : float or array(float)
        A single point (float) or an array of points where each
        polynomial should be evaluated

    n : int
        The integer specifying which Chebychev polynomial is the last
        to be computed

    kind : float, optional(default=1.0)
        The "kind" of Chebychev polynomial to compute. Only accepts
        values 1 for first kind or 2 for second kind

    Returns
    -------
    results : array (float, ndim=x.ndim+1)
        The results of computation. This will be an :math:`(n+1 \\times
        dim \\dots)` where :math:`(dim \\dots)` is the shape of x. Each
        slice along the first dimension represents a new Chebychev
        polynomial. This dimension has length :math:`n+1` because it
        includes :math:`\\phi_0` which is equal to 1 :math:`\\forall x`

    """
    x = np.asarray(x)
    dim = x.shape
    results = np.zeros((n + 1,) + dim)
    results[0, ...] = np.ones(dim)
    results[1, ...] = x * kind
    for i in range(2, n + 1):
        results[i, ...] = 2 * x * results[i - 1, ...] - results[i - 2, ...]
    return results


def s_n(n):
    """
    Finds the set :math:`S_n` , which is the :math:`n` th Smolyak set of
    Chebychev extrema

    Parameters
    ----------
    n : int
        The index :math:`n` specifying which Smolyak set to compute

    Returns
    -------
    s_n : array (float, ndim=1)
        An array containing all the Chebychev extrema in the set
        :math:`S_n`

    """

    if n == 1:
        return np.array([0.0])

    # Apply the necessary transformation to get the nested sequence
    m_i = 2 ** (n - 1) + 1

    # Create an array of values that will be passed in to calculate
    # the set of values
    comp_vals = np.arange(1.0, m_i + 1.0)

    # Values are - cos(pi(j-1)/(n-1)) for j in [1, 2, ..., n]
    vals = -1.0 * np.cos(np.pi * (comp_vals - 1.0) / (m_i - 1.0))
    vals[np.where(np.abs(vals) < 1e-14)] = 0.0

    return vals


def a_chain(n):
    """
    Finds all of the unidimensional disjoint sets of Chebychev extrema
    that are used to construct the grid.  It improves on past algorithms
    by noting  that :math:`A_{n} = S_{n}` [evens] except for :math:`A_1
    = \{0\}`  and :math:`A_2 = \{-1, 1\}` . Additionally, :math:`A_{n} =
    A_{n+1}` [odds] This prevents the calculation of these nodes
    repeatedly. Thus we only need to calculate biggest of the S_n's to
    build the sequence of :math:`A_n` 's

    Parameters
    ----------
    n : int
      This is the number of disjoint sets from Sn that this should make

    Returns
    -------
    A_chain : dict (int -> list)
      This is a dictionary of the disjoint sets that are made.  They are
      indexed by the integer corresponding

    """

    # # Start w finding the biggest Sn(We will subsequently reduce it)
    Sn = s_n(n)

    A_chain = {}
    A_chain[1] = [0.0]
    A_chain[2] = [-1.0, 1.0]

    # Need a for loop to extract remaining elements
    for seq in range(n, 2, -1):
        num = Sn.size
        # Need odd indices in python because indexing starts at 0
        A_chain[seq] = tuple(Sn[range(1, num, 2)])
        # A_chain.append(list(Sn[range(1, num, 2)]))
        Sn = Sn[range(0, num, 2)]

    return A_chain


def phi_chain(n):
    """
    For each number in 1 to n, compute the Smolyak indices for the
    corresponding basis functions. This is the :math:`n` in
    :math:`\\phi_n`

    Parameters
    ----------
    n : int
        The last Smolyak index :math:`n` for which the basis polynomial
        indices should be found

    Returns
    -------
    aphi_chain : dict (int -> list)
        A dictionary whose keys are the Smolyak index :math:`n` and
        values are lists containing all basis polynomial subscripts for
        that Smolyak index

    """

    # First create a dictionary
    aphi_chain = {}

    aphi_chain[1] = [1]
    aphi_chain[2] = [2, 3]

    curr_val = 4
    for i in range(3, n + 1):
        end_val = 2 ** (i - 1) + 1
        temp = range(curr_val, end_val + 1)
        aphi_chain[i] = temp
        curr_val = end_val + 1

    return aphi_chain


## ---------------------- ##
# - Construction Utilities -#
## ---------------------- ##


def smol_inds(d, mu):
    """
    Finds all of the indices that satisfy the requirement that
    :math:`d \leq \sum_{i=1}^d \leq d + \mu`.

    Parameters
    ----------
    d : int
        The number of dimensions in the grid

    mu : int or array (int, ndim=1)
        The parameter mu defining the density of the grid. If an array,
        there must be d elements and an anisotropic grid is formed

    Returns
    -------
    true_inds : array
        A 1-d Any array containing all d element arrays satisfying the
        constraint

    Notes
    -----
    This function is used directly by build_grid and poly_inds

    """
    if isinstance(mu, int):
        max_mu = mu
    else:
        if mu.size != d:
            raise ValueError("mu must have d elements. It has %i" % mu.size)
        max_mu = int(np.max(mu))

    # Need to capture up to value mu + 1 so in python need mu+2
    possible_values = range(1, max_mu + 2)

    # find all (i1, i2, ... id) such that their sum is in range
    # we want; this will cut down on later iterations
    poss_inds = [
        el
        for el in combinations_with_replacement(possible_values, d)
        if d < sum(el) <= d + max_mu
    ]

    if isinstance(mu, int):
        true_inds = [[el for el in permute(list(val))] for val in poss_inds]
    else:
        true_inds = [
            [el for el in permute(list(val)) if all(el <= mu + 1)] for val in poss_inds
        ]

    # Add the d dimension 1 array so that we don't repeat it a bunch
    # of times
    true_inds.extend([[[1] * d]])

    tinds = list(chain.from_iterable(true_inds))

    return tinds


def poly_inds(d, mu, inds=None):
    """
    Build indices specifying all the Cartesian products of Chebychev
    polynomials needed to build Smolyak polynomial

    Parameters
    ----------
    d : int
        The number of dimensions in grid / polynomial

    mu : int
        The parameter mu defining the density of the grid

    inds : list (list (int)), optional (default=None)
        The Smolyak indices for parameters d and mu. Should be computed
        by calling `smol_inds(d, mu)`. If None is given, the indices
        are computed using this function call

    Returns
    -------
    phi_inds : array : (int, ndim=2)
        A two dimensional array of integers where each row specifies a
        new set of indices needed to define a Smolyak basis polynomial

    Notes
    -----
    This function uses smol_inds and phi_chain. The output of this
    function is used by build_B to construct the B matrix

    """
    if inds is None:
        inds = smol_inds(d, mu)

    if isinstance(mu, int):
        max_mu = mu
    else:
        max_mu = max(mu)

    aphi = phi_chain(max_mu + 1)

    base_polys = []

    for el in inds:
        temp = [aphi[i] for i in el]
        # Save these indices that we iterate through because
        # we need them for the chebychev polynomial combination
        # inds.append(el)
        base_polys.extend(list(product(*temp)))

    return base_polys


def build_grid(d, mu, inds=None):
    """
    Use disjoint Smolyak sets to construct Smolyak grid of degree d and
    density parameter :math:`mu`

    The return value is an :math:`n \\times d` Array, where :math:`n`
    is the number of points in the grid

    Parameters
    ----------
    d : int
        The number of dimensions in the grid

    mu : int
        The density parameter for the grid

    inds : list (list (int)), optional (default=None)
        The Smolyak indices for parameters d and mu. Should be computed
        by calling `smol_inds(d, mu)`. If None is given, the indices
        are computed using this function call

    Returns
    -------
    grid : array (float, ndim=2)
        The Smolyak grid for the given d, :math:`mu`

    """
    if inds is None:
        inds = smol_inds(d, mu)

    # Get An chain
    if isinstance(mu, int):
        An = a_chain(mu + 1)
    else:  # Anisotropic case
        An = a_chain(max(mu) + 1)

    points = []

    # Need to get the correct indices

    for el in inds:
        temp = [An[i] for i in el]
        # Save these indices that we iterate through because
        # we need them for the chebychev polynomial combination
        # inds.append(el)
        points.extend(list(product(*temp)))

    grid = np.array(points)

    return grid


def build_B(d, mu, pts, b_inds=None, deriv=False):
    """
    Compute the matrix B from equation 22 in JMMV 2013
    Translation of dolo.numeric.interpolation.smolyak.SmolyakBasic

    Parameters
    ----------
    d : int
        The number of dimensions on the grid

    mu : int or array (int, ndim=1, legnth=d)
        The mu parameter used to define grid

    pts : array (float, dims=2)
        Arbitrary d-dimensional points. Each row is assumed to be a new
        point. Often this is the smolyak grid returned by calling
        `build_grid(d, mu)`

    b_inds : array (int, ndim=2)
        The polynomial indices for parameters a given grid. These should
        be  computed by calling `poly_inds(d, mu)`.

    deriv : bool
        Whether or not to compute the values needed for the derivative matrix
        B_prime.

    Returns
    -------
    B : array (float, ndim=2)
        The matrix B that represents the Smolyak polynomial
        corresponding to grid

    B_Prime : array (float, ndim=3), optional (default=false)
        This will be the 3 dimensional array representing the gradient of the
        Smolyak polynomial at each of the points. It is only returned when
        `deriv=True`

    """
    if b_inds is None:
        inds = smol_inds(d, mu)
        b_inds = poly_inds(d, mu, inds)

    if isinstance(mu, int):
        max_mu = mu
    else:
        max_mu = max(mu)

    Ts = cheby2n(pts.T, m_i(max_mu + 1))
    npolys = len(b_inds)
    npts = pts.shape[0]
    B = np.empty((npts, npolys), order="F")
    for ind, comb in enumerate(b_inds):
        B[:, ind] = reduce(mul, [Ts[comb[i] - 1, i, :] for i in range(d)])

    if deriv:
        # TODO: test this. I am going to bed.
        Us = cheby2n(pts.T, m_i(max_mu + 1), kind=2.0)
        Us = np.concatenate([np.zeros((1, d, npts)), Us], axis=0)
        for i in range(Us.shape[0]):
            Us[i, :, :] = Us[i, :, :] * i

        der_B = np.zeros((npolys, d, npts))

        for i in range(d):
            for ind, comb in enumerate(b_inds):
                der_B[ind, i, :] = reduce(
                    mul,
                    [
                        (Ts[comb[k] - 1, k, :] if i != k else Us[comb[k] - 1, k, :])
                        for k in range(d)
                    ],
                )

        return B, der_B

    return B


# def exp_B(d, mu, grid):
#     """
#     write a nice doc string if it works
#     """
#     npts = grid.shape[0]
#     num_chebs = m_i(mu + 1)
#     max_ind = d + mu
#     aphi = phi_chain(mu + 1)

#     B = np.ones((npts, npts))

#     # These are simply all the values of phi_n (up to n=mu+1) where all
#     # other indices on the phi are 1 (hence valued at 1)
#     easy_B = chebyvalto(grid, num_chebs)

#     B[:, :d*(num_chebs-1)] = easy_B

#     # Create a tracker to keep track of indexes
#     B_col_mrk = d*(num_chebs - 1)

#     # Now we need to account for all the cross products
#     # We have the values we need hiding in B already.  No need to
#     # compute any more.  They are multiplications of different numbers
#     # of elements from the pieces of easy_B.
#     if mu==2:
#         for i in range(d-1):

#             mult_inds = np.hstack([np.arange(i+1, d), np.arange(d + (i+1), 2*d)])

#             temp1 = easy_B[:, i].reshape(npts, 1) * easy_B[:, mult_inds]
#             temp2 = temp2 = easy_B[:, i+d].reshape(npts, 1) * easy_B[:, mult_inds]

#             new_cols = temp1.shape[1] + temp2.shape[1]
#             B[:, B_col_mrk: B_col_mrk + new_cols] = np.hstack([temp1, temp2])
#             B_col_mrk = B_col_mrk + new_cols

#     #-----------------------------------------------------------------#
#     #-----------------------------------------------------------------#
#     # This part will be the general section.  Above I am trying to
#     # make it work with just mu=2
#     # NOTE: Below this point the code is incomplete.  At best this is
#     # some general pseudo-code to write the generalization step.  Hoping
#     # to make it handle general cases.
#     #-----------------------------------------------------------------#
#     #-----------------------------------------------------------------#

#     # for i in range(2, mu+1):
#     #     curr_ind = i

#     #     while True:
#     #         curr_dim = 2
#     #         curr_col = 0

#     #         # Find which possible polynomials can be reached (lowest is 2)
#     #         poss_inds = np.arange(2, m_i(some function of curr_ind, d, mu)+1)

#     #         for dd in range(d-1):
#     #             # Create range of d to be used to build the fancy index
#     #             mult_ind = np.arange(curr_col+1, d)

#     #             # Initialize array for fancy index.  Want to add arange(d) + (d*i)
#     #             # for every chebyshev polynomial that we need to reach with these
#     #             # indexes
#     #             mult_inds = np.array([])
#     #             for tt in range(some condition for what is max polynomial we reach -1):
#     #                 mult_inds = np.hstack([mult_inds, mult_inds + (d*tt)])

#     #             # this will create the column times all the stuff following it
#     #             # in the other indexes
#     #             temp1 = easy_B[:, curr_col] * easy_B[:, mult_inds]

#     #             new_cols = temp1.shape[1]

#     #             B[:, B_col_mrk: B_col_mrk + new_cols] = temp1

#     #             while d>curr_dim and condition for continuing is met:
#     #                 curr_dim += 1
#     #                 for mm in range(curr_col + 2, d-1):
#     #                     for bb in mult_inds[:-1]:
#     #                         temp2 = easy_B[:, bb*d + mm] * temp1
#     #                         new_cols2 = temp2.shape[1]
#     #                         B[:, B_col_mrk: B_col_mrk + new_cols2]

#                 # Need to continue code.  It is not done yet

#     return B

## ------------------ ##
# - Class: SmolyakGrid -#
## ------------------ ##


class SmolyakGrid(object):
    """
    This class currently takes a dimension and a degree of polynomial
    and builds the Smolyak Sparse grid.  We base this on the work by
    Judd, Maliar, Maliar, and Valero (2013).

    Parameters
    ----------
    d : int
        The number of dimensions in the grid

    mu : int or array(int, ndim=1, length=d)
        The "density" parameter for the grid


    Attributes
    ----------
    d : int
        This is the dimension of grid that you are building

    mu : int
        mu is a parameter that defines the fineness of grid that we
        want to build

    lb : array (float, ndim=2)
        This is an array of the lower bounds for each dimension

    ub : array (float, ndim=2)
        This is an array of the upper bounds for each dimension

    cube_grid : array (float, ndim=2)
        The Smolyak sparse grid on the domain :math:`[-1, 1]^d`

    grid: : array (float, ndim=2)
        The sparse grid, transformed to the user-specified bounds for
        the domain

    inds : list (list (int))
        This is a lists of lists that contains all of the indices

    B : array (float, ndim=2)
        This is the B matrix that is used to do lagrange interpolation

    B_L : array (float, ndim=2)
        Lower triangle matrix of the decomposition of B

    B_U : array (float, ndim=2)
        Upper triangle matrix of the decomposition of B

    Examples
    --------
    >>> s = SmolyakGrid(3, 2)
    >>> s
    Smolyak Grid:
        d: 3
        mu: 2
        npoints: 25
        B: 0.65% non-zero
    >>> ag = SmolyakGrid(3, [1, 2, 3])
    >>> ag
    Anisotropic Smolyak Grid:
        d: 3
        mu: 1 x 2 x 3
        npoints: 51
        B: 0.68% non-zero

    """

    def __init__(self, d, mu, lb=None, ub=None):
        self.d = d

        if lb is None:  # default is [-1, 1]^d
            self.lb = -1 * np.ones(d)
        elif isinstance(lb, int) or isinstance(lb, float):  # scalar. copy it
            self.lb = np.ones(d) * lb
        elif isinstance(lb, list) or isinstance(lb, np.ndarray):
            lb = np.asarray(lb)
            if lb.size == d:
                self.lb = lb
            else:
                raise ValueError(
                    "lb must be a scalar or array-like object" + "with d elements."
                )

        if ub is None:  # default is [-1, 1]^d
            self.ub = 1 * np.ones(d)
        elif isinstance(ub, int) or isinstance(ub, float):  # scalar. copy it
            self.ub = np.ones(d) * ub
        elif isinstance(ub, list) or isinstance(ub, np.ndarray):
            ub = np.asarray(ub)
            if ub.size == d:
                self.ub = ub
            else:
                raise ValueError(
                    "lb must be a scalar or array-like object" + "with d elements."
                )

        if d <= 1:
            raise ValueError("Number of dimensions must be >= 2")

        if isinstance(mu, int):  # Isotropic case
            if mu < 1:
                raise ValueError("The parameter mu needs to be > 1.")

            self.mu = mu
            self.inds = smol_inds(d, mu)
            self.pinds = poly_inds(d, mu, inds=self.inds)
            self.cube_grid = build_grid(self.d, self.mu, self.inds)
            self.grid = self.cube2dom(self.cube_grid)
            self.B = build_B(self.d, self.mu, self.cube_grid, self.pinds)

        else:  # Anisotropic case
            mu = np.asarray(mu)

            if any(mu < 1):
                raise ValueError("Each element in mu needs to be > 1.")

            if len(mu) != d:
                raise ValueError("For Anisotropic grid, mu must have len d ")

            self.mu = mu
            self.inds = smol_inds(d, mu)
            self.pinds = poly_inds(d, mu, inds=self.inds)
            self.cube_grid = build_grid(self.d, self.mu, self.inds)
            self.grid = self.cube2dom(self.cube_grid)
            self.B = build_B(self.d, self.mu, self.cube_grid, self.pinds)

        # Compute LU decomposition of B
        l, u = lu(self.B, True)  # pass permute_l as true. See scipy docs
        self.B_L = l
        self.B_U = u

    def __repr__(self):
        npoints = self.cube_grid.shape[0]
        nz_pts = np.count_nonzero(self.B)
        pct_nz = nz_pts / (npoints**2.0)

        if isinstance(self.mu, int):
            msg = "Smolyak Grid:\n\td: {0} \n\tmu: {1} \n\tnpoints: {2}"
            msg += "\n\tB: {3:.2f}% non-zero"
            return msg.format(self.d, self.mu, self.cube_grid.shape[0], pct_nz)
        else:  # Anisotropic grid
            msg = "Anisotropic Smolyak Grid:"
            msg += "\n\td: {0} \n\tmu: {1} \n\tnpoints: {2}"
            msg += "\n\tB: {3:.2f}% non-zero"
            mu_str = " x ".join(map(str, self.mu))
            return msg.format(self.d, mu_str, self.cube_grid.shape[0], pct_nz)

    def __str__(self):
        return self.__repr__()

    def dom2cube(self, pts):
        """
        Takes a point(s) and transforms it(them) into the [-1, 1]^d domain
        """
        # Could add some bounds checks to make sure points are in the
        # correct domain (between low and up bounds) and if right dim

        lb = self.lb
        ub = self.ub

        centers = lb + (ub - lb) / 2
        radii = (ub - lb) / 2

        trans_pts = (pts - centers) / radii

        return trans_pts

    def cube2dom(self, pts):
        """
        Takes a point(s) and transforms it(them) from domain [-1, 1]^d
        back into the desired domain
        """
        # Also could use some bounds checks/other stuff to make sure
        # that everything being passed in is viable

        lb = self.lb
        ub = self.ub

        centers = lb + (ub - lb) / 2
        radii = (ub - lb) / 2

        inv_trans_pts = pts * radii + centers

        return inv_trans_pts

    def plot_grid(self):
        """
        Beautifully plots the grid for the 2d and 3d cases

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        grid = self.grid
        if grid.shape[1] == 2:
            xs = grid[:, 0]
            ys = grid[:, 1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
            ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
            ax.plot(xs, ys, ".", markersize=6)
            ax.set_title("Smolyak grid: $d=%i, \; \\mu=%i$" % (self.d, self.mu))
            plt.show()
        elif grid.shape[1] == 3:
            xs = grid[:, 0]
            ys = grid[:, 1]
            zs = grid[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(xs, ys, zs)
            ax.set_title("Smolyak grid: $d=%i, \; \\mu=%i$" % (self.d, self.mu))
            plt.show()
        else:
            raise ValueError("Can only plot 2 or 3 dimensional problems")

        return fig
