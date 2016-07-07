"""
Function approximation using complete polynomials

@author : Spencer Lyon
@date : 2016-01-21 17:06

"""
import numpy as np
from scipy.linalg import lstsq
from numba import jit

# does not list monomials in the right order
#
# def complete_inds(n, d):
#     """
#     Return all combinations of powers in an n dimensional d degree
#     complete polynomial. This will include a term for all 0th order
#     variables (i.e. a constant)
#
#     Parameters
#     ----------
#     n : int
#         The number of parameters in the polynomial
#
#     d : int
#         The degree of the complete polynomials
#
#     Returns
#     -------
#     inds : filter
#         A python filter object that contains all the indices inside a
#         generator
#
#     """
#     i = itertools.product(*[range(d + 1) for i in range(n)])
#     return filter(lambda x: sum(x) <= d, i)


@jit(nopython=True)
def n_complete(n, d):
    """
    Return the number of terms in a complete polynomial of degree d in n
    variables

    Parameters
    ----------
    n : int
        The number of parameters in the polynomial

    d : int
        The degree of the complete polynomials

    Returns
    -------
    m : int
        The number of terms in the polynomial

    See Also
    --------
    See `complete_inds`

    """
    out = 1
    denom = 1
    for i in range(d):
        tmp = 1
        denom *= i + 1
        for j in range(i + 1):
            tmp *= (n + j)

        # out += tmp // math.factorial(i + 1)
        out += tmp // denom
    return out


def complete_polynomial(z, d):
    """
    Construct basis matrix for complete polynomial of degree `d`, given
    input data `z`.

    Parameters
    ----------
    z : np.ndarray(size=(nvariables, nobservations))
        The degree 1 realization of each variable. For example, if
        variables are `q`, `r`, and `s`, then `z` should be
        `z = np.row_stack([q, r, s])`

    d : int
        An integer specifying the degree of the complete polynomial

    Returns
    -------
    out : np.ndarray(size=(ncomplete(nvariables, d), nobservations))
        The basis matrix for a complete polynomial of degree d

    """
    # check inputs
    assert d >= 0, "d must be non-negative"
    z = np.asarray(z)

    # compute inds allocate space for output
    nvar, nobs = z.shape
    out = np.zeros((n_complete(nvar, d), nobs))

    if d > 5:
        raise ValueError("Complete polynomial only implemeted up to degree 5")

    # populate out with jitted function
    _complete_poly_impl(z, d, out)

    return out


@jit(nopython=True, cache=True)
def _complete_poly_impl_vec(z, d, out):
    "out and z should be vectors"
    nvar = z.shape[0]

    out[0] = 1.0

    # fill first order stuff
    if d >= 1:
        for i in range(1, nvar + 1):
            out[i] = z[i - 1]

    if d == 1:
        return

    # now we need to fill in row nvar and beyond
    ix = nvar
    if d == 2:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                out[ix] = z[i1] * z[i2]

        return

    if d == 3:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                out[ix] = z[i1] * z[i2]

                for i3 in range(i2, nvar):
                    ix += 1
                    out[ix] = z[i1] * z[i2] * z[i3]

        return

    if d == 4:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                out[ix] = z[i1] * z[i2]

                for i3 in range(i2, nvar):
                    ix += 1
                    out[ix] = z[i1] * z[i2] * z[i3]

                    for i4 in range(i3, nvar):
                        ix += 1
                        out[ix] = z[i1] * z[i2] * z[i3] * z[i4]

        return

    if d == 5:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                out[ix] = z[i1] * z[i2]

                for i3 in range(i2, nvar):
                    ix += 1
                    out[ix] = z[i1] * z[i2] * z[i3]

                    for i4 in range(i3, nvar):
                        ix += 1
                        out[ix] = z[i1] * z[i2] * z[i3] * z[i4]

                        for i5 in range(i4, nvar):
                            ix += 1
                            out[ix] = z[i1] * z[i2] * z[i3] * z[i4] * z[i5]

        return


@jit(nopython=True, cache=True)
def _complete_poly_impl(z, d, out):
    nvar = z.shape[0]
    nobs = z.shape[1]

    for k in range(nobs):
        out[0, k] = 1.0

    # fill first order stuff
    if d >= 1:
        for i in range(1, nvar + 1):
            for k in range(nobs):
                out[i, k] = z[i - 1, k]

    if d == 1:
        return

    # now we need to fill in row nvar and beyond
    ix = nvar
    if d == 2:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    out[ix, k] = z[i1, k] * z[i2, k]

        return

    if d == 3:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    out[ix, k] = z[i1, k] * z[i2, k]

                for i3 in range(i2, nvar):
                    ix += 1
                    for k in range(nobs):
                        out[ix, k] = z[i1, k] * z[i2, k] * z[i3, k]

        return

    if d == 4:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    out[ix, k] = z[i1, k] * z[i2, k]

                for i3 in range(i2, nvar):
                    ix += 1
                    for k in range(nobs):
                        out[ix, k] = z[i1, k] * z[i2, k] * z[i3, k]

                    for i4 in range(i3, nvar):
                        ix += 1
                        for k in range(nobs):
                            out[ix, k] = (z[i1, k] * z[i2, k] * z[i3, k] *
                                          z[i4, k])

        return

    if d == 5:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    out[ix, k] = z[i1, k] * z[i2, k]

                for i3 in range(i2, nvar):
                    ix += 1
                    for k in range(nobs):
                        out[ix, k] = z[i1, k] * z[i2, k] * z[i3, k]

                    for i4 in range(i3, nvar):
                        ix += 1
                        for k in range(nobs):
                            out[ix, k] = (z[i1, k] * z[i2, k] * z[i3, k] *
                                          z[i4, k])

                        for i5 in range(i4, nvar):
                            ix += 1
                            for k in range(nobs):
                                out[ix, k] = (z[i1, k] * z[i2, k] * z[i3, k] *
                                              z[i4, k] * z[i5, k])

        return


class CompletePolynomial:

    def __init__(self, n, d):
        self.n = n
        self.d = d

    def fit_values(self, s, x):
        Phi = complete_polynomial(s.T, self.d).T
        self.Phi = Phi
        self.coefs = np.ascontiguousarray(lstsq(Phi, x)[0])

    def __call__(self, s):

        Phi = complete_polynomial(s.T, self.d).T
        return np.dot(Phi, self.coefs)
