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
            tmp *= n + j

        # out += tmp // math.factorial(i + 1)
        out += tmp // denom
    return out


#
# Complete Polynomials Basis
#
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
    if d > 5:
        raise ValueError("Complete polynomial only implemeted up to degree 5")

    # Assure z is array
    z = np.asarray(z)

    # compute inds allocate space for output
    if np.ndim(z) == 1:
        nvar = z.size
        out = np.zeros(n_complete(nvar, d))
        # populate out with jitted function
        _complete_poly_impl_vec(z, d, out)
    else:
        nvar, nobs = z.shape
        out = np.zeros((n_complete(nvar, d), nobs))
        # populate out with jitted function
        _complete_poly_impl(z, d, out)

    return out


# TODO: Currently turning off all cache arguments so that
#       code works. This will be fixed in numba 0.32
# @jit(nopython=True, cache=True)
@jit(nopython=True)
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


# @jit(nopython=True, cache=True)
@jit(nopython=True)
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
                            out[ix, k] = z[i1, k] * z[i2, k] * z[i3, k] * z[i4, k]

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
                            out[ix, k] = z[i1, k] * z[i2, k] * z[i3, k] * z[i4, k]

                        for i5 in range(i4, nvar):
                            ix += 1
                            for k in range(nobs):
                                out[ix, k] = (
                                    z[i1, k] * z[i2, k] * z[i3, k] * z[i4, k] * z[i5, k]
                                )

        return


#
# Complete Polynomials Derivative Basis
#
def complete_polynomial_der(z, d, der):
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

    der : int
        An integer specifying which variable to take derivative wrt --
        a 0 means take derivative wrt first variable in z etc...

    Returns
    -------
    out : np.ndarray(size=(ncomplete(nvariables, d), nobservations))
        The basis matrix for the derivative of a complete polynomial
        of degree d with respect to variable der

    """
    # check inputs
    assert d >= 0, "d must be non-negative"
    assert der >= 0, "derivative must be non-negative"
    if d > 5:
        raise ValueError("Complete polynomial only implemeted up to degree 5")

    # Ensure z is a numpy array
    z = np.asarray(z)

    # compute inds allocate space for output
    if np.ndim(z) == 1:
        nvar = z.size
        assert der < nvar, "derivative integer must be smaller than nobs in z"
        out = np.zeros(n_complete(nvar, d))
        # populate with jitted function
        _complete_poly_der_impl_vec(z, d, der, out)
    else:
        nvar, nobs = z.shape
        assert der < nvar, "derivative integer must be smaller than nobs in z"
        out = np.zeros((n_complete(nvar, d), nobs))
        # populate out with jitted function
        _complete_poly_der_impl(z, d, der, out)

    return out


# @jit(nopython=True, cache=True)
@jit(nopython=True)
def _complete_poly_der_impl_vec(z, d, der, out):
    "out and z should be vectors"
    nvar = z.shape[0]

    out[0] = 0.0

    # fill first order stuff
    if d >= 1:
        # All linear terms except for one (the variable itself) are 0
        for i in range(nvar):
            out[i + 1] = 0.0
        out[der + 1] = 1.0

    if d == 1:
        return

    # now we need to fill in row nvar and beyond
    ix = nvar
    if d == 2:
        for i1 in range(nvar):
            # Get coefficients and values
            (c1, t1) = (1, 1.0) if i1 == der else (0, z[i1])
            for i2 in range(i1, nvar):
                (c2, t2) = (c1 + 1, 1.0) if i2 == der else (c1, z[i2])

                # Update index and out
                ix += 1
                out[ix] = c2 * t1 * t2 * z[der] ** (c2 - 1) if c2 > 0 else 0.0

        return

    if d == 3:
        for i1 in range(nvar):
            # Get coefficients and values
            (c1, t1) = (1, 1.0) if i1 == der else (0, z[i1])
            for i2 in range(i1, nvar):
                (c2, t2) = (c1 + 1, 1.0) if i2 == der else (c1, z[i2])
                ix += 1
                out[ix] = c2 * t1 * t2 * z[der] ** (c2 - 1) if c2 > 0 else 0.0

                for i3 in range(i2, nvar):
                    (c3, t3) = (c2 + 1, 1.0) if i3 == der else (c2, z[i3])
                    ix += 1
                    out[ix] = c3 * t1 * t2 * t3 * z[der] ** (c3 - 1) if c3 > 0 else 0.0

        return

    if d == 4:
        for i1 in range(nvar):
            # Get coefficients and values
            (c1, t1) = (1, 1.0) if i1 == der else (0, z[i1])
            for i2 in range(i1, nvar):
                (c2, t2) = (c1 + 1, 1.0) if i2 == der else (c1, z[i2])
                ix += 1
                out[ix] = c2 * t1 * t2 * z[der] ** (c2 - 1) if c2 > 0 else 0.0

                for i3 in range(i2, nvar):
                    (c3, t3) = (c2 + 1, 1.0) if i3 == der else (c2, z[i3])
                    ix += 1
                    out[ix] = c3 * t1 * t2 * t3 * z[der] ** (c3 - 1) if c3 > 0 else 0.0

                    for i4 in range(i3, nvar):
                        (c4, t4) = (c3 + 1, 1.0) if i4 == der else (c3, z[i4])
                        ix += 1
                        out[ix] = (
                            c4 * t1 * t2 * t3 * t4 * z[der] ** (c4 - 1)
                            if c4 > 0
                            else 0.0
                        )

        return

    if d == 5:
        for i1 in range(nvar):
            # Get coefficients and values
            (c1, t1) = (1, 1.0) if i1 == der else (0, z[i1])
            for i2 in range(i1, nvar):
                (c2, t2) = (c1 + 1, 1.0) if i2 == der else (c1, z[i2])
                ix += 1
                out[ix] = c2 * t1 * t2 * z[der] ** (c2 - 1) if c2 > 0 else 0.0

                for i3 in range(i2, nvar):
                    (c3, t3) = (c2 + 1, 1.0) if i3 == der else (c2, z[i3])
                    ix += 1
                    out[ix] = c3 * t1 * t2 * t3 * z[der] ** (c3 - 1) if c3 > 0 else 0.0

                    for i4 in range(i3, nvar):
                        (c4, t4) = (c3 + 1, 1.0) if i4 == der else (c3, z[i4])
                        ix += 1
                        out[ix] = (
                            c4 * t1 * t2 * t3 * t4 * z[der] ** (c4 - 1)
                            if c4 > 0
                            else 0.0
                        )

                        for i5 in range(i4, nvar):
                            (c5, t5) = (c4 + 1, 1.0) if i5 == der else (c4, z[i5])
                            ix += 1
                            out[ix] = (
                                c5 * t1 * t2 * t3 * t4 * t5 * z[der] ** (c5 - 1)
                                if c5 > 0
                                else 0.0
                            )

        return


# @jit(nopython=True, cache=True)
@jit(nopython=True)
def _complete_poly_der_impl(z, d, der, out):
    nvar = z.shape[0]
    nobs = z.shape[1]

    for k in range(nobs):
        out[0, k] = 0.0

    # fill first order stuff
    if d >= 1:
        # Make sure everything has zeros in it
        for i in range(nvar):
            for k in range(nobs):
                out[i + 1, k] = 0.0

        # Then place ones where they belong in variable
        for k in range(nobs):
            out[der + 1, k] = 1.0

    if d == 1:
        return

    # now we need to fill in row nvar and beyond
    ix = nvar
    if d == 2:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                    c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                    out[ix, k] = c2 * t1 * t2 * z[der, k] ** (c2 - 1) if c2 > 0 else 0.0

        return

    if d == 3:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                    c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                    out[ix, k] = c2 * t1 * t2 * z[der, k] ** (c2 - 1) if c2 > 0 else 0.0

                for i3 in range(i2, nvar):
                    ix += 1
                    for k in range(nobs):
                        c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                        c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                        c3, t3 = (c2 + 1, 1.0) if i3 == der else (c2, z[i3, k])
                        out[ix, k] = (
                            c3 * t1 * t2 * t3 * z[der, k] ** (c3 - 1) if c3 > 0 else 0.0
                        )

        return

    if d == 4:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                    c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                    out[ix, k] = c2 * t1 * t2 * z[der, k] ** (c2 - 1) if c2 > 0 else 0.0

                for i3 in range(i2, nvar):
                    ix += 1
                    for k in range(nobs):
                        c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                        c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                        c3, t3 = (c2 + 1, 1.0) if i3 == der else (c2, z[i3, k])
                        out[ix, k] = (
                            c3 * t1 * t2 * t3 * z[der, k] ** (c3 - 1) if c3 > 0 else 0.0
                        )

                    for i4 in range(i3, nvar):
                        ix += 1
                        for k in range(nobs):
                            c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                            c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                            c3, t3 = (c2 + 1, 1.0) if i3 == der else (c2, z[i3, k])
                            c4, t4 = (c3 + 1, 1.0) if i4 == der else (c3, z[i4, k])
                            out[ix, k] = (
                                c4 * t1 * t2 * t3 * t4 * z[der, k] ** (c4 - 1)
                                if c4 > 0
                                else 0.0
                            )

        return

    if d == 5:
        for i1 in range(nvar):
            for i2 in range(i1, nvar):
                ix += 1
                for k in range(nobs):
                    c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                    c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                    out[ix, k] = c2 * t1 * t2 * z[der, k] ** (c2 - 1) if c2 > 0 else 0.0

                for i3 in range(i2, nvar):
                    ix += 1
                    for k in range(nobs):
                        c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                        c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                        c3, t3 = (c2 + 1, 1.0) if i3 == der else (c2, z[i3, k])
                        out[ix, k] = (
                            c3 * t1 * t2 * t3 * z[der, k] ** (c3 - 1) if c3 > 0 else 0.0
                        )

                    for i4 in range(i3, nvar):
                        ix += 1
                        for k in range(nobs):
                            c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                            c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                            c3, t3 = (c2 + 1, 1.0) if i3 == der else (c2, z[i3, k])
                            c4, t4 = (c3 + 1, 1.0) if i4 == der else (c3, z[i4, k])
                            out[ix, k] = (
                                c4 * t1 * t2 * t3 * t4 * z[der, k] ** (c4 - 1)
                                if c4 > 0
                                else 0.0
                            )

                        for i5 in range(i4, nvar):
                            ix += 1
                            for k in range(nobs):
                                c1, t1 = (1, 1.0) if i1 == der else (0, z[i1, k])
                                c2, t2 = (c1 + 1, 1.0) if i2 == der else (c1, z[i2, k])
                                c3, t3 = (c2 + 1, 1.0) if i3 == der else (c2, z[i3, k])
                                c4, t4 = (c3 + 1, 1.0) if i4 == der else (c3, z[i4, k])
                                c5, t5 = (c4 + 1, 1.0) if i5 == der else (c4, z[i5, k])
                                out[ix, k] = (
                                    c5 * t1 * t2 * t3 * t4 * t5 * z[der, k] ** (c5 - 1)
                                    if c5 > 0
                                    else 0.0
                                )

        return


class CompletePolynomial:
    def __init__(self, n, d):
        self.n = n
        self.d = d

    def fit_values(self, s, x, damp=0.0):
        Phi = complete_polynomial(s.T, self.d).T
        self.Phi = Phi
        if damp == 0.0:
            self.coefs = np.ascontiguousarray(lstsq(Phi, x)[0])
        else:
            new_coefs = np.ascontiguousarray(lstsq(Phi, x)[0])
            self.coefs = (1 - damp) * new_coefs + damp * self.coefs

    def der(self, s, der):
        dPhi = complete_polynomial_der(s.T, self.d, der).T
        return np.dot(dPhi, self.coefs)

    def __call__(self, s):

        Phi = complete_polynomial(s.T, self.d).T
        return np.dot(Phi, self.coefs)
