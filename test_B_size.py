from interpolation.smolyak.grid import build_grid, smol_inds, poly_inds, cheby2n, m_i, reduce, mul

import numpy as np

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
    B = np.empty((npts, npolys), order='F')
    for ind, comb in enumerate(b_inds):
        B[:, ind] = reduce(mul, [Ts[comb[i] - 1, i, :]
                           for i in range(d)])

    if deriv:
        # TODO: test this. I am going to bed.
        Us = cheby2n(pts.T, m_i(max_mu + 1), kind=2.0)
        Us = np.concatenate([np.zeros((1, d, npts)), Us], axis=0)
        for i in range(Us.shape[0]):
            Us[i, :, :] = Us[i, :, :] * i

        der_B = np.zeros((npolys, d, npts))

        for i in range(d):
            for ind, comb in enumerate(b_inds):
                der_B[ind, i, :] = reduce(mul, [(Ts[comb[k] - 1, k, :] if i != k
                                          else Us[comb[k] - 1, k, :])
                                          for k in range(d)])

        return B, der_B



    return B

d = 12
mu = 3
grid = build_grid(d,mu)
grid.shape

Phi = build_B(d,mu,grid)

import scipy.sparse

Phi.shape

Phi[abs(Phi)<1e-7] = 0

Phi_S = scipy.sparse.csc_matrix(Phi)
# Phi_S[0,:].todense()
Phi_S.nnz/(Phi_S.shape[0]*Phi_S.shape[1])

Phi.nbytes/1e6

ss = (abs(Phi)>1e-6).sum(axis=1)

ss.max()/Phi.shape[0]

import scipy.sparse.linalg
from scipy.sparse.linalg import spsolve

dx = np.random.random(grid.shape[0])

%time spsolve(Phi_S, dx, use_umfpack=False)

%time scipy.sparse.linalg.bicg(Phi_S, dx)

# %time scipy.sparse.linalg.cg(Phi_S, dx)

%time scipy.sparse.linalg.gmres(Phi_S,dx)

%time scipy.sparse.linalg.minres(Phi_S, dx)

%time scipy.sparse.linalg.qmr(Phi_S, dx)


%time np.linalg.solve(Phi, dx)
