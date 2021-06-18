import numpy as np
from numba import jit, njit
from numba import generated_jit
from numba.extending import overload


from distutils.version import LooseVersion
from numba import __version__

if LooseVersion(__version__) >= "0.43":
    overload_options = {"strict": False}
else:
    overload_options = {}

# used by njitted routines (frozen)
basis = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0)

#
@njit
def solve_deriv_interp_1d(bands, coefs):

    M = coefs.shape[0] - 2

    # Solve interpolating equations
    # First and last rows are different

    bands[0, 1] /= bands[0, 0]
    bands[0, 2] /= bands[0, 0]
    bands[0, 3] /= bands[0, 0]
    bands[0, 0] = 1.0
    bands[1, 1] -= bands[1, 0] * bands[0, 1]
    bands[1, 2] -= bands[1, 0] * bands[0, 2]
    bands[1, 3] -= bands[1, 0] * bands[0, 3]
    bands[0, 0] = 0.0
    bands[1, 2] /= bands[1, 1]
    bands[1, 3] /= bands[1, 1]
    bands[1, 1] = 1.0

    # Now do rows 2 through M+1
    for row in range(2, M + 1):
        bands[row, 1] -= bands[row, 0] * bands[row - 1, 2]
        bands[row, 3] -= bands[row, 0] * bands[row - 1, 3]
        bands[row, 2] /= bands[row, 1]
        bands[row, 3] /= bands[row, 1]
        bands[row, 0] = 0.0
        bands[row, 1] = 1.0

    # Do last row
    bands[M + 1, 1] -= bands[M + 1, 0] * bands[M - 1, 2]
    bands[M + 1, 3] -= bands[M + 1, 0] * bands[M - 1, 3]
    bands[M + 1, 2] -= bands[M + 1, 1] * bands[M, 2]
    bands[M + 1, 3] -= bands[M + 1, 1] * bands[M, 3]
    bands[M + 1, 3] /= bands[M + 1, 2]
    bands[M + 1, 2] = 1.0

    coefs[M + 1] = bands[(M + 1), 3]
    # Now back substitute up
    for row in range(M, 0, -1):
        coefs[row] = bands[row, 3] - bands[row, 2] * coefs[row + 1]

    # Finish with first row
    coefs[0] = bands[0, 3] - bands[0, 1] * coefs[1] - bands[0, 2] * coefs[2]


@njit
def find_coefs_1d(δ, data, coefs, bands):

    M = bands.shape[0] - 2

    # Setup boundary conditions

    # Left boundary
    abcd_left = (1.0 * δ * δ, -2.0 * δ * δ, 1.0 * δ * δ, 0.0)

    # Right boundary
    abcd_right = (1.0 * δ * δ, -2.0 * δ * δ, 1.0 * δ * δ, 0.0)

    for i in range(4):
        bands[0, i] = abcd_left[i]
        bands[M + 1, i] = abcd_right[i]

    for i in range(M):
        for j in range(3):
            bands[i + 1, j] = basis[j]
            bands[i + 1, 3] = data[i]

    solve_deriv_interp_1d(bands, coefs)


def _filter_cubic():
    pass


# non allocating version
@overload(_filter_cubic, **overload_options)
def __filter_cubic(grid, D, C):

    d = len(grid.types)

    if D.ndim > d:

        def ___filter_cubic(grid, D, C):

            n_x = C.shape[-1]
            for i_x in range(n_x):
                _filter_cubic(grid, D[..., i_x], C[..., i_x])

        return ___filter_cubic

    if d == 1:

        def ___filter_cubic(grid, D, C):

            dinv_0 = (grid[0][1] - grid[0][0]) / grid[0][2]

            Mx = D.shape[0]

            Nx = Mx + 2

            # First, solve in the X-direction
            bands = np.zeros((Mx + 2, 4))
            find_coefs_1d(dinv_0, D, C, bands)

        return ___filter_cubic

    if d == 2:

        def ___filter_cubic(grid, D, C):

            dinv_0 = (grid[0][1] - grid[0][0]) / grid[0][2]
            dinv_1 = (grid[1][1] - grid[1][0]) / grid[1][2]

            Mx = D.shape[0]
            My = D.shape[1]

            Nx = Mx + 2
            Ny = My + 2

            # First, solve in the X-direction
            bands = np.zeros((Mx + 2, 4))
            for iy in range(My):
                find_coefs_1d(dinv_0, D[:, iy], C[:, iy], bands)

            # Now, solve in the Y-direction
            bands = np.zeros((My + 2, 4))
            for ix in range(Nx):
                find_coefs_1d(dinv_0, C[ix, :], C[ix, :], bands)

        return ___filter_cubic

    if d == 3:

        def ___filter_cubic(grid, D, C):

            dinv_0 = (grid[0][1] - grid[0][0]) / grid[0][2]
            dinv_1 = (grid[1][1] - grid[1][0]) / grid[1][2]
            dinv_2 = (grid[2][1] - grid[2][0]) / grid[2][2]

            Mx = D.shape[0]
            My = D.shape[1]
            Mz = D.shape[2]

            Nx = Mx + 2
            Ny = My + 2
            Nz = Mz + 2

            bands = np.zeros((Mx + 2, 4))
            for iy in range(My):
                for iz in range(Mz):
                    find_coefs_1d(dinv_0, D[:, iy, iz], C[:, iy, iz], bands)

            # Now, solve in the Y-direction
            bands = np.zeros((My + 2, 4))
            for ix in range(Nx):
                for iz in range(Mz):
                    find_coefs_1d(dinv_1, C[ix, :, iz], C[ix, :, iz], bands)

            # Now, solve in the Z-direction
            bands = np.zeros((Mz + 2, 4))
            for ix in range(Nx):
                for iy in range(Ny):
                    find_coefs_1d(dinv_2, C[ix, iy, :], C[ix, iy, :], bands)

        return ___filter_cubic

    if d == 4:

        def ___filter_cubic(grid, D, C):

            dinv_0 = (grid[0][1] - grid[0][0]) / grid[0][2]
            dinv_1 = (grid[1][1] - grid[1][0]) / grid[1][2]
            dinv_2 = (grid[2][1] - grid[2][0]) / grid[2][2]
            dinv_3 = (grid[3][1] - grid[3][0]) / grid[3][2]

            Mx = D.shape[0]
            My = D.shape[1]
            Mz = D.shape[2]
            Mz4 = D.shape[3]

            Nx = Mx + 2
            Ny = My + 2
            Nz = Mz + 2
            Nz4 = Mz4 + 2

            # First, solve in the X-direction
            bands = np.zeros((Mx + 2, 4))
            for iy in range(My):
                for iz in range(Mz):
                    for iz4 in range(Mz4):
                        find_coefs_1d(
                            dinv_0, D[:, iy, iz, iz4], C[:, iy, iz, iz4], bands
                        )

            # Now, solve in the Y-direction
            bands = np.zeros((My + 2, 4))
            for ix in range(Nx):
                for iz in range(Mz):
                    for iz4 in range(Mz4):
                        find_coefs_1d(
                            dinv_1, C[ix, :, iz, iz4], C[ix, :, iz, iz4], bands
                        )

            # Now, solve in the Z-direction
            bands = np.zeros((Mz + 2, 4))
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz4 in range(Mz4):
                        find_coefs_1d(
                            dinv_2, C[ix, iy, :, iz4], C[ix, iy, :, iz4], bands
                        )

            # Now, solve in the Z4-direction
            bands = np.zeros((Mz4 + 2, 4))
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        find_coefs_1d(dinv_3, C[ix, iy, iz, :], C[ix, iy, iz, :], bands)

        return ___filter_cubic


# allocating version
@overload(_filter_cubic)
def __filter_cubic(grid, D):

    d = len(grid.types)
    if D.ndim == d:
        if D.ndim == 1:

            def ___filter_cubic(grid, D):
                C = np.zeros(D.shape[0] + 2)
                _filter_cubic(grid, D, C)
                return C

        if D.ndim == 2:

            def ___filter_cubic(grid, D):
                C = np.zeros((D.shape[0] + 2, D.shape[1] + 2))
                _filter_cubic(grid, D, C)
                return C

        if D.ndim == 3:

            def ___filter_cubic(grid, D):
                C = np.zeros((D.shape[0] + 2, D.shape[1] + 2, D.shape[2] + 2))
                _filter_cubic(grid, D, C)
                return C

        if D.ndim == 4:

            def ___filter_cubic(grid, D):
                C = np.zeros(
                    (D.shape[0] + 2, D.shape[1] + 2, D.shape[2] + 2, D.shape[3] + 2)
                )
                _filter_cubic(grid, D, C)
                return C

        return ___filter_cubic
    elif D.ndim == d + 1:
        if d == 1:

            def ___filter_cubic(grid, D):
                C = np.zeros((D.shape[0] + 2, D.shape[1]))
                _filter_cubic(grid, D, C)
                return C

        if d == 2:

            def ___filter_cubic(grid, D):
                C = np.zeros((D.shape[0] + 2, D.shape[1] + 2, D.shape[2]))
                _filter_cubic(grid, D, C)
                return C

        if d == 3:

            def ___filter_cubic(grid, D):
                C = np.zeros(
                    (D.shape[0] + 2, D.shape[1] + 2, D.shape[2] + 2, D.shape[3])
                )
                _filter_cubic(grid, D, C)
                return C

        if d == 4:

            def ___filter_cubic(grid, D):
                C = np.zeros(
                    (
                        D.shape[0] + 2,
                        D.shape[1] + 2,
                        D.shape[2] + 2,
                        D.shape[3] + 2,
                        D.shape[4],
                    )
                )
                _filter_cubic(grid, D, C)
                return C

        return ___filter_cubic


@njit
def prefilter_cubic(*args):
    return _filter_cubic(*args)


filter_cubic = prefilter_cubic


def _prefilter():
    pass


import numba

none = numba.typeof(None)


@numba.extending.overload(_prefilter)
def _ov_prefilter(grid, V, k, out=None):

    if isinstance(k, numba.types.Literal):

        if k.literal_value == 1:

            def _impl_prefilter(grid, V, k, out=None):
                return V  # should we copy it here ?

            return _impl_prefilter

        if k.literal_value == 3:

            def _impl_prefilter(grid, V, k, out=None):
                if out is None:
                    return prefilter_cubic(grid, V)
                else:
                    return prefilter_cubic(grid, V, out)

            return _impl_prefilter
    else:

        def ugly_workaround(grid, V, k, out=None):
            return (numba.literally(k),) ^ (k / 2)


@njit
def prefilter(grid, V, out=None, k=3):
    return _prefilter(grid, V, numba.literally(k), out=out)
