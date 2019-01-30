from numba import jit, njit

@njit
def multilinear_interpolation_2d(smin, smax, orders, V, s, output):

    d = 2
    N = s.shape[0]

    order_0 = orders[0]
    order_1 = orders[1]

    for n in range(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-smin[0])/(smax[0]-smin[0])
            sn_1 = (s_1-smin[1])/(smax[1]-smin[1])

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1

            # v_ij: values on vertices of hypercube "containing" the point
            v_00 = V[(q_0) , (q_1)]
            v_01 = V[(q_0) , (q_1+1)]
            v_10 = V[(q_0+1) , (q_1)]
            v_11 = V[(q_0+1) , (q_1+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))

from numba import prange

@njit(parallel=True)
def multilinear_interpolation_2d_bis(smin, smax, orders, V, s, output):

    d = 2
    N = s.shape[0]

    order_0 = orders[0]
    order_1 = orders[1]

    a_0 = smin[0]
    a_1 = smin[1]

    b_0 = smax[0]
    b_1 = smax[1]

    # activate prange (optionnally)
    for n in prange(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-a_0)/(b_0-a_0)
            sn_1 = (s_1-a_1)/(b_1-a_1)

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1

            # v_ij: values on vertices of hypercube "containing" the point
            v_00 = V[(q_0) , (q_1)]
            v_01 = V[(q_0) , (q_1+1)]
            v_10 = V[(q_0+1) , (q_1)]
            v_11 = V[(q_0+1) , (q_1+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))

@njit(parallel=True)
def multilinear_interpolation_2d_grid(grid, V, s, output):

    N = s.shape[0]

    a_0 = grid[0][0]
    b_0 = grid[0][1]
    order_0 = grid[0][2]

    a_1 = grid[0][0]
    b_1 = grid[0][1]
    order_1 = grid[0][2]


    # activate prange (optionnally)
    for n in prange(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-a_0)/(b_0-a_0)
            sn_1 = (s_1-a_1)/(b_1-a_1)

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1

            # v_ij: values on vertices of hypercube "containing" the point
            v_00 = V[(q_0) , (q_1)]
            v_01 = V[(q_0) , (q_1+1)]
            v_10 = V[(q_0+1) , (q_1)]
            v_11 = V[(q_0+1) , (q_1+1)]

            # interpolated/extrapolated value
            output[n] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))



@njit(parallel=True)
def mult_multilinear_interpolation_2d_grid(grid, V, s, output):

    N = s.shape[0]
    n_x = V.shape[2]

    a_0 = grid[0][0]
    b_0 = grid[0][1]
    order_0 = grid[0][2]

    a_1 = grid[0][0]
    b_1 = grid[0][1]
    order_1 = grid[0][2]


    # activate prange (optionnally)
    for n in prange(N):

            # (s_1, ..., s_d) : evaluation point
            s_0 = s[ n, 0 ]
            s_1 = s[ n, 1 ]

            # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
            sn_0 = (s_0-a_0)/(b_0-a_0)
            sn_1 = (s_1-a_1)/(b_1-a_1)

            # q_k : index of the interval "containing" s_k
            q_0 = max( min( int(sn_0 *(order_0-1)), (order_0-2) ), 0 )
            q_1 = max( min( int(sn_1 *(order_1-1)), (order_1-2) ), 0 )

            # lam_k : barycentric coordinate in interval k
            lam_0 = sn_0*(order_0-1) - q_0
            lam_1 = sn_1*(order_1-1) - q_1

            # v_ij: values on vertices of hypercube "containing" the point
            for i_x in range(n_x):

                v_00 = V[(q_0) , (q_1), i_x]
                v_01 = V[(q_0) , (q_1+1), i_x]
                v_10 = V[(q_0+1) , (q_1), i_x]
                v_11 = V[(q_0+1) , (q_1+1), i_x]

                # interpolated/extrapolated value
                output[n, i_x] = (1-lam_0)*((1-lam_1)*(v_00) + (lam_1)*(v_01)) + (lam_0)*((1-lam_1)*(v_10) + (lam_1)*(v_11))


def mlininterp(grid: Tuple, c: Array, u: Array)->float:
    N = u.shape[0]
    res = np.zeros(N)
    for n in range(N):
        uu = extract_row(u, n, grid)
        # get indices and barycentric coordinates
        tmp = fmap(get_index, grid, uu)
        indices, barycenters = funzip(tmp)
        coeffs = get_coeffs(c, indices)
        res[n] = tensor_reduction(coeffs, barycenters)
    return res



def index(inds):
    return str.join('_',  [str(e) for e in inds] )

def rindex(binds):
    # M = ['M_{}*'.format(i) for i in range(len(binds)-1)] + ['']

    N = ['(q_{}{})'.format(n,'+1'*i) for n,i in enumerate(binds)]
    # return str.join(' , ',  [ str.join('', e) for e in zip(M,N) ])
    return str.join(' , ',  N )

def make_formula(d,ind,mm):
    if len(ind) == d:
        return 'v_{}'.format(index(ind))
    else:
        j = len(ind)
        ind1 = ind + (0,)
        ind2 = ind + (1,)
        s = "(1-λ_{j})*({a}) + (λ_{j})*({b})".format(j=j, a=make_formula(d,ind1,mm), b=make_formula(d,ind2,mm))
        return s

index([0,1,3])

rindex([0,2,3])

make_formula(2,tuple([]),None)

from interpolation.multilinear.fungen import gen_tensor_reduction

l
gen_tensor_reduction('C', ['l[{}]'.format(i) for i in range(2)])
# if __name__ == "__main__":

import numpy as np
smin = np.array([0.0, 0.0])
smax = np.array([1.0, 1.0])
orders = np.array( [50, 50])
C = np.random.random((50, 50))

N = 1000000
d = 2
s = np.random.random((N,d))

out = np.empty(N)

grid=(
    (0.0, 1.0, 50),
    (0.0, 1.0, 50),
)

%timeit multilinear_interpolation_2d(smin, smax, orders, C, s, out)
%timeit multilinear_interpolation_2d_bis(smin, smax, orders, C, s, out)


%timeit multilinear_interpolation_2d_grid(grid, C, s, out)

## multiple values
n_x = 8
CC = np.random.random((50,50, n_x))
out = np.empty((N,n_x))

%timeit mult_multilinear_interpolation_2d_grid(grid, CC, s, out)
%timeit mult_multilinear_interpolation_2d_grid_unroll(grid, CC, s, out)
