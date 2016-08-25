from numba import float64, int64
import numpy as np

class CompactBasisMatrix:

    def __init__(self, indices: int64[:], vals: float64[:, :], m=None):
        # inds: 1d array of integers (for each line, column index of first nnz element)
        # vals: 2d array of floats
        # m: number of bases functions

        inds = np.array(indices,dtype=int)
        vals = np.array(vals,dtype=float)
        assert(vals.ndim==2)
        assert(inds.shape[0]==vals.shape[0])
        if m is None:
            m = inds.max() + vals.shape[1]
        self.m = m
        self.shape = (len(inds), self.m)
        self.inds = inds
        self.vals = vals

    def as_matrix(self, sparse=False):
        nm = self.vals.shape[1]
        cols = (self.inds[:, None].repeat(nm, axis=1) +
                np.arange(nm)[None, :].repeat(self.shape[0], axis=0)).ravel()
        rows = np.arange(self.shape[0])[:, None].repeat(nm, axis=1).ravel()
        if not sparse:
            res = np.zeros(self.shape)
            res[rows, cols] = self.vals.ravel()
        else:
            from scipy.sparse import coo_matrix
            res = coo_matrix((self.vals.ravel(), (rows, cols)), shape=self.shape)

        return res

    def as_spmatrix(self):
        return self.as_matrix(sparse=True)


class CompactBasisArray:

    def __init__(self, indices, vals, m=None):
        # vals: tuple of 2d arrays
        inds = np.array(indices, dtype=int)
        vals = [np.array(v, dtype=float) for v in vals]
        shape = {mat.shape for mat in vals}
        assert(len(shape)==1)
        shape = shape.pop()
        assert(inds.shape[0] == shape[0])
        if m is None:
            m = inds.max() + shape[1]
        self.m = m
        self.inds = inds
        self.vals = vals
        self.q = len(vals) # number of derivatives
        self.shape = (len(inds), self.m, self.q)
#
    @property
    def matrices(self):
        return [CompactBasisMatrix(self.inds, v, self.m) for v in self.vals]

    def as_array(self):
        return np.concatenate([m.as_matrix()[:,:,None] for m in self.matrices], axis=2)

from interpolation.linear_bases.kronecker import kronecker_times_compact, kronecker_times_compact_diff

class CompactKroneckerProduct:

    def __init__(self, matrices): #: List[CompactBasisMatrix]):
        # right now, arrays is a list of vectors or matrices
        # TODO: allow for 3d-array
        self.matrices = matrices
        self.d = len(matrices)


    def __mul__(self, c):
        if self.matrices[0].vals.shape[1] != 4:
            raise Exception("This is only implemented for cubic splines.")
        c = np.asarray(c)
        if c.ndim == self.d:
            return self.__mul__(c[...,None])[...,0]
        else:
            if len(self.matrices[0].shape) == 2:
                matrices = tuple([(mat.inds,mat.vals,mat.m)  for mat in self.matrices])
                return kronecker_times_compact(matrices, c)
            else:
                matrices = tuple([(mat.inds,mat.as_array(),mat.m)  for mat in self.matrices])
                return kronecker_times_compact_diff(matrices, c)





## Tests



def test_compact_basis_matrix():
    inds = [2, 3, 3]
    vals = [[0.1, 0.2], [-0.1, -0.2], [1.1, 2.1]]
    cbm = CompactBasisMatrix(inds, vals)
    sol = np.array([[ 0. ,  0. ,  0.1,  0.2,  0. ],
           [ 0. ,  0. ,  0. , -0.1, -0.2],
           [ 0. ,  0. ,  0. ,  1.1,  2.1]])
    from numpy.testing import assert_equal
    assert_equal(sol, cbm.as_matrix())
    assert_equal(sol, cbm.as_spmatrix().todense())


def test_compact_basis_array():

    from numpy.testing import assert_equal
    A = np.zeros((4,2)) + 0.1
    B = np.zeros((4,2)) + 0.2
    C = np.zeros((4,2)) + 0.3

    inds = [1,2,0,1]

    cba = CompactBasisArray(inds, [A,B,C], m=5)

    mat = CompactBasisMatrix(inds, C, m=5).as_matrix()
    assert_equal( cba.matrices[2].as_matrix(), mat )
    assert_equal(mat, cba.as_array()[:,:,2])


def test_kron_compact_basis_matrix():
    inds = [2, 3, 3]
    vals = [[0.1, 0.2]*2, [-0.1, -0.2]*2, [1.1, 2.1]*2]
    cbm_1 = CompactBasisMatrix(inds, vals, m=8)
    cbm_2 = CompactBasisMatrix(inds, vals, m=8)
    ckp = CompactKroneckerProduct(matrices=[cbm_1, cbm_2])
    c = np.random.random((8,8,6))


def test_kron_compact_basis_array():

    from numpy.testing import assert_equal
    N = 6
    A = np.zeros((N,4)) + 0.1
    B = np.zeros((N,4)) + 0.2
    C = np.zeros((N,4)) + 0.3

    inds = [1]*6

    cba_1 = CompactBasisArray(inds, [A,B,C], m=7)
    cba_2 = CompactBasisArray(inds, [A,B,C], m=6)

    ckp = CompactKroneckerProduct(matrices=[cba_1, cba_2])

    c = np.random.random((7,6,3))

if __name__ == '__main__':

    import time
    t1 = time.time()
    test_compact_basis_array()
    test_compact_basis_matrix()
    test_kron_compact_basis_matrix()
    test_kron_compact_basis_array()
    t2 = time.time()
    print("Elapsed : {}".format(t2-t1))
