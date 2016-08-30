from interpolation.linear_bases.kronecker import kronecker_times_compact, kronecker_times_compact_diff
from numba import float64, int64
import numpy as np

class CompactBasisMatrix:

    def __init__(self, indices: int64[:], vals: float64[:, :], m=None):
        # inds: 1d array of integers (for each line, column index of first nnz element)
        # vals: 2d array of floats
        # m: number of bases functions

        inds = np.array(indices,dtype=int)
        vals = np.array(vals,dtype=float)
        if inds.ndim==0 and vals.ndim==1:
            inds = inds[None,...]
            vals = vals[None,...]
        assert(vals.ndim==2)
        assert(inds.shape[0]==vals.shape[0])
        if m is None:
            m = inds.max() + vals.shape[1]
        self.m = m
        self.shape = (len(inds), self.m)
        self.inds = inds
        self.vals = vals

    def __matmul__(self, mat):
        return self.as_matrix() @ mat

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

    def __init__(self, indices: int64[:], vals: float64[:, :, :], m: int64):
        # vals: tuple of 2d arrays
        inds = np.array(indices, dtype=int)
        vals = np.array(vals, dtype=float)
        # assert(len(shape)==1)
        # shape = shape.pop()
        # assert(inds.shape[0] == shape[0])
        self.m = m
        self.inds = inds
        self.vals = vals
        self.q = vals.shape[2] # number of derivatives
        self.shape = (len(inds), self.m, self.q)
#
    @property
    def matrices(self):
        print(self.inds.shape)
        print(self.vals.shape)
        print(self.shape)
        return [CompactBasisMatrix(self.inds, self.vals[:,:,i], self.m) for i in range(self.vals.shape[2])]

    def as_array(self):
        return np.concatenate([m.as_matrix()[:,:,None] for m in self.matrices], axis=2)


class CompactKroneckerProduct:

    def __init__(self, matrices): #: List[CompactBasisMatrix]):
        # right now, arrays is a list of vectors or matrices
        # TODO: allow for 3d-array
        self.matrices = matrices
        self.d = len(matrices)


    def as_matrix(self):

        from interpolation.linear_bases.product import KroneckerProduct
        return KroneckerProduct([m.as_matrix() for m in self.matrices]).as_matrix()

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
