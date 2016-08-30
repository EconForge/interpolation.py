
def test_kron_compact_basis_matrix():

    from interpolation.linear_bases.compact_matrices import CompactBasisMatrix
    from interpolation.linear_bases.compact_matrices import CompactKroneckerProduct

    import numpy as np
    inds = [2, 3, 3]
    vals = [[0.1, 0.2]*2, [-0.1, -0.2]*2, [1.1, 2.1]*2]
    cbm_1 = CompactBasisMatrix(inds, vals, m=8)
    cbm_2 = CompactBasisMatrix(inds, vals, m=8)
    ckp = CompactKroneckerProduct(matrices=[cbm_1, cbm_2])
    c = np.random.random((8,8,6))


def test_kron_compact_basis_array():


    from interpolation.linear_bases.compact_matrices import CompactBasisArray
    from interpolation.linear_bases.compact_matrices import CompactKroneckerProduct

    import numpy as np

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


def test_kronecker_times():

    import numpy as np

    from interpolation.linear_bases.kronecker import kronecker_times_compact, kronecker_times_compact_diff

    N = 100000
    ind_0 = np.random.randint(6,size=N)
    ind_1 = np.random.randint(6,size=N)
    m_0 = 12
    m_1 = 20
    A = np.random.random((N,4))
    B = np.random.random((N,4))
    c = np.random.random((12,20,2))

    res = kronecker_times_compact(((ind_0,A,m_0), (ind_1, B, m_1)), c)

    # test 2

    A = np.random.random((N,4,2))
    B = np.random.random((N,4,2))
    c = np.random.random((12,20,2))

    res = kronecker_times_compact_diff(((ind_0,A,m_0), (ind_1, B, m_1)), c)
