
def test_chebychev():

    import numpy as np

    from interpolation.linear_bases.basis_chebychev import ChebychevBasis

    def fun(x): return np.sin(x**2/5) + 2*x
    def dfun(x): return np.cos(x**2/5)*2*x/5 + 2

    xvec = np.linspace(-0.5, 5,200)
    yvec = fun(xvec)



    cb = ChebychevBasis(min=-0.5, max=5, n=5)

    coeffs = cb.filter(fun(cb.nodes))

    from matplotlib import pyplot as plt

    plt.figure()
    plt.subplot(121)
    Phi = cb.eval(xvec)
    ivec = Phi @ coeffs
    plt.plot(xvec, yvec)
    plt.plot(xvec,ivec)
    plt.plot(cb.nodes, fun(cb.nodes),'x')


    plt.subplot(122)
    d_yvec = dfun(xvec)
    d_Phi = cb.eval(xvec, orders=1)
    d_ivec = d_Phi @ coeffs
    plt.plot(xvec, d_yvec)
    plt.plot(xvec, d_ivec)
    plt.plot(cb.nodes, dfun(cb.nodes),'x')



    ff = np.concatenate( [fun(cb.nodes)[:,None] for i in range(2)], axis=1 )

    ff.shape
    coeffs = cb.filter(ff)
    coeffs.shape
    # plt.show()



def test_uniform_cubic_splines():
    import numpy as np

    from interpolation.linear_bases.basis_uniform_cubic_splines import UniformSplineBasis

    uf = UniformSplineBasis(0, 1, 10)
    f = lambda x: np.sin(3*x**2)
    x = np.linspace(0, 1, 10)
    C = uf.filter(f(uf.nodes))

    from matplotlib import pyplot as plt
    # %matplotlib inline

    res = ( uf.eval(0.5, orders=1) )
    res2 = ( uf.eval(0.5, orders=0) )
    res3 = ( uf.eval(0.5, orders=[0,1,2]) )


    vals = uf.eval(x).as_matrix()
    dvals = uf.eval(x, orders=1).as_matrix()
    d2vals = uf.eval(x, orders=2).as_matrix()



    plt.figure()
    plt.subplot(311)
    plt.plot(x, vals[:,1])
    plt.plot(x, vals[:,2])
    plt.plot(x, vals[:,3])
    plt.plot(x, vals[:,4])

    plt.subplot(312)
    plt.plot(x, dvals[:,1])
    plt.plot(x, dvals[:,2])
    plt.plot(x, dvals[:,3])
    plt.plot(x, dvals[:,4])


    di = (dvals[1:,:]-dvals[:-1,:])/(x[1]-x[0])
    plt.subplot(313)
    plt.plot(x, d2vals[:,1])
    plt.plot(x, d2vals[:,2])
    plt.plot(x, d2vals[:,3])
    plt.plot(x, d2vals[:,4])
    plt.plot(x[1:], di[:,4], color='cyan', linestyle=':')
    plt.ylim(-200,200)

    # plt.show()
    epsilon=1e-08

    di = (uf.eval(x+epsilon).as_matrix()-np.array(uf.eval(x).as_matrix()))/epsilon



    tab = (np.concatenate([dvals[:,4:5], di[:,4:5]], axis=1))

    di = (uf.eval(x+epsilon,orders=1).as_matrix()-np.array(uf.eval(x,orders=1).as_matrix()))/epsilon

    tab = (np.concatenate([d2vals[:,4:5], di[:,4:5]], axis=1))

    # print(tab.shape)
    # print(tab[500:1000])
    # print(abs(d2vals[1:,:]-di[1:,:]).max())
    # plt.plot(x, d2vals[:,4])
    # plt.plot(x[1:], di[:,4]) #-d2vals[1:,4], linestyle=':')
    # # plt.ylim(-10,10)
    # plt.plot(x, d2vals[:,4]) #-d2vals[1:,4], linestyle=':')
    # plt.show()



def test_compact_basis_matrix():

    from interpolation.linear_bases.compact_matrices import CompactBasisMatrix

    import numpy as np
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

    from interpolation.linear_bases.compact_matrices import CompactBasisArray, CompactBasisMatrix

    import numpy as np

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
