def test_bases_1():

    import numpy as np
    from numpy.testing import assert_equal, assert_almost_equal

    from interpolation.linear_bases.basis_chebychev import ChebychevBasis
    from interpolation.linear_bases.basis_linear import UniformLinearSplineBasis, LinearSplineBasis
    from interpolation.linear_bases.basis_uniform_cubic_splines import UniformSplineBasis
    # from interpolation.linear_bases.basis_splines import UniformSplineBasis as UniformSplineBasis_2

    def fun(x): return np.sin(x**2/5) + 2*x
    def dfun(x): return np.cos(x**2/5)*2*x/5 + 2

    Bases = [
        ChebychevBasis,
        UniformLinearSplineBasis,
        LinearSplineBasis
        UniformSplineBasis,  # cubic splines
        ]
        
    n = 200

    xvec = np.linspace(-0.5, 5, n)
    yvec = fun(xvec)

    x2vec = np.linspace(-0.5, 5, 10000)

    for Basis in Bases:

        print(Basis)
        # cb = Basis(min=-0.5, max=5, n=5)
        try:
            cb = Basis(-0.5, 5, n)
        except:
            cb = Basis(np.linspace(-0.5,5,n))

        assert(cb.n == n)

        vals = fun(cb.nodes)

        assert(vals.shape==(n,))

        print("Scalar valued interpolation")
        coeffs = cb.filter(vals)
        assert(coeffs.shape == (cb.m,))

        print("Test fitting matrix")
        B = cb.B
        assert(B.shape==(cb.m,cb.m))
        offset = (cb.m-cb.n)/2
        vals_with_bc = np.zeros(cb.m)
        vals_with_bc[offset:offset+cb.n] = vals

        try:
            # does not work with compact matrix
            coeffs_2 = np.linalg.solve(B, vals_with_bc)
        except:
            coeffs_2 = np.linalg.solve(B.as_matrix(), vals_with_bc)

        assert_almost_equal(coeffs, coeffs_2)


        print("Evaluating on the nodes")
        Phi = cb.Phi(cb.nodes)
        interp_vals = Phi@coeffs
        print(interp_vals-vals)
        assert_almost_equal(interp_vals, vals)


        print("Evaluating on a fine grid")
        Phi = cb.Phi(x2vec)
        interp_vals = Phi@coeffs
        true_vals = fun(x2vec)
        err = (abs(interp_vals-true_vals)).max()
        print(err)
        assert(err<1e-3)

        print("Test vector evaluation")
        mvals = np.concatenate([vals[:,None],vals[:,None]*2],axis=1)
        mcoeffs = cb.filter(mvals)
        interp_mvals = Phi@mcoeffs
        print(interp_mvals[:,0]-interp_vals)
        assert_almost_equal(interp_mvals[:,0], interp_vals)
        assert_almost_equal(interp_mvals[:,1], interp_vals*2)
