def test_derivatives():

    from ..interp import SmolyakInterp
    from ..grid import SmolyakGrid

    d = 5
    N = 100
    mu = 2
    f = lambda x: (x).sum(axis=1)

    import numpy.random

    ub = numpy.random.random(d) + 6
    lb = numpy.random.random(d) - 5
    sg = SmolyakGrid(d, mu, lb=lb, ub=ub)

    values = f(sg.grid)
    si = SmolyakInterp(sg, values)
    gg = numpy.random.random((N, d))

    res, res_s, res_c, res_x = si.interpolate(
        gg, deriv=True, deriv_th=True, deriv_X=True
    )

    T = sg.grid.shape[0]

    assert res.shape == (N,)
    assert res_s.shape == (N, d)
    assert res_c.shape == (N, T)
    assert res_x.shape == (N, T)

    # res_s should be identically 1
    assert abs(res_s - 1.0).max() < 1e-8

    epsilon = 1e-6

    # Test derivatives w.r.t. values

    si2 = SmolyakInterp(sg, values)

    def ff(y):
        x = y.reshape(values.shape)
        si2.update_theta(x)
        return si2.interpolate(gg).ravel()

    y0 = values.ravel()
    r0 = ff(y0)
    jac = numpy.zeros((len(r0), len(y0)))
    for n in range(len(y0)):
        yi = y0.copy()
        yi[n] += epsilon
        jac[:, n] = (ff(yi) - r0) / epsilon
    jac = jac.reshape((N, T))
    assert abs(jac - res_x).max() < 1e-7
    # note that accuracy of either numerical or direct computation is not very accurate

    # Test derivatives w.r.t. coefficients

    theta_0 = si.theta.copy()

    def ff_c(y_c):
        si2.theta = y_c.reshape(theta_0.shape)
        return si2.interpolate(gg).ravel()

    r0 = ff_c(theta_0)
    jac = numpy.zeros((len(r0), len(theta_0)))
    for n in range(len(y0)):
        ti = theta_0.copy()
        ti[n] += epsilon
        jac[:, n] = (ff_c(ti) - r0) / epsilon
    jac = jac.reshape((N, T))

    assert abs(jac - res_c).max() < 1e-7
