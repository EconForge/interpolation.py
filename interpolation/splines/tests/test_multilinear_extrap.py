def test_multilinear_extrap():

    import numpy as np

    N = 100000
    K = 10
    d = 1
    a = np.array([0.0] * d)
    b = np.array([1.0] * d)
    n = np.array([K] * d)
    grid_u = ((0.0, 1.0, K),) * d

    from interpolation.cartesian import mlinspace

    s = mlinspace(a, b, n)

    f = lambda x: (x**2).sum(axis=1)

    x = f(s)
    v = x.reshape(n)

    from interpolation.splines.eval_splines import eval_linear

    pp = np.linspace(-0.5, 1.5, 100)[:, None]

    from interpolation.splines.option_types import options

    xx_ = eval_linear(grid_u, v, pp)
    xx_cst = eval_linear(grid_u, v, pp, options.CONSTANT)
    xx_lin = eval_linear(grid_u, v, pp, options.LINEAR)
    xx_nea = eval_linear(grid_u, v, pp, options.NEAREST)

    yy_ = np.array([eval_linear(grid_u, v, p_) for p_ in pp])
    yy_cst = np.array([eval_linear(grid_u, v, p_, options.CONSTANT) for p_ in pp])
    yy_lin = np.array([eval_linear(grid_u, v, p_, options.LINEAR) for p_ in pp])
    yy_nea = np.array([eval_linear(grid_u, v, p_, options.NEAREST) for p_ in pp])

    zz_ = xx_.copy() * 0
    zz_cst = xx_cst.copy() * 0
    zz_lin = xx_lin.copy() * 0
    zz_nea = xx_nea.copy() * 0

    print("Inplace")
    eval_linear(grid_u, v, pp, zz_)
    eval_linear(grid_u, v, pp, zz_cst, options.CONSTANT)
    eval_linear(grid_u, v, pp, zz_lin, options.LINEAR)
    eval_linear(grid_u, v, pp, zz_nea, options.NEAREST)

    assert (abs(xx_ - yy_) + (abs(xx_ - zz_))).max() < 1e-10
    assert (abs(xx_cst - yy_cst) + (abs(xx_cst - zz_cst))).max() < 1e-10
    assert (abs(xx_lin - yy_lin) + (abs(xx_lin - zz_lin))).max() < 1e-10
    assert (abs(xx_nea - yy_nea) + (abs(xx_nea - zz_nea))).max() < 1e-10
