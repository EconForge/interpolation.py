def test_eval_splines():

    from interpolation.splines.eval_splines import eval_linear, eval_cubic
    from interpolation.multilinear.mlinterp import mlinterp
    import numpy as np

    N = 1000
    K = 100
    d = 2
    grid = ((0.0, 1.0, K),) * d
    grid_nu = ((0.0, 1.0, K), np.linspace(0, 1, K))
    n_x = 6

    V = np.random.random((K, K))
    VV = np.random.random((K, K, n_x))

    C = np.random.random((K + 2, K + 2))
    CC = np.random.random((K + 2, K + 2, n_x))

    points = np.random.random((N, 2))

    out = np.random.random((N))
    Cout = np.random.random((N, n_x))

    eval_linear(grid, V, points, out)
    # eval_cubic(grid, C, points[0,:], out[0,0])
    eval_linear(grid, VV, points, Cout)
    eval_linear(grid, VV, points[0, :], Cout[0, :])

    mlinterp(grid, V, points)
    eval_linear(grid, V, points)
    eval_linear(grid, V, points)
    eval_linear(grid, V, points, out)

    print("OK 2")
    eval_linear(grid, V, points[0, :])
    res_0 = eval_linear(grid, VV, points)
    res_1 = eval_linear(grid, VV, points[0, :])

    # nonuniform grid:
    res_0_bis = eval_linear(grid_nu, VV, points)
    res_1_bis = eval_linear(grid_nu, VV, points[0, :])

    assert abs(res_0 - res_0_bis).max() < 1e-10
    assert abs(res_1 - res_1_bis).max() < 1e-10

    print("OK 3")
    eval_cubic(grid, C, points, out)
    # eval_cubic(grid, C, points[0,:], out[0,0])
    eval_cubic(grid, CC, points, Cout)
    eval_cubic(grid, CC, points[0, :], Cout[0, :])

    print("OK 4")
    eval_cubic(grid, C, points)
    eval_cubic(grid, C, points[0, :])
    eval_cubic(grid, CC, points)
    eval_cubic(grid, CC, points[0, :])

    print("OK 5")

    ####
    ###
    ###
