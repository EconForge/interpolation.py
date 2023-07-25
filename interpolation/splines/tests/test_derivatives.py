def test_derivatives():
    from interpolation.splines.eval_splines import eval_spline, eval_cubic, eval_linear

    import numpy as np

    grid = ((0.0, 1.0, 10),)

    # grid = ((0.0, 1.0, 0.1),(0.0, 1.0, 0.1))

    C = np.linspace(0, 1, 10)
    Cx = np.concatenate([C[:, None], C[:, None] * 2])
    points = np.random.random((10, 1))

    eval_spline(
        grid, C, (-0.1,), out=None, k=1, diff="None", extrap_mode="nearest"
    )  # no alloc
    eval_spline(
        grid, C, (-0.1,), out=None, k=1, diff="None", extrap_mode="constant"
    )  # no alloc
    eval_spline(
        grid, C, (-0.1,), out=None, k=1, diff="None", extrap_mode="linear"
    )  # no alloc

    eval_spline(
        grid, C, (1.1,), out=None, k=1, diff="None", extrap_mode="nearest"
    )  # no alloc
    eval_spline(
        grid, C, (1.1,), out=None, k=1, diff="None", extrap_mode="constant"
    )  # no alloc
    eval_spline(
        grid, C, (1.1,), out=None, k=1, diff="None", extrap_mode="linear"
    )  # no alloc

    eval_spline(
        grid, Cx, points[0, :], out=None, k=1, diff="None", extrap_mode="linear"
    )

    eval_spline(grid, C, points, out=None, k=1, diff="None", extrap_mode="linear")
    eval_spline(grid, Cx, points, out=None, k=1, diff="None", extrap_mode="linear")

    orders = str(((0,), (1,)))

    eval_spline(
        grid, C, points[0, :], out=None, k=1, diff=orders, extrap_mode="linear"
    )  # no alloc
    eval_spline(
        grid, Cx, points[0, :], out=None, k=1, diff=orders, extrap_mode="linear"
    )
    eval_spline(grid, C, points, out=None, k=1, diff=orders, extrap_mode="linear")
    eval_spline(grid, Cx, points, out=None, k=1, diff=orders, extrap_mode="linear")

    out = eval_spline(
        grid, Cx, points, out=None, k=1, diff=orders, extrap_mode="linear"
    )
    out2 = np.zeros_like(out)
    eval_spline(grid, Cx, points, out=out2, k=1, diff=orders, extrap_mode="linear")
    print(abs(out - out2).max())

    k = 3

    eval_spline(grid, C, points[0, :], out=None, k=3, diff="None", extrap_mode="linear")

    eval_spline(grid, C, points, out=None, k=k, diff="None", extrap_mode="linear")
    eval_spline(
        grid, Cx, points[0, :], out=None, k=k, diff="None", extrap_mode="linear"
    )
    eval_spline(grid, Cx, points, out=None, k=k, diff="None", extrap_mode="linear")

    orders = str(((0,), (1,)))

    eval_spline(grid, C, points[0, :], out=None, k=k, diff=orders, extrap_mode="linear")
    eval_spline(grid, C, points, out=None, k=k, diff=orders, extrap_mode="linear")
    eval_spline(
        grid, Cx, points[0, :], out=None, k=k, diff=orders, extrap_mode="linear"
    )
    eval_spline(grid, Cx, points, out=None, k=k, diff=orders, extrap_mode="linear")
