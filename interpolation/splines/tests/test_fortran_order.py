def test_eval_with_fortran_order():

    import numpy as np

    C_F = np.random.random((50, 50)).T
    C = C_F.copy()

    print(C.flags)
    print(C_F.flags)

    p = np.random.random((10, 2))

    p_F = p.T.copy().T

    from interpolation.splines import eval_linear

    out_F_C = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C_F, p)
    out_C_C = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C, p)

    out_F_F = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C_F, p_F)
    out_C_F = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C, p_F)

    assert np.array_equal(out_F_C, out_C_C)
    assert np.array_equal(out_F_C, out_F_F)
    assert np.array_equal(out_F_C, out_C_F)


def test_guvectorize_compatilibity():

    ### this tests type dispatch when order='A'

    import numpy as np
    from numba import guvectorize, f8

    from interpolation.splines import eval_linear

    # The function to interpolate
    xs = np.linspace(0, 10, 101)
    values = np.sin(xs)

    # The points to interpolate
    points = np.array([[v] for v in range(11)], dtype=np.float64)

    # The output array
    out = np.zeros(11, dtype=np.float64)

    # Wrap eval_linear() in a guvectorized function
    @guvectorize([(f8[:], f8[:], f8[:, :], f8[:])], "(i),(i),(j,k)->(j)", nopython=True)
    def wrapper(xs, values, points, out):
        """Calls eval_linear()."""
        eval_linear((xs,), values, points, out)

    wrapper(xs, values, points, out)
