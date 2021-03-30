def test_eval_with_fortran_order():

    import numpy as np

    C_F = np.random.random( (50, 50) ).T
    C = C_F.copy()

    print(C.flags)
    print(C_F.flags)

    p = np.random.random((10,2))

    p_F = p.T.copy().T


    from interpolation.splines import eval_linear

    out_F_C = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C_F, p)
    out_C_C = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C, p)

    out_F_F = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C_F, p_F)
    out_C_F = eval_linear(((0.0, 1.0, 48), (0.0, 1.0, 48)), C, p_F)

    assert np.array_equal(out_F_C, out_C_C)
    assert np.array_equal(out_F_C, out_F_F)
    assert np.array_equal(out_F_C, out_C_F)