def test_readonly_vec():

    import numpy as np
    from interpolation.splines.eval_splines import eval_cubic

    vals = np.random.random((7, 7, 2))
    grid = ((-0.053333333333333344, 0.053333333333333344, 5), (5.0, 15.0, 5))

    # this works
    points = np.array([[0.0, 9.35497829]])
    x = eval_cubic(grid, vals, points)

    #
    pp = points.copy()
    pp.flags.writeable = False
    x = eval_cubic(grid, vals, pp)


def test_readonly():

    import numpy as np
    from interpolation.splines.eval_splines import eval_cubic

    vals = np.random.random((7, 7, 2))
    grid = ((-0.053333333333333344, 0.053333333333333344, 5), (5.0, 15.0, 5))

    # this works
    points = np.array([0.0, 9.35497829])
    x = eval_cubic(grid, vals, points)

    #
    pp = points.copy()
    pp.flags.writeable = False
    x = eval_cubic(grid, vals, pp)
