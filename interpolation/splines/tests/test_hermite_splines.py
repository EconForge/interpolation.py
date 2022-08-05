def test_hermite_splines():

    from interpolation.splines.hermite import HermiteInterpolationVect
    import numpy as np

    N = 10000  # Number of points in the initial dataset
    K = 1000  # Number of new points to interpolate

    # Initial dataset
    # grid = ((0.0, 1.0, K),)        # Creation of an x-axis grid (xi)
    grid = np.linspace(0.0, 1.0, N)  # Creation of an x-axis grid (xi)
    points = np.random.random((N))  # Random values for f(xi)
    dpoints = np.random.random((N))  # Random derivatives for f'(xi)

    # Generate new points
    newgrid = np.random.random((K))

    # Interpolation
    out = HermiteInterpolationVect(newgrid, grid, points, dpoints)

    print("OK")
