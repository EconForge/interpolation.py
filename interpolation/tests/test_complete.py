import numpy as np
import numpy as np

from interpolation.complete_poly import CompletePolynomial

def test_complete_scalar():

    def f(x, y): return x

    points = np.random.random((1000, 2))
    vals = f(points[:, 0], points[:, 1])

    cp = CompletePolynomial(2, 3)
    cp.fit_values(points, vals)

    evals = cp(points)

    assert(evals.shape == vals.shape)
    assert(abs(evals - vals).max() < 1e-10)

    cp.fit_values(points, vals, damp=0.5)



def test_complete_vector():

    def f(x, y): return x
    def f2(x, y): return x**3 - y

    points = np.random.random((1000, 2))
    vals = np.column_stack([
            f(points[:, 0], points[:, 1]),
            f2(points[:, 0], points[:, 1])
        ])

    cp = CompletePolynomial(2, 3)
    cp.fit_values(points, vals)

    evals = cp(points)

    assert(evals.shape == vals.shape)
    assert(abs(evals - vals).max() < 1e-10)

    cp.fit_values(points, vals, damp=0.5)


if __name__ == '__main__':
    test_complete_scalar()
    test_complete_vector()
