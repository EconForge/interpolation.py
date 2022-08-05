import numpy as np

from interpolation.complete_poly import CompletePolynomial
from interpolation.complete_poly import (
    n_complete,
    complete_polynomial,
    complete_polynomial_der,
    _complete_poly_impl,
    _complete_poly_impl_vec,
    _complete_poly_der_impl,
    _complete_poly_der_impl_vec,
)


def test_complete_scalar():
    def f(x, y):
        return x

    points = np.random.random((1000, 2))
    vals = f(points[:, 0], points[:, 1])

    cp = CompletePolynomial(2, 3)
    cp.fit_values(points, vals)

    evals = cp(points)

    assert evals.shape == vals.shape
    assert abs(evals - vals).max() < 1e-10

    cp.fit_values(points, vals, damp=0.5)


def test_complete_vector():
    def f(x, y):
        return x

    def f2(x, y):
        return x**3 - y

    points = np.random.random((1000, 2))
    vals = np.column_stack(
        [f(points[:, 0], points[:, 1]), f2(points[:, 0], points[:, 1])]
    )

    cp = CompletePolynomial(2, 3)
    cp.fit_values(points, vals)

    evals = cp(points)

    assert evals.shape == vals.shape
    assert abs(evals - vals).max() < 1e-10

    cp.fit_values(points, vals, damp=0.5)


def test_complete_derivative():

    # Test derivative vector
    z = np.array([1, 2, 3])
    sol_vec = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    out_vec = np.empty(n_complete(3, 2))
    _complete_poly_der_impl_vec(z, 2, 0, out_vec)
    assert abs(out_vec - sol_vec).max() < 1e-10

    # Test derivative matrix
    z = np.arange(1, 7).reshape(3, 2)
    out_mat = complete_polynomial_der(z, 2, 1)
    assert abs(out_mat[0, :]).max() < 1e-10
    assert abs(out_mat[2, :] - np.ones(2)).max() < 1e-10
    assert abs(out_mat[-2, :] - np.array([5.0, 6.0])).max() < 1e-10


def test_complete_vec_vs_mat():
    # Matrix for allocation
    temp = np.ones(n_complete(2, 3)) * 5.0
    temp_mat = np.ones((n_complete(2, 3), 3))

    # Point at which to evaluate
    z = np.array([0.9, 1.05])
    z_mat = np.array([[0.9, 0.95, 1.0], [1.05, 1.0, 0.95]])

    foo = complete_polynomial(z, 2)
    bar = complete_polynomial(z_mat, 2)[:, 0]
    assert np.allclose(foo, bar)

    foo = complete_polynomial_der(z, 2, 0)
    bar = complete_polynomial_der(z_mat, 2, 0)[:, 0]
    assert np.allclose(foo, bar)

    foo = complete_polynomial_der(z, 4, 0)
    bar = complete_polynomial_der(z_mat, 4, 0)[:, 0]
    assert np.allclose(foo, bar)


if __name__ == "__main__":
    test_complete_scalar()
    test_complete_vector()
    test_complete_derivative()
    test_complete_vec_vs_mat()
