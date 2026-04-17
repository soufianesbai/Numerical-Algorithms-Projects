import sys

sys.path.append("../code")
import partie2
import numpy as np


def test_conjugate_gradient():
    """Tests the conjugate gradient method with various test cases."""
    
    A1 = np.array([[1, 0], [0, 1]])
    B1 = np.array([2, 3])
    X1 = np.zeros_like(B1)
    X1_sol = partie2.conjugate_gradient(A1, B1, X1)
    assert np.allclose(A1 @ X1_sol, B1)

    A2 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    B2 = np.array([1, 0, 1])
    X2 = np.zeros_like(B2)
    X2_sol = partie2.conjugate_gradient(A2, B2, X2)
    assert np.allclose(A2 @ X2_sol, B2)

    A3 = np.array([[4, 0], [0, 5]])
    B3 = np.array([8, 10])
    X3 = np.zeros_like(B3).reshape(-1, 1)
    X3_sol = partie2.conjugate_gradient(A3, B3, X3)
    assert np.allclose(A3 @ X3_sol, B3)

    A4 = np.array(
        [
            [10, -1, 2, 0, 3],
            [-1, 11, -1, 3, 2],
            [2, -1, 10, -1, -2],
            [0, 3, -1, 8, 1],
            [3, 2, -2, 1, 9],
        ]
    )
    B4 = np.array([7, -4, 6, 5, -3])
    X4 = np.zeros_like(B4).reshape(-1, 1)
    X4_sol = partie2.conjugate_gradient(A4, B4, X4)
    assert np.allclose(A4 @ X4_sol, B4)

    print("All conjugate_gradient tests passed")


def test_preconditioned_conjugate_gradient():
    """Tests the preconditioned conjugate gradient method with various test cases."""
    
    A5 = np.array([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    B5 = np.array([8, 10, 12])
    X5 = np.zeros_like(B5)
    eps = 1e-6
    nmax = 100
    X5_sol = partie2.preconditioned_conjugate_gradient(A5, B5, X5, eps, nmax)
    assert np.allclose(A5 @ X5_sol, B5)

    A6 = np.array([[10, -1, 0, 0], [-1, 11, -1, 0], [0, -1, 12, -1], [0, 0, -1, 13]])
    B6 = np.array([1, 2, 3, 4])
    X6 = np.zeros_like(B6)
    eps = 1e-6
    nmax = 200
    X6_sol = partie2.preconditioned_conjugate_gradient(A6, B6, X6, eps, nmax)
    assert np.allclose(A6 @ X6_sol, B6)

    print("All preconditioned_conjugate_gradient tests passed")


if __name__ == "__main__":
    test_conjugate_gradient()
    test_preconditioned_conjugate_gradient()
