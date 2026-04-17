import pytest
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.append("../code")
import partie3


def test_laplacian():
    for N in range(2, 21):
        A = partie3.laplacian(N, 1 / (N + 1))
        A_dense = A.toarray()

        # Verify diagonal values are -4
        assert np.all(np.diagonal(A_dense) == -4), f"Test failed for diagonal in N={N}"

        # Verify adjacent elements are 1, except for borders
        for i in range(N**2):
            if i % N != 0 and i % N != N - 1:  # Not on borders
                assert (
                    A_dense[i, i - 1] == 1
                ), f"Test failed for N={N}, element at ({i},{i-1})"
                assert (
                    A_dense[i, i + 1] == 1
                ), f"Test failed for N={N}, element at ({i},{i+1})"
            if i // N != 0 and i // N != N - 1:  # Not on top or bottom
                assert (
                    A_dense[i, i - N] == 1
                ), f"Test failed for N={N}, element at ({i},{i-N})"
                assert (
                    A_dense[i, i + N] == 1
                ), f"Test failed for N={N}, element at ({i},{i+N})"

    print("All laplacien tests passed!")

def test_solve_heat_equation():
    """Tests the solve_heat_equation function."""
    N = 3

    F = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  

    T_sol = partie3.solve_heat_equation(F, N)

    expected_T = np.zeros((N, N))

    assert np.allclose(T_sol, expected_T), f"Test failed! Computed T: {T_sol}, Expected T: {expected_T}"


    F = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])  

    T_sol = partie3.solve_heat_equation(F, N)

    assert T_sol.shape == (N, N), f"Test failed! The shape of the solution is incorrect: {T_sol.shape}"
    assert np.isfinite(T_sol).all(), "Test failed! The solution contains NaN or Inf values"

    print("All solve_heat_equation tests passed")


if __name__ == "__main__":
    test_laplacian()
    test_solve_heat_equation()

