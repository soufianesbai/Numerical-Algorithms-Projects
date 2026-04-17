import pytest
import sys
import numpy as np

sys.path.append("../code")
import partie1

def test_complete_cholesky():
    """Tests the complete_cholesky function with valid and invalid inputs."""

    # --- Valid test cases ---

    A1 = np.array([[4, 2], [2, 3]])
    T1 = partie1.complete_cholesky(A1)
    assert np.allclose(T1 @ T1.T, A1), "Test failed for A1"

    A2 = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    T2 = partie1.complete_cholesky(A2)
    assert np.allclose(T2 @ T2.T, A2), "Test failed for A2"

    A3 = np.eye(3)
    T3 = partie1.complete_cholesky(A3)
    assert np.allclose(T3, A3), "Test failed for A3 (identity matrix)"

    A4 = np.array([[9]])
    T4 = partie1.complete_cholesky(A4)
    assert np.allclose(T4 @ T4.T, A4), "Test failed for A4 (1x1 matrix)"

    A5 = np.array([[42, 15, -5, -10], [15, 36, 0, 4], [-5, 0, 89, -3], [-10, 4, -3, 17]])
    T5 = partie1.complete_cholesky(A5)
    assert np.allclose(T5 @ T5.T, A5), "Test failed for A5"

    A6 = np.diag([1, 4, 9, 16, 25])
    T6 = partie1.complete_cholesky(A6)
    assert np.allclose(T6 @ T6.T, A6), "Test failed for A6 (diagonal matrix)"
    
    # --- Invalid test cases ---
    
    with pytest.raises(ValueError, match="symmetric"):
        partie1.complete_cholesky(np.array([[1, 2], [3, 4]]))  # Not symmetric

    with pytest.raises(ValueError, match="positive definite"):
        partie1.complete_cholesky(np.array([[1, 2], [2, 1]]))  # Symmetric but not positive definite

    print("Test complete_cholesky: Ok")

def test_generate_sparse_matrix():
    """Tests the generate_sparse_matrix function with different inputs."""

    n1, density1, low1, high1 = 5, 0.3, -10, 10
    M1 = partie1.generate_sparse_matrix(n1, density1, low1, high1)
    
    assert M1.shape == (n1, n1), "Test failed: Shape is incorrect"
    assert np.allclose(M1, M1.T), "Test failed: Matrix is not symmetric"
    
    n2, density2, low2, high2 = 10, 0.1, -5, 5
    M2 = partie1.generate_sparse_matrix(n2, density2, low2, high2)
    assert M2.shape == (n2, n2), "Test failed: Shape is incorrect"
    assert np.allclose(M2, M2.T), "Test failed: Matrix is not symmetric"
    expected_nonzero_max = int(density2 * (n2 * n2)) + n2
    assert np.count_nonzero(M2) <= expected_nonzero_max, f"Test failed: Too many nonzero elements ({np.count_nonzero(M2)} > {expected_nonzero_max})"

    n3, density3, low3, high3 = 6, 0.9, 1, 20
    M3 = partie1.generate_sparse_matrix(n3, density3, low3, high3)

    assert M3.shape == (n3, n3), "Test failed: Shape is incorrect"
    assert np.allclose(M3, M3.T), "Test failed: Matrix is not symmetric"
    assert np.all(M3 >= low3) and np.all(M3 <= high3), "Test failed: Values out of bounds"

    n4, density4, low4, high4 = 1, 1.0, -3, 3
    M4 = partie1.generate_sparse_matrix(n4, density4, low4, high4)

    assert M4.shape == (1, 1), "Test failed: Shape is incorrect"
    assert M4[0, 0] >= low4 and M4[0, 0] <= high4, "Test failed: Value out of bounds"

    print("All generate_sparse_matrix tests passed")

def test_incomplete_cholesky():
    """Tests the incomplete_cholesky function."""
    
    A1 = np.array([[4, 2, 2],
                   [2, 4, 2],
                   [2, 2, 4]], dtype=float)
    
    T1 = partie1.incomplete_cholesky(A1)
    
    # Check if T1 is lower triangular
    assert np.allclose(T1, np.tril(T1)), "Test failed: T1 is not lower triangular"
    
    # Check if T1 * T1.T is close to A1
    assert np.allclose(np.dot(T1, T1.T), A1), "Test failed: T1 * T1.T is not close to A1"

    A2 = np.array([[4, 1, 2, 3],
                   [1, 4, 1, 2],
                   [2, 1, 4, 1],
                   [3, 2, 1, 4]], dtype=float)
    
    T2 = partie1.incomplete_cholesky(A2)
    
    # Check if T2 is lower triangular
    assert np.allclose(T2, np.tril(T2)), "Test failed: T2 is not lower triangular"
    
    # Check if T2 * T2.T is close to A2
    assert np.allclose(np.dot(T2, T2.T), A2), "Test failed: T2 * T2.T is not close to A2"

    A3 = np.array([[0, 1, 2],
                   [1, 4, 1],
                   [2, 1, 4]], dtype=float)
    
    T3 = partie1.incomplete_cholesky(A3)
    
    # Check if T3 is lower triangular
    assert np.allclose(T3, np.tril(T3)), "Test failed: T3 is not lower triangular"
    
    assert np.allclose(T3[1:, :1], 0), "Test failed: Non-zero elements found outside lower triangular part"


    A4 = np.array([[6, 3, 3],
                   [3, 6, 3],
                   [3, 3, 6]], dtype=float)
    
    T4 = partie1.incomplete_cholesky(A4)
    
    # Check if T4 is lower triangular
    assert np.allclose(T4, np.tril(T4)), "Test failed: T4 is not lower triangular"
    
    # Check if T4 * T4.T is close to A4
    assert np.allclose(np.dot(T4, T4.T), A4), "Test failed: T4 * T4.T is not close to A4"
    
    print("All incomplete_cholesky tests passed")


def test_is_well_conditioned():
    """Tests the is_well_conditioned function."""
    
    n = 5
    # Create a symmetric positive definite (SPD) matrix
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  
    A += n * np.eye(n) 
    
    result_complete = partie1.is_well_conditioned(A, fact=0)
    result_incomplete = partie1.is_well_conditioned(A, fact=1)
    
    print("is_well_conditioned (complete):", result_complete)
    print("is_well_conditioned (incomplete):", result_incomplete)
    print("Test is_well_conditioned: OK")


def test_is_well_conditioned_verbose():
    """Tests the is_well_conditioned function with detailed condition number analysis."""
    
    n = 5
    # Create a symmetric positive definite (SPD) matrix
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  
    A += n * np.eye(n)  

    M_complete = partie1.complete_cholesky(A)
    M_incomplete = partie1.incomplete_cholesky(A)

    res_complete = np.linalg.inv(M_complete).T @ np.linalg.inv(M_complete)
    res_incomplete = np.linalg.inv(M_incomplete).T @ np.linalg.inv(M_incomplete)

    cond_A = np.linalg.cond(A)
    cond_res_complete = np.linalg.cond(res_complete)
    cond_res_incomplete = np.linalg.cond(res_incomplete)

    print("Condition number of A            :", cond_A)
    print("Condition number after complete CH:", cond_res_complete)
    print("Condition number after incomplete CH:", cond_res_incomplete)



if __name__ == "__main__":
    test_complete_cholesky()
    test_generate_sparse_matrix()
    test_incomplete_cholesky()
    test_is_well_conditioned()
    test_is_well_conditioned_verbose()

