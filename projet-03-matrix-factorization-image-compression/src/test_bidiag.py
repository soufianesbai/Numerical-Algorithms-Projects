import numpy as np
import bidiag

def test_bidiagonal():
    """Test the bidiagonalization function with various test cases."""

    # Test case 1: Identity matrix (should remain identity after bidiagonalization)
    A1 = np.eye(5)  # Identity matrix of size 5x5
    Q_left1, BD1, Q_right1 = bidiag.bidiagonalize_matrix(A1)
    
    assert np.allclose(A1, np.dot(Q_left1, np.dot(BD1, Q_right1)), atol=1e-10)
    assert np.allclose(Q_left1 @ Q_left1.T, np.eye(A1.shape[0]), atol=1e-10)
    assert np.allclose(Q_right1 @ Q_right1.T, np.eye(A1.shape[1]), atol=1e-10)

    # Test case 2: Random 5x5 matrix
    np.random.seed(42)  # Fix seed for reproducibility
    A2 = np.random.rand(5, 5)
    Q_left2, BD2, Q_right2 = bidiag.bidiagonalize_matrix(A2)

    # Check reconstruction
    assert np.allclose(A2, np.dot(Q_left2, np.dot(BD2, Q_right2)), atol=1e-10)
    assert np.allclose(Q_left2 @ Q_left2.T, np.eye(A2.shape[0]), atol=1e-10)
    assert np.allclose(Q_right2 @ Q_right2.T, np.eye(A2.shape[1]), atol=1e-10)

    # Test case 3: Structured 5x5 matrix with increasing values
    A3 = np.array(np.arange(1, 26).reshape(5, 5), dtype='float64')
    Q_left3, BD3, Q_right3 = bidiag.bidiagonalize_matrix(A3)

    # Zero out small numerical errors for better comparison
    BD3[np.isclose(BD3, 0)] = 0
    result3 = np.dot(Q_left3, np.dot(BD3, Q_right3))
    result3[np.isclose(result3, 0)] = 0

    # Check reconstruction 
    assert np.allclose(A3, result3, atol=1e-10)
    assert np.allclose(Q_left3 @ Q_left3.T, np.eye(A3.shape[0]), atol=1e-10)
    assert np.allclose(Q_right3 @ Q_right3.T, np.eye(A3.shape[1]), atol=1e-10)

    print("All bidiagonalization tests passed successfully!")

test_bidiagonal()
