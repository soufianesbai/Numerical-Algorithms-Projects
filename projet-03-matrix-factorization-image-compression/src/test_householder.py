import numpy as np
import householder

def test_projection():
    """Test the projection function."""
    U = np.array([1, 2, 3], dtype=float)
    V = np.array([1, 2, 3], dtype=float)
    
    # Case U == V 
    result = householder.projection(U, V)
    expected = np.zeros_like(U)
    assert np.allclose(result, expected), f"Test failed for U == V: {result}"

    # General case ( U != V )
    V = np.array([4, 5, 6], dtype=float)
    result = householder.projection(U, V)
    expected = (U - V) / np.linalg.norm(U - V)  
    assert np.allclose(result, expected), f"Test failed for U != V: {result}"
    
    print("All projection tests passed successfully!")

def test_householder_matrix():
    """Test the householder_matrix function."""
    N = np.array([1, 0], dtype=float)
    result = householder.householder_matrix(N)
    expected = np.array([[-1, 0], [0, 1]], dtype=float)  
    assert np.allclose(result, expected), f"Test failed for householder_matrix: {result}"
    
    print("All householder_matrix tests passed successfully!")
    
def test_householder_vector():
    """Test the householder_vector function."""
    N = np.array([1, 0], dtype=float)
    X = np.array([2, 1], dtype=float)  
    expected = np.array([-2, 1], dtype=float)  

    result = householder.householder_vector(N, X)

    assert np.allclose(result, expected), f"Test failed for householder_vector: {result}"

    print("All householder_vector tests passed successfully!")


def test_householder_on_matrix():
    """Test the householder_on_matrix function."""
    U = np.array([1, 0], dtype=float)
    V = np.array([0, 1], dtype=float)
    M = np.array([[2, 4], [3, 5]], dtype=float)
    
    result = householder.householder_on_matrix(U, V, M)
    expected = np.array([[3, 5], [2, 4]], dtype=float)  
    assert np.allclose(result, expected), f"Test failed for householder_on_matrix: {result}"

    print("All householder_on_matrix tests passed successfully!")


test_projection()
test_householder_matrix()
test_householder_vector()
test_householder_on_matrix()
print("All tests passed!")
