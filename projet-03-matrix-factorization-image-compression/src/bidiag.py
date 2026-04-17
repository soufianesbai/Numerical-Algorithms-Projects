import numpy as np
import householder 

def sign(y):
    """ 
    Returns the sign of y. 

    Args:
        y (float): A scalar value.

    Returns:
        float: -1.0 if y is negative, 1.0 if y is positive or zero.

    Notes:
        - This function ensures numerical stability by explicitly returning floating-point values.
    """
    return -1.0 if y < 0 else 1.0
    
def bidiagonalize_matrix(A):
    """ 
    Performs the bidiagonalization of matrix A using Householder transformations.

    Args:
        A (numpy.ndarray): A real matrix of shape (n, m).

    Returns:
        tuple: (Q_left, BD, Q_right)
            - Q_left (numpy.ndarray): An orthogonal matrix of shape (n, n).
            - BD (numpy.ndarray): A bidiagonal matrix of shape (n, m).
            - Q_right (numpy.ndarray): An orthogonal matrix of shape (m, m).

    Notes:
        - The function constructs BD using Householder reflections applied from the left and right.
        - The decomposition satisfies the relation: A = Q_left * BD * Q_right.
        - Q_left and Q_right are orthogonal matrices such that Q_left.T @ Q_left = I and Q_right.T @ Q_right = I.
    """
    n = A.shape[0]
    m = A.shape[1]
    Q_left = np.eye(n) # Initialize left orthogonal matrix
    Q_right = np.eye(m) # Initialize right orthogonal matrix
    BD = A.copy() # Copy of A to modify into a bidiagonal form
    
    for i in range(min(n,m)):
        # Step 1: Apply Householder reflection from the left
        if i < m - 1:
            X = BD[i:,i] # Extract column vector
            alpha = sign(X[0]) * np.linalg.norm(X)
            Y = np.zeros(n - i)
            Y[0] = alpha
            U = householder.projection(X,Y) # Compute Householder vector
            H_left = householder.householder_matrix(U) # Generate Householder matrix
            
            # Apply transformation to BD and update Q_left
            BD[i:,i:] = np.dot(H_left,BD[i:,i:])
            Q_left[:,i:] = np.dot(Q_left[:,i:],H_left)
            
        # Step 2: Apply Householder reflection from the right
        if i < n - 2:
            X = BD[i, i + 1:] # Extract row vector
            alpha = sign(X[0]) * np.linalg.norm(X)
            Y = np.zeros(m - i - 1)
            Y[0] = alpha
            U = householder.projection(X,Y) # Compute Householder vector
            H_right = householder.householder_matrix(U) # Generate Householder matrix
            
            # Apply transformation to BD and update Q_right
            BD[i,i + 1:] = Y
            BD[i + 1:,i + 1:] = np.dot(BD[i + 1:,i + 1:],H_right)
            Q_right[i + 1:,:] = np.dot(H_right,Q_right[i + 1:,:])
            
        # Invariant: At each iteration, we must have Q_left @ BD @ Q_right ≈ A
        assert np.allclose(Q_left @ BD @ Q_right, A, atol=1e-10), \
            f"Invariant violated at step {i}"
                
    return Q_left, BD, Q_right
