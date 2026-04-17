import numpy as np

def projection(U,V):
    """Computes the normal vector N for the Householder transformation that maps U to V.

    Args:
        U (numpy.ndarray): A column vector of shape (n, 1).
        V (numpy.ndarray): A column vector of shape (n, 1), with the same norm as U.

    Returns:
        numpy.ndarray: A unit normal vector N used to construct the Householder matrix.

    Notes:
        - If U and V are identical, the function returns a zero vector.
        - The function avoids division by zero when U and V are identical.
    """
    diff = U - V
    norm_diff = np.linalg.norm(diff)
    
    if norm_diff == 0:
        return np.zeros_like(U) # Special case: U == V, no transformation needed.
    else:
        return (diff / norm_diff)


def householder_matrix(N):
    """Constructs the Householder matrix H using the normal vector N.

    Args:
        N (numpy.ndarray): A column vector of shape (n, 1) representing the Householder normal.

    Returns:
        numpy.ndarray: The Householder matrix of shape (n, n).

    Notes:
        - The Householder matrix is given by H = I - 2 * N * N.T.
        - This matrix reflects a vector across the hyperplane orthogonal to N.
    """
    return np.eye(len(N)) - 2 * np.outer(N, N)

def householder_vector(N,X):
    """Applies the Householder transformation to a given vector X.

    Args:
        N (numpy.ndarray): A column vector of shape (n, 1) representing the Householder normal.
        X (numpy.ndarray): A column vector of shape (n, 1) to be transformed.

    Returns:
        numpy.ndarray: The transformed vector H(X), where H is the Householder matrix.

    Notes:
        - This function optimizes the computation by avoiding explicit matrix storage.
        - Uses the formula: H(X) = X - 2 * (N.T @ X) * N.
    """
    return X - 2 * np.dot(N, X) * N 

def householder_on_matrix(U,V,M):
    """Applies the Householder transformation to each column of a given matrix.

    Args:
        U (numpy.ndarray): A column vector of shape (n, 1).
        V (numpy.ndarray): A column vector of shape (n, 1), with the same norm as U.
        M (numpy.ndarray): A matrix of shape (n, m) whose columns will be transformed.

    Returns:
        numpy.ndarray: The transformed matrix H(M), where H is the Householder matrix.

    Notes:
        - Instead of computing H explicitly, we apply the transformation to each column.
        - Complexity is reduced to O(nm) compared to O(n^2 m) for a full matrix product.
    """
    m = M.shape[1]
    M_projected = M.copy()
    N = projection(U, V) 
    for i in range(m): 
        M_projected[:, i] = householder_vector(N, M[:, i])
    return M_projected


