import numpy as np
from scipy.sparse import random as sparse_random
import scipy.sparse as sp

# ----------------------- Complete Factorization -----------------------

# ------- Question 1 -------


def complete_cholesky(A):
    """Performs Cholesky decomposition on a symmetric positive-definite matrix.

    Args:
        A (numpy.ndarray): A symmetric positive-definite matrix of shape (n, n).

    Returns:
        numpy.ndarray: A lower triangular matrix T such that A = T @ T.T.

    Raises:
        ValueError: If A is not symmetric or not positive definite.
    """
    if not np.allclose(A, A.T):  # Check if A is symmetric
        raise ValueError("Matrix is not symmetric.")

    # Check if the matrix is positive definite
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not positive definite.")
    
    n = A.shape[0]
    T = np.zeros(np.shape(A))

    for i in range(n):
        T[i, i] = np.sqrt(A[i, i] - np.sum(T[i, :i] ** 2))

        for j in range(i + 1, n):
            T[j, i] = (A[j, i] - np.sum(T[i, :i] * T[j, :i])) / T[i, i]

    return T


# Time Complexity : O(1/3 * (n^3))

# ------- Question 2 -------

# The complexity of solving the linear system A.x = b consists of:
# - O(n^3) for Cholesky decomposition
# - O(n^2) for solving T.y = b and (T.T).x = y (T.T represents the transpose of T)
# - Therefore, the total complexity for solving the system is O(n^3).
# - However, if the Cholesky decomposition is already computed, the remaining complexity is only O(n^2).


# ----------------------- Incomplete Factorization  -----------------------

# ------- Question 1 -------


import numpy as np

def generate_sparse_matrix(n, density, low, high):
    """Generates a random sparse symmetric matrix.

    Args:
        n (int): The dimension of the square matrix.
        density (float): The density of nonzero elements.
        low (int): The minimum value of random elements.
        high (int): The maximum value of random elements.

    Returns:
        numpy.ndarray: A sparse symmetric matrix with integer values.
    """
    # Initialize an empty matrix
    M = np.zeros((n, n), dtype=int)

    # Maximum number of nonzero elements 
    num_nonzero = int(density * (n * (n - 1) / 2))

    indices = set()
    while len(indices) < num_nonzero:
        i, j = np.random.randint(0, n, size=2)
        if i != j and (i, j) not in indices:
            value = np.random.randint(low, high + 1)
            M[i, j] = value
            M[j, i] = value  
            indices.add((i, j))

    np.fill_diagonal(M, 1)
    
    M = np.clip(M, low, high)

    return M



# sparse_matrix = generate_sparse_matrix(5, 5, density=0.2, low=1, high=10)
# print(sparse_matrix)
# print("Lignes", sparse_matrix.row)
# print("Colones", sparse_matrix.col)
# print("Valeurs", sparse_matrix.data)
# print(sparse_matrix.toarray())


# ------- Question 2 -------


def incomplete_cholesky(A):
    """Performs incomplete Cholesky factorization on a symmetric matrix.

    Args:
        A (numpy.ndarray): A symmetric matrix of shape (n, n).

    Returns:
        numpy.ndarray: A lower triangular matrix T as an approximation of Cholesky decomposition.
    """

    n = A.shape[0]
    T = np.zeros(np.shape(A))

    for i in range(n):
        if not (A[i, i]):  # Skip if diagonal element is zero
            continue
        T[i, i] = np.sqrt(A[i, i] - np.sum(T[i, :i] ** 2))

        for j in range(i + 1, n):
            if not (A[j, i]):  # Skip if the element is zero
                continue
            T[j, i] = (A[j, i] - np.sum(T[i, :i] * T[j, :i])) / T[i, i]

    return T


# ------- Question 3 -------

# The worst-case complexity is the same as standard Cholesky factorization (O(n^3)).
# However, since incomplete Cholesky skips zero elements, it can be significantly faster for sparse matrices.

# ------- Question 4 -------


def is_well_conditioned(A, fact):
    """Checks if the Cholesky factorization results in a well-conditioned matrix.

    Args:
        A (numpy.ndarray): The input matrix to be factorized.
        fact (int): 0 for complete Cholesky, 1 for incomplete Cholesky.

    Returns:
        bool: True if the condition number improves, False otherwise.
    """
    if not (fact):
        M = complete_cholesky(A)
    else:
        M = incomplete_cholesky(A)

    res = np.linalg.inv(M).T @ np.linalg.inv(M)

    return np.linalg.cond(res) < np.linalg.cond(A)
