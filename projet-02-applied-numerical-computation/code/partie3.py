import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import partie2


def laplacian(N, h):
    # Generates the Laplacian matrix A of size N^2 x N^2.

    # Diagonal matrix with -4 on the main diagonal representing the main diagonal of the Laplacian matrix
    diagonal = -4 * np.ones(N**2)

    # Diagonal matrix of 1 for adjacent elements
    off_diagonal1 = np.ones(N**2 - 1)
    off_diagonal2 = np.ones(N**2 - 1)

    # Set to 0 for the elements on the borders of each row
    for i in range(1, N):
        off_diagonal1[i * N - 1] = 0

    # Constructing the sparse matrix with the diagonals
    A = sp.diags(
        [diagonal, off_diagonal1, off_diagonal1, off_diagonal2, off_diagonal2],
        [0, -1, 1, -N, N],
        shape=(N**2, N**2),
    )

    # print(f"Laplacian Matrix for N={N}:")
    # print(A.toarray())
    # print("\n")

    return A


def solve_heat_equation(F, N):
    """
    Parameters:
        F (numpy.ndarray): The right-hand side term representing heat sources.
        N (int): The number of interior grid points along one axis.
        
    Returns:
        numpy.ndarray: The computed temperature field.
    """
    # Grid spacing
    h = 1 / (N + 1)

    # Laplacian matrix
    A = laplacian(N, h)

    T = np.zeros(N * N)

    # Solve the linear system using the conjugate gradient method
    T = partie2.conjugate_gradient(A, F, T)

    # Reshape the solution to a N x N matrix
    T = T.reshape((N, N))

    return T
