import numpy as np
from numpy import linalg as LA
from scipy.linalg import solve_triangular
import partie1

# ------- Question 1 -------

"""Le premier problème est la division par p.T * Ap. Si Ap est mal conditionné, on aura une division par 0
Le deuxième de problème est l'absence de vérification des hypothèses sur A"""

# ------- Question 2 -------

"""Le gradient conjugé ne fait pas de produit matrice-matrice à l'image de la décomposition de Cholesky complète ou incomplète, d'où la complexité s'améliore passant de O(n^3) à O(n^2).
Pour plus de rigueur la compléxité du Gradient conjugué est O(kn^2) où k est le conditionnement de A. Mais cela conserve la complexité quadratique de la méthode """


# ------- Question 3 -------


def conjugate_gradient(A, B, X):
    """Solves the linear system A.X = B using the conjugate gradient method.

    Args:
        A (numpy.ndarray): Symmetric positive-definite matrix.
        B (numpy.ndarray): Right-hand side vector.
        X (numpy.ndarray): Initial guess for the solution.

    Returns:
        numpy.ndarray: Approximate solution to A.X = B.
    """
    B = B.reshape(-1, 1)
    X = X.reshape(-1, 1)
    R = B - A @ X
    P = R
    rsold = float(R.T @ R)
    for i in range(len(B)):
        AP = A @ P
        denom = float(P.T @ AP)
        if abs(denom) < 1e-10:  # Avoid division by zero
            break
        alpha = rsold / denom
        X = X + alpha * P
        R = R - alpha * AP
        rsnew = R.T @ R
        if np.sqrt(rsnew) < 1e-10:  # Convergence check
            break
        P = R + (rsnew / rsold) * P
        rsold = rsnew
    return X.ravel()


# ------- Question 4/5 -------


def preconditioned_conjugate_gradient(A, B, X, eps, Nmax):
    """Solves the linear system A.X = B using the preconditioned conjugate gradient method.

    Args:
        A (numpy.ndarray): Symmetric positive-definite matrix.
        B (numpy.ndarray): Right-hand side vector.
        X (numpy.ndarray): Initial guess for the solution.
        eps (float): Convergence tolerance.
        Nmax (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Approximate solution to A.X = B.
    """
    R = B - A @ X
    k = 0
    rsold = R.T @ R
    L = partie1.incomplete_cholesky(A)
    while ((LA.norm(R, np.inf) / LA.norm(B, np.inf)) > eps) and (k <= Nmax):
        Y = solve_triangular(L, R, lower=True)
        Z = solve_triangular(L.T, Y, lower=False)
        rsnew = R.T @ Z
        if k == 0:
            P = R
        else:
            gamma = rsnew / rsold
            P = R + gamma * P
        AP = A @ P
        alpha = rsnew / (P.T @ AP)
        X = X + alpha * P
        R = R - alpha * AP
        rsold = rsnew
    return X
