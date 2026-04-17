import numpy as np

def newton_raphson(f, J, U0, N=100, epsilon=1e-10, alpha=0.1, beta=0.5, min_t = 1e-8):
    """
    Implement the Newton-Raphson algorithm with backtracking to find a root of the equation f(U) = 0

    Parameters: 
    f : function f(U) whose root is sought
    J : Jacobian matrix of f
    U0 : Initial guess
    N : Maximal number of iterations 
    epsilon : Convergence tolerance
    alpha : Backtracking parameter (acceptance condition)
    beta : Backtracking parameter (step size reduction)
    Returns: 
    An approximate solution of f(U) = 0, or None if convergence fails
    """

    U = np.array(U0, dtype=float)
    best_U, best_norm = None, float('inf')
    for _ in range(N):
        F_val = f(U)
        norm = np.linalg.norm(F_val)

        if norm < epsilon:
            return U

        if norm < best_norm:
            best_U, best_norm = U.copy(), norm

        try:
            H = J(U)
            V = np.linalg.solve(H, -F_val)
        except np.linalg.LinAlgError:
            break

        t = 1.0
        for _ in range(50):
            new_U = U + t * V
            if np.linalg.norm(f(new_U)) <= (1 - alpha * t) * norm:
                U = new_U
                break
            t *= beta
        else:
            break

    return best_U if best_norm < 10 * epsilon else None
